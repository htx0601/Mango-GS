from typing import Any, Mapping

import numpy as np
import pytorch3d.ops
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.deform_utils import cal_arap_error, cal_connectivity_from_points, arap_deformation_loss

try:
    from torch_batch_svd import svd
    print('Using speed up torch_batch_svd!')
except ImportError:
    svd = torch.svd
    print('Use original torch svd!')



def standardize_quaternion(quaternions: torch.Tensor) -> torch.Tensor:
    return torch.where(quaternions[..., 0:1] < 0, -quaternions, quaternions)


def quaternion_raw_multiply(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    aw, ax, ay, az = torch.unbind(a, -1)
    bw, bx, by, bz = torch.unbind(b, -1)
    ow = aw * bw - ax * bx - ay * by - az * bz
    ox = aw * bx + ax * bw + ay * bz - az * by
    oy = aw * by - ax * bz + ay * bw + az * bx
    oz = aw * bz + ax * by - ay * bx + az * bw
    return torch.stack((ow, ox, oy, oz), -1)


def quaternion_multiply(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    ab = quaternion_raw_multiply(a, b)
    return standardize_quaternion(ab)


def quaternion_to_matrix(quaternions: torch.Tensor) -> torch.Tensor:
    r, i, j, k = torch.unbind(quaternions, -1)
    two_s = 2.0 / (quaternions * quaternions).sum(-1)
    o = torch.stack(
        (
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ),
        -1,
    )
    return o.reshape(quaternions.shape[:-1] + (3, 3))


def _sqrt_positive_part(x: torch.Tensor) -> torch.Tensor:
    """
    Returns torch.sqrt(torch.max(0, x))
    but with a zero subgradient where x is 0.
    """
    ret = torch.zeros_like(x)
    positive_mask = x > 0
    ret[positive_mask] = torch.sqrt(x[positive_mask])
    return ret


def matrix_to_quaternion(matrix: torch.Tensor) -> torch.Tensor:
    """
    Convert rotations given as rotation matrices to quaternions.

    Args:
        matrix: Rotation matrices as tensor of shape (..., 3, 3).

    Returns:
        quaternions with real part first, as tensor of shape (..., 4).
    """
    if matrix.size(-1) != 3 or matrix.size(-2) != 3:
        raise ValueError(f"Invalid rotation matrix shape {matrix.shape}.")

    batch_dim = matrix.shape[:-2]
    m00, m01, m02, m10, m11, m12, m20, m21, m22 = torch.unbind(
        matrix.reshape(batch_dim + (9,)), dim=-1
    )

    q_abs = _sqrt_positive_part(
        torch.stack(
            [
                1.0 + m00 + m11 + m22,
                1.0 + m00 - m11 - m22,
                1.0 - m00 + m11 - m22,
                1.0 - m00 - m11 + m22,
            ],
            dim=-1,
        )
    )

    # we produce the desired quaternion multiplied by each of r, i, j, k
    quat_by_rijk = torch.stack(
        [
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([q_abs[..., 0] ** 2, m21 - m12, m02 - m20, m10 - m01], dim=-1),
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([m21 - m12, q_abs[..., 1] ** 2, m10 + m01, m02 + m20], dim=-1),
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([m02 - m20, m10 + m01, q_abs[..., 2] ** 2, m12 + m21], dim=-1),
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([m10 - m01, m20 + m02, m21 + m12, q_abs[..., 3] ** 2], dim=-1),
        ],
        dim=-2,
    )

    # We floor here at 0.1 but the exact level is not important; if q_abs is small,
    # the candidate won't be picked.
    flr = torch.tensor(0.1).to(dtype=q_abs.dtype, device=q_abs.device)
    quat_candidates = quat_by_rijk / (2.0 * q_abs[..., None].max(flr))

    # if not for numerical problems, quat_candidates[i] should be same (up to a sign),
    # forall i; we pick the best-conditioned one (with the largest denominator)

    return quat_candidates[
        torch.nn.functional.one_hot(q_abs.argmax(dim=-1), num_classes=4) > 0.5, :
    ].reshape(batch_dim + (4,))


def get_embedder(multires, i=1):
    if i == -1:
        return nn.Identity(), 3

    embed_kwargs = {
        'include_input': True,
        'input_dims': i,
        'max_freq_log2': multires - 1,
        'num_freqs': multires,
        'log_sampling': True,
        'periodic_fns': [torch.sin, torch.cos],
    }

    embedder_obj = Embedder(**embed_kwargs)
    embed = lambda x, eo=embedder_obj: eo.embed(x)
    return embed, embedder_obj.out_dim


class Embedder:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.create_embedding_fn()

    def create_embedding_fn(self):
        embed_fns = []
        d = self.kwargs['input_dims']
        out_dim = 0
        if self.kwargs['include_input']:
            embed_fns.append(lambda x: x)
            out_dim += d

        max_freq = self.kwargs['max_freq_log2']
        N_freqs = self.kwargs['num_freqs']

        if self.kwargs['log_sampling']:
            freq_bands = 2. ** torch.linspace(0., max_freq, steps=N_freqs)
        else:
            freq_bands = torch.linspace(2. ** 0., 2. ** max_freq, steps=N_freqs)

        for freq in freq_bands:
            for p_fn in self.kwargs['periodic_fns']:
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq: p_fn(x * freq))
                out_dim += d

        self.embed_fns = embed_fns
        self.out_dim = out_dim

    def embed(self, inputs):
        return torch.cat([fn(inputs) for fn in self.embed_fns], -1)


class ProgressiveBandFrequency(nn.Module):
    def __init__(self, in_channels: int, n_frequencies=12, no_masking_step=5000):
        super().__init__()
        self.N_freqs = n_frequencies
        self.in_channels, self.n_input_dims = in_channels, in_channels
        self.funcs = [torch.sin, torch.cos]
        self.freq_bands = 2 ** torch.linspace(0, self.N_freqs - 1, self.N_freqs)
        self.n_output_dims = self.in_channels * (len(self.funcs) * self.N_freqs)
        self.n_masking_step = no_masking_step
        self.cur_step = nn.Parameter(torch.tensor(-1), requires_grad=False)
        self.update_step(0)

    def forward(self, x):
        out = []
        for freq, mask in zip(self.freq_bands, self.mask):
            for func in self.funcs:
                out += [func(freq * x) * mask]
        return torch.cat(out, -1)

    def update_step(self, global_step):
        if global_step > self.cur_step.item():
            if self.n_masking_step <= 0 or global_step is None or not self.training:
                self.mask = torch.ones(self.N_freqs, dtype=torch.float32, device=torch.device("cuda:0"))
            else:
                self.mask = (1.0 - torch.cos(torch.pi* (global_step / self.n_masking_step * self.N_freqs - torch.arange(0, self.N_freqs, device=torch.device("cuda:0"))).clamp(0, 1))) / 2.0
                # print(f"Update mask of Freq: {global_step}/{self.n_masking_step} {self.mask}")
            self.cur_step.data = torch.ones_like(self.cur_step) * global_step


def farthest_point_sample(xyz, npoint):
    """
    Input:
        xyz: pointcloud data, [B, N, C]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [B, npoint]
    """
    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    distance = torch.ones(B, N).to(device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
    batch_indices = torch.arange(B, dtype=torch.long).to(device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, C)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]
    return centroids


def landmark_interpolate(landmarks, steps, step, interpolation='log'):
    stage = (step >= np.array(steps)).sum()
    if stage == len(steps):
        return max(0, landmarks[-1])
    elif stage == 0:
        return 0
    else:
        ldm1, ldm2 = landmarks[stage-1], landmarks[stage]
        if ldm2 <= 0:
            return 0
        step1, step2 = steps[stage-1], steps[stage]
        ratio = (step - step1) / (step2 - step1)
        if interpolation == 'log':
            return np.exp(np.log(ldm1) * (1 - ratio) + np.log(ldm2) * ratio)
        elif interpolation == 'linear':
            return ldm1 * (1 - ratio) + ldm2 * ratio
        else:
            print(f'Unknown interpolation type: {interpolation}')
            raise NotImplementedError


class LearnedKNNMetric(nn.Module):
    def __init__(self, in_q, in_k, shared=True):
        super().__init__()

        def make_net(dim):
            return nn.Sequential(
                nn.Linear(dim, 64),
                nn.ReLU(inplace=True),
                nn.Linear(64, dim),
            )

        self.f = make_net(in_q)
        self.g = self.f if shared and in_q == in_k else make_net(in_k)
        for net in (self.f, self.g):
            for layer in net:
                if isinstance(layer, nn.Linear):
                    nn.init.kaiming_uniform_(layer.weight, nonlinearity='relu')
                    nn.init.zeros_(layer.bias)

    def forward(self, q_in, k_in):
        return self.f(q_in), self.g(k_in)


class MultiFrameDeformNetwork(nn.Module):
    def __init__(self, D=8, W=256, input_ch=3, output_ch=59, T=4, t_multires=6, multires=10,
                 is_blender=False, local_frame=False, pred_opacity=False, pred_color=False, resnet_color=True,
                 hash_color=False, color_wrt_dir=False, progressive_brand_time=False, max_d_scale=-1,
                 use_tcnn=False, **kwargs):
        super(MultiFrameDeformNetwork, self).__init__()
        if use_tcnn:
            raise ValueError('The public release does not include the experimental tiny-cuda-nn MLP path.')
        if pred_color and hash_color and not resnet_color:
            raise ValueError('The public release does not include the experimental hash-color branch.')

        self.name = 'mlp_video'
        self.is_blender = is_blender
        self.D, self.W, self.T = D, W, T
        self.local_frame = local_frame
        self.pred_opacity = pred_opacity
        self.pred_color = pred_color
        self.resnet_color = resnet_color
        self.hash_color = (not resnet_color) and hash_color
        self.color_wrt_dir = color_wrt_dir
        self.max_d_scale = max_d_scale
        self.use_tcnn = use_tcnn

        self.skips = [4]
        self.attn_id = [i * 2 for i in range(1, D // 2 - 1)]
        self.temporal_attn_type = 'transformer'

        self.use_time_robust_mod = False
        self.time_rel = True
        self.time_film_rank = min(64, self.W // 2)
        self.time_dropout_base_p = 0.2
        self.register_buffer('global_step', torch.zeros((), dtype=torch.long))
        self.register_buffer('max_step_for_dropout', torch.tensor(20000, dtype=torch.long))

        self.progressive_brand_time = progressive_brand_time
        self.t_multires = 6 if is_blender else 10
        if self.progressive_brand_time:
            self.embed_time_fn = ProgressiveBandFrequency(in_channels=1, n_frequencies=self.t_multires)
            time_input_ch = self.embed_time_fn.n_output_dims
        else:
            self.embed_time_fn, time_input_ch = get_embedder(self.t_multires, 1)
        self.embed_fn, xyz_input_ch = get_embedder(multires, 3)

        self.input_ch = xyz_input_ch if self.use_time_robust_mod else (xyz_input_ch + time_input_ch)

        self.reg_loss = 0.

        def make_linear(ic, oc):
            return nn.Linear(ic, oc)

        if self.is_blender:
            self.time_out = 30
            self.timenet = nn.Sequential(
                nn.Linear(time_input_ch, 256), nn.ReLU(inplace=True),
                nn.Linear(256, self.time_out))

            self.linear = nn.ModuleList(
                [make_linear(xyz_input_ch + self.time_out, W)] + [
                    make_linear(W, W) if i not in self.skips else make_linear(W + xyz_input_ch + self.time_out, W)
                    for i in range(D - 1)]
            )

        else:
            self.linear = nn.ModuleList(
                [make_linear(self.input_ch, W)] + [
                    make_linear(W, W) if i not in self.skips else make_linear(W + self.input_ch, W)
                    for i in range(D - 1)]
            )
            self.time_bottleneck = nn.Sequential(
                make_linear(time_input_ch, self.time_film_rank), nn.ReLU(inplace=True),
                make_linear(self.time_film_rank, self.time_film_rank), nn.ReLU(inplace=True),
            )
            self.film_layers = nn.ModuleList([
                nn.Linear(self.time_film_rank, 2 * self.W) for _ in range(self.D)
            ])
            for lin in self.film_layers:
                nn.init.zeros_(lin.weight)
                nn.init.zeros_(lin.bias)

        self.attn_heads = 2
        self.d_attn = self.W

        if self.temporal_attn_type == 'transformer':
            self.temporal_attn = nn.ModuleList([])
            for _ in self.attn_id:
                block = nn.ModuleDict({
                    "proj_in": nn.Linear(self.W, self.d_attn, bias=False),
                    "mha": nn.MultiheadAttention(embed_dim=self.d_attn,
                                                 num_heads=self.attn_heads,
                                                 batch_first=True),
                    "proj_gate":  nn.Linear(self.d_attn, self.W, bias=True),
                    "proj_delta": nn.Linear(self.d_attn, self.W, bias=True),
                })
                nn.init.zeros_(block["proj_delta"].weight)
                nn.init.zeros_(block["proj_delta"].bias)
                nn.init.zeros_(block["proj_gate"].weight)
                nn.init.constant_(block["proj_gate"].bias, -3.0)
                self.temporal_attn.append(block)

        else:
            self.temporal_attn = nn.ModuleList()
            for _ in self.attn_id:
                block = nn.Sequential(
                    make_linear(self.T, 64),
                    nn.ReLU(),
                    make_linear(64, self.T),
                )
                self.temporal_attn.append(block)

        self.is_blender = is_blender

        self.gaussian_warp = nn.Sequential(nn.Linear(W, W), nn.ReLU(inplace=True), nn.Linear(W, 3))
        self.gaussian_scaling = nn.Sequential(nn.Linear(W, W), nn.ReLU(inplace=True), nn.Linear(W, 3))
        self.gaussian_rotation = nn.Sequential(nn.Linear(W, W), nn.ReLU(inplace=True), nn.Linear(W, 4))

        self.local_frame = local_frame
        if self.local_frame:
            self.local_rotation = nn.Linear(W, 4)
            nn.init.normal_(self.local_rotation.weight, mean=0, std=1e-4)
            nn.init.zeros_(self.local_rotation.bias)

        for layer in self.linear:
            if hasattr(layer, 'weight'):
                nn.init.kaiming_uniform_(layer.weight, mode='fan_in', nonlinearity='relu')
            if hasattr(layer, 'bias'):
                nn.init.zeros_(layer.bias)

        self._init_head(self.gaussian_warp, last_std=1e-5)
        self._init_head(self.gaussian_scaling, last_std=1e-8)
        self._init_head(self.gaussian_rotation, last_std=1e-5)

        if self.pred_opacity:
            self.gaussian_opacity = nn.Sequential(nn.Linear(W, W), nn.ReLU(inplace=True), nn.Linear(W, 1))
            self._init_head(self.gaussian_opacity, last_std=1e-5)
        if self.pred_color:
            if self.resnet_color:
                in_dim = xyz_input_ch + W if self.color_wrt_dir else self.linear[0].weight.shape[
                                                                         -1] + W  # Color depends on Direction or Position-Time
                self.gaussian_color = nn.Sequential(nn.Linear(in_dim, W), nn.ReLU(), nn.Linear(W, W), nn.ReLU(),
                                                    nn.Linear(W, 3))
                self._init_head(self.gaussian_color, last_std=1e-5)
            else:
                self.gaussian_color = nn.Sequential(nn.Linear(W, W), nn.ReLU(inplace=True), nn.Linear(W, 3))
                self._init_head(self.gaussian_color, last_std=1e-5)

    def trainable_parameters(self):
        return [{'params': list(self.parameters()), 'name': 'mlp'}]

    def _init_head(self, head: nn.Sequential, last_std: float):
        """Initialize Linear -> ReLU -> Linear heads:
           - hidden Linear layers: kaiming_uniform for ReLU
           - final Linear layer: N(0, last_std)
           - all biases: zero
        """
        linear_layers = [m for m in head if isinstance(m, nn.Linear)]
        if len(linear_layers) == 0:
            return
        for lin in linear_layers[:-1]:
            nn.init.kaiming_uniform_(lin.weight, mode='fan_in', nonlinearity='relu')
            if lin.bias is not None:
                nn.init.zeros_(lin.bias)
        last = linear_layers[-1]
        nn.init.normal_(last.weight, mean=0.0, std=last_std)
        if last.bias is not None:
            nn.init.zeros_(last.bias)

    def _process_time_embedding(self, t, is_multiframe=True):
        """
        t: (N*T,1) or (1,1) after view. It is used to produce t_emb and then reshaped back to (N,T,C).
        Returns processed t_emb as (N,T,C), or (N,C) for single-frame input.
        """
        t_emb = self.embed_time_fn(t)  # (N*T,C) or (N,C)

        if self.training and self.use_time_robust_mod:
            step = int(self.global_step.item())
            maxs = int(self.max_step_for_dropout.item())
            p = float(self.time_dropout_base_p) * max(0.0, 1.0 - step / maxs)
            if p > 1e-6:
                C = t_emb.shape[-1]
                device = t_emb.device
                drop_mask = torch.ones(C, device=device, dtype=t_emb.dtype)
                high = torch.arange(C // 2, C, device=device)
                keep = (torch.rand_like(high.float()) > p).float()
                drop_mask[high] = keep
                t_emb = t_emb * drop_mask  # broadcast to the batch
        return t_emb

    def _film_from_time(self, t_emb_proc, N, T, is_multiframe: bool):
        """
        t_emb_proc: (N*T,C) or (N,C).
        Returns a length-D list of (gamma, beta) tensors aligned with h:
        multi-frame -> (N,T,W), single-frame -> (N,W).
        """
        if is_multiframe:
            t_feat = self.time_bottleneck(t_emb_proc).view(N, T, -1)  # (N,T,R)
        else:
            t_feat = self.time_bottleneck(t_emb_proc)  # (N,R)

        gb_list = []
        for i in range(self.D):
            proj = self.film_layers[i](t_feat)  # (N,T,2W) or (N,2W)
            if is_multiframe:
                gamma, beta = torch.chunk(proj, 2, dim=-1)  # (N,T,W)
            else:
                gamma, beta = torch.chunk(proj, 2, dim=-1)  # (N,W)
            gb_list.append((gamma, beta))
        return gb_list

    def forward(self, x, t, train_mask=None, **kwargs):
        N, input_T = x.shape[0], t.shape[1]
        is_input_multiframe = input_T == self.T
        t_emb = t.view(-1, 1) if is_input_multiframe else t
        x_emb = self.embed_fn(x)

        t_emb = self._process_time_embedding(t_emb, is_input_multiframe)
        if self.is_blender:
            t_emb = self.timenet(t_emb)  # better for D-NeRF Dataset

        if is_input_multiframe is True and train_mask is not None:
            t_emb[train_mask, :] = 0.0

        t_emb = t_emb.unsqueeze(0).expand(N, self.T, -1) if is_input_multiframe else t_emb.expand(N, -1)
        x_emb = x_emb.unsqueeze(1).expand(-1, self.T, -1) if is_input_multiframe else x_emb

        if not self.use_time_robust_mod:
            h = torch.cat([x_emb, t_emb], dim=-1)
            for i, l in enumerate(self.linear):
                h = F.relu(self.linear[i](h))

                if i in self.attn_id and is_input_multiframe:
                    j = self.attn_id.index(i)
                    if self.temporal_attn_type == 'transformer':
                        blk = self.temporal_attn[self.attn_id.index(i)]
                        h_in = blk["proj_in"](h)  # (N,T,W)->(N,T,d_attn)
                        h_attn, _ = blk["mha"](h_in, h_in, h_in)
                        gate = torch.sigmoid(blk["proj_gate"](h_attn))  # (N,T,W)
                        delta = blk["proj_delta"](h_attn)  # (N,T,W)
                        h = h * gate + delta
                    else:
                        h_attn = self.temporal_attn[j](h.permute(0, 2, 1))
                        h = h_attn.permute(0, 2, 1) + h

                if i in self.skips:
                    h = torch.cat([x_emb, t_emb, h], -1)
        else:
            h = x_emb

            film_gb = self._film_from_time(
                t_emb_proc=t_emb.reshape(-1, t_emb.shape[-1]) if is_input_multiframe else t_emb,
                N=N, T=self.T, is_multiframe=is_input_multiframe
            )

            for i, l in enumerate(self.linear):
                h = self.linear[i](h)
                h = F.relu(h)
                gamma, beta = film_gb[i]
                if is_input_multiframe:
                    h = h * (1.0 + gamma) + beta  # (N,T,W)
                else:
                    h = h * (1.0 + gamma.unsqueeze(1)) + beta.unsqueeze(1) if h.dim() == 3 else h * (1.0 + gamma) + beta

                if i in self.attn_id and is_input_multiframe and self.temporal_attn_type == 'transformer':
                    blk = self.temporal_attn[self.attn_id.index(i)]
                    h_in = blk["proj_in"](h)  # (N,T,W)->(N,T,d_attn)
                    h_attn, _ = blk["mha"](h_in, h_in, h_in)
                    gate = torch.sigmoid(blk["proj_gate"](h_attn))  # (N,T,W)
                    delta = blk["proj_delta"](h_attn)  # (N,T,W)
                    h = h + gate * delta
                    h = h.permute(0, 2, 1)

                if i in self.skips:
                    h = torch.cat([x_emb, h], -1)

        h = h.permute(1, 0, 2) if is_input_multiframe else h  # (T, N, )

        d_xyz = self.gaussian_warp(h)
        scaling = self.gaussian_scaling(h)
        rotation = self.gaussian_rotation(h)

        if self.max_d_scale > 0:
            scaling = torch.tanh(scaling) * np.log(self.max_d_scale)

        return_dict = {'d_xyz': d_xyz, 'd_rotation': rotation, 'd_scaling': scaling, 'hidden': h}
        if self.pred_opacity:
            return_dict['d_opacity'] = self.gaussian_opacity(h)
        else:
            return_dict['d_opacity'] = None
        if self.pred_color:
            if self.resnet_color:
                if self.color_wrt_dir:
                    if 'camera_center' in kwargs:
                        dir_pp = (x - kwargs['camera_center'].repeat(x.shape[0], 1))
                        dir_pp_normalized = dir_pp / dir_pp.norm(dim=1, keepdim=True)
                        return_dict['d_color'] = self.gaussian_color(
                            torch.cat([self.embed_fn(dir_pp_normalized), h], dim=-1))
                    else:
                        return_dict['d_color'] = None
                else:
                    return_dict['d_color'] = self.gaussian_color(torch.cat([x_emb, t_emb, h], dim=-1))
            else:
                return_dict['d_color'] = self.gaussian_color(h)
        else:
            return_dict['d_color'] = None
        if self.local_frame:
            return_dict['local_rotation'] = self.local_rotation(h)
        return return_dict

    def update(self, iteration, *args, **kwargs):
        if self.progressive_brand_time:
            self.embed_time_fn.update_step(iteration)
        self.global_step.data = torch.tensor(int(iteration), device=self.global_step.device)
        return


class MultiFrameControlNodeWarp(nn.Module):
    def __init__(self, is_blender, T=4, init_pcl=None, node_num=512, K=3, use_hash=False, hash_time=False,
                 enable_densify_prune=False, pred_opacity=False, pred_color=False, with_arap_loss=False,
                 with_node_weight=True, local_frame=False, d_rot_as_res=True, skinning=False, hyper_dim=2,
                 progressive_brand_time=False, max_d_scale=-1, is_scene_static=False, enable_learned_metric=False,
                 **kwargs):
        super().__init__()
        if use_hash or hash_time:
            raise ValueError("The public release supports only the Mango-GS MLP deformation path, not hash deformation.")
        self.T = T
        self.K = K
        self.use_hash = use_hash
        self.hash_time = hash_time
        self.enable_dp = enable_densify_prune
        self.name = 'video_node'
        self.with_node_weight = with_node_weight
        self.reg_loss = 0.
        self.local_frame = local_frame
        self.d_rot_as_res = d_rot_as_res
        self.hyper_dim = hyper_dim if not skinning else 0  # skinning should not be with hyper
        self.is_blender = is_blender
        self.pred_opacity = pred_opacity
        self.pred_color = pred_color
        self.max_d_scale = max_d_scale
        self.is_scene_static = is_scene_static

        self.use_dq = False

        self.skinning = skinning  # As skin model, discarding KNN weighting
        if with_arap_loss and not self.is_scene_static:
            self.lambda_arap_landmarks = [1e-4, 1e-4, 1e-5, 1e-5, 0]
            self.lambda_arap_steps = [0, 5000, 10000, 20000, 20001]
        else:
            self.lambda_arap_landmarks = [0]
            self.lambda_arap_steps = [0]

        self.network = MultiFrameDeformNetwork(is_blender=is_blender, T=self.T, local_frame=local_frame, pred_opacity=pred_opacity,
                                         pred_color=pred_color, progressive_brand_time=progressive_brand_time,
                                         max_d_scale=max_d_scale, use_tcnn=False).cuda()

        self.register_buffer('inited', torch.tensor(False))
        self.nodes = nn.Parameter(torch.randn(node_num, 3 + 3 + self.hyper_dim))
        if not self.skinning:
            self._node_radius = nn.Parameter(torch.randn(node_num))
            if self.with_node_weight:
                self._node_weight = nn.Parameter(torch.zeros_like(self.nodes[:, :1]), requires_grad=with_node_weight)
        if init_pcl is not None:
            self.init(init_pcl)

        self.nodes_color_visualization = torch.ones_like(self.nodes)

        self.cached_nn_weight = False
        self.nn_weight, self.nn_dist, self.nn_idxs = None, None, None

        self.node_code_dim = self.nodes.shape[1] - 3

        self.enable_learned_metric = enable_learned_metric
        self.metric = None
        if self.enable_learned_metric:
            self.metric = LearnedKNNMetric(
                in_q=3 + self.hyper_dim,  # gs: x(3)+feature(hyper_dim)
                in_k=3 + self.hyper_dim,  # node-key: x_proxy(3)+feature
                shared=True).cuda()

    def update(self, iteration):
        self.network.update(iteration)

    def trainable_parameters(self):
        deform_params = list(self.network.parameters())
        if self.metric is not None:
            deform_params += list(self.metric.parameters())
        if self.skinning:
            return [{'params': deform_params, 'name': 'deform'},
                    {'params': [self.nodes], 'name': 'nodes'}]
        elif self.with_node_weight:
            return [{'params': deform_params, 'name': 'deform'},
                    {'params': [self.nodes, self._node_radius, self._node_weight], 'name': 'nodes'}]
        else:
            return [{'params': deform_params, 'name': 'deform'},
                    {'params': [self.nodes, self._node_radius], 'name': 'nodes'}]

    @property
    def param_names(self):
        if self.skinning:
            param_names = ['nodes', 'deform']
        elif self.with_node_weight:
            param_names = ['nodes', '_node_radius', '_node_weight']
        else:
            param_names = ['nodes', '_node_radius']
        return param_names


    def load_state_dict(self, state_dict: Mapping[str, Any], strict: bool = True):
        param_names = self.param_names
        for key in state_dict:
            if key in param_names:
                node_param = state_dict[key]
                if getattr(self, key).shape != node_param.shape:
                    print(
                        f'Loading nodes mismatching the original setting: {getattr(self, key).shape} and {node_param.shape}')
                    setattr(self, key, nn.Parameter(node_param))
                else:
                    getattr(self, key).data = node_param
            elif key.startswith('gs_'):
                name = key[3:]
                try:
                    getattr(self.as_gaussians, name).data = state_dict[key]
                except:
                    print(f'Directly set as values for {key} when loading deform gaussians')
                    setattr(self.as_gaussians, name, state_dict[key])
        for key in param_names:
            if key in state_dict:
                state_dict.pop(key)
        super().load_state_dict(state_dict=state_dict, strict=False)

    def state_dict(self):
        state_dict = super().state_dict()
        if hasattr(self, 'gs') and self.gs is not None:
            for name in self.gs.param_names():
                state_dict['gs_' + name] = getattr(self.gs, name)
        return state_dict

    @property
    def node_radius(self):
        return torch.exp(self._node_radius)

    @property
    def node_weight(self):
        return torch.sigmoid(self._node_weight)

    @property
    def node_num(self):
        return self.nodes.shape[0]

    def init(self, opt, init_pcl, hyper_pcl=None, keep_all=False, force_init=False, as_gs_force_with_motion_mask=False,
             force_gs_keep_all=False, reset_bbox=True, **kwargs):
        if self.inited and not force_init:
            return
        self.inited.data = torch.ones_like(self.inited)
        self.register_buffer('inited', torch.tensor(True))
        if keep_all or self.node_num > init_pcl.shape[0]:
            self.nodes = nn.Parameter(
                torch.cat([init_pcl.float(), 1e-2 * torch.ones([init_pcl.shape[0], self.hyper_dim]).float().cuda()],
                          dim=-1))
            init_nodes_idx = None
            print('Initialization with all pcl. Need to reset the optimizer.')
        else:
            pcl_to_samp = init_pcl if hyper_pcl is None else hyper_pcl
            init_nodes_idx = farthest_point_sample(pcl_to_samp.detach()[None], self.node_num)[0]

            self.nodes.data = nn.Parameter(torch.cat(
                [init_pcl[init_nodes_idx].float(), init_pcl[init_nodes_idx].float(), 1e-2 * torch.ones([self.node_num, self.hyper_dim]).float().cuda()],
                dim=-1))
        scene_range = init_pcl.max() - init_pcl.min()
        if self.skinning:
            if 'feature' in kwargs:
                gs_weights = kwargs['feature']
                radius = .1 * scene_range + 1e-7
                initial_weights = - torch.log(
                    (init_pcl[:, None] - self.nodes[None, ..., :3]).square().sum(dim=-1) / radius ** 2)
                gs_weights.data = initial_weights
        else:
            if keep_all or self.node_num > init_pcl.shape[0]:
                self._node_radius = nn.Parameter(
                    torch.log(.1 * scene_range + 1e-7) * torch.ones([self.node_num]).float().to(scene_range.device))
                self._node_weight = nn.Parameter(torch.zeros_like(torch.zeros_like(self.nodes[:, :1])))
            else:
                self._node_radius.data = nn.Parameter(
                    torch.log(.1 * scene_range + 1e-7) * torch.ones([self.node_num]).float().to(scene_range.device))
                self._node_weight.data = torch.zeros_like(torch.zeros_like(self.nodes[:, :1]))
        self.gs = None
        if force_gs_keep_all:
            self.init_gaussians(init_pcl=init_pcl, with_motion_mask=as_gs_force_with_motion_mask)
        else:
            self.init_gaussians(init_pcl=self.nodes[..., :3], with_motion_mask=as_gs_force_with_motion_mask)
        self.as_gaussians.training_setup(opt)
        print(f'Control node initialized with {self.nodes.shape[0]} from {init_pcl.shape[0]} points.')
        return init_nodes_idx

    def expand_time(self, t):
        N = self.nodes.shape[0]
        t = t.unsqueeze(0).expand(N, -1)
        return t

    def cal_nn_weight(self, x: torch.Tensor, K=None, feature=None, gs_knn_feature=None, nodes=None, gs_kernel=True, temperature=1., learning_metric=False):
        if self.skinning:
            nn_weight = torch.softmax(feature, dim=-1)
            nn_idx = torch.arange(0, self.node_num, dtype=torch.long).cuda()
            return nn_weight, None, nn_idx
        else:
            if self.cached_nn_weight and self.nn_weight is not None:
                return self.nn_weight, self.nn_dist, self.nn_idxs
            else:
                K = self.K if K is None else K
                q = torch.cat([x.detach(), feature], dim=-1)
                k = self.nodes[..., 3:]

                if learning_metric and self.metric is not None:
                    q_, k_ = self.metric(q[None], k[None])
                    nn_dist, nn_idxs, _ = pytorch3d.ops.knn_points(q_, k_, K=K)
                else:
                    nn_dist, nn_idxs, _ = pytorch3d.ops.knn_points(q[None], k[None], None, None, K=K)
                nn_dist, nn_idxs = nn_dist[0], nn_idxs[0]
                if gs_kernel:
                    nn_radius = self.node_radius[nn_idxs]
                    nn_weight = torch.exp(- nn_dist / (2 * nn_radius ** 2))
                    if self.with_node_weight:
                        nn_node_weight = self.node_weight[nn_idxs]
                        nn_weight = nn_weight * nn_node_weight[..., 0]
                    nn_weight = nn_weight + 1e-7
                    nn_weight = nn_weight / nn_weight.sum(dim=-1, keepdim=True)
                    if self.cached_nn_weight:
                        self.nn_weight = nn_weight
                        self.nn_dist = nn_dist
                        self.nn_idxs = nn_idxs
                    return nn_weight, nn_dist, nn_idxs
                else:
                    nn_weight = torch.softmax(- nn_dist / temperature, dim=-1)
                    return nn_weight, nn_dist, nn_idxs

    def cal_nn_weight_floyd(self, x: torch.Tensor, t0: torch.Tensor, cur_node: torch.Tensor, K=None, GraphK=2,
                            temperature=1., cache_name='floyd', XisNode=False):
        if not hasattr(self, f'{cache_name}_nn_dist') or (
                t0 is not None and (getattr(self, f'{cache_name}_t') - t0).abs().max() > 1e-2):
            node_dist_mat = self.geodesic_distance_floyd(cur_node=cur_node, K=GraphK)
            floyd_nn_dist, floyd_nn_idx = node_dist_mat.sort(dim=1)
            offset = 1 if XisNode else 0
            floyd_nn_dist = floyd_nn_dist[:, offset:K + offset]
            floyd_nn_idx = floyd_nn_idx[:, offset:K + offset]
            setattr(self, f'{cache_name}_nn_dist', floyd_nn_dist)
            setattr(self, f'{cache_name}_nn_idx', floyd_nn_idx)
            if t0 is not None:
                setattr(self, f'{cache_name}_t', t0.clone())
        nn_dist, nn_idxs, _ = pytorch3d.ops.knn_points(x[None], cur_node[None], None, None, K=1)  # N, K
        nn_dist, nn_idxs = nn_dist[0, :, 0], nn_idxs[0, :, 0]  # N
        knn_dist, knn_idxs = getattr(self, f'{cache_name}_nn_dist')[nn_idxs] + nn_dist[:, None], \
        getattr(self, f'{cache_name}_nn_idx')[nn_idxs]
        knn_weight = torch.softmax(- knn_dist / temperature, dim=-1)
        return knn_weight, knn_dist, knn_idxs

    def query_network(self, x, t, train_mask=None, **kwargs):
        values = self.network(x=x, t=t, train_mask=train_mask, **kwargs)
        return values

    def node_deform(self, t, train_mask=None, detach_node=True, **kwargs):
        tshape = t.shape
        nodes = self.nodes[..., :3]
        if detach_node:
            nodes = nodes.detach()
        values = self.query_network(x=nodes, t=t, train_mask=train_mask, **kwargs)
        return values

    @torch.no_grad()
    def sample_node_deform(self, samp_num=512, sv_path='./deform'):
        t = torch.linspace(0, 1, samp_num).float().cuda()
        chunk = 16
        start = 0
        values = {}
        while start < samp_num:
            end = min(start + chunk, samp_num)
            t_ = t[None, start: end, None].expand(self.node_num, end - start, 1)
            values_ = self.node_deform(t_)
            for key in values_:
                if values_[key] is not None:
                    values_[key] = values_[key].permute(1, 0, 2)
                    if key not in values:
                        values[key] = values_[key]
                    else:
                        values[key] = torch.cat([values[key], values_[key]], dim=0)
            start = end
        values_np = {key: values[key].detach().cpu().numpy() for key in values if key != 'hidden'}
        np.savez(sv_path, **values_np, nodes=self.nodes.detach().cpu().numpy())
        print(f"Successfully save {values_np.keys()} into {sv_path}! Without hidden features!")

    def get_trajectory(self, t_samp_num=8):
        t_samp = torch.linspace(0, 1, t_samp_num).cuda()
        t_samp = t_samp[None, :, None].expand(self.node_num, t_samp_num, 1)  # M, T, 1
        node_deform = self.node_deform(t=t_samp)
        trajectory = self.nodes[:, None, :3].detach() + node_deform['d_xyz']  # M, T, 3
        for key in node_deform:
            node_deform[key] = node_deform[key][:, 0] if node_deform[key] is not None else None
        return trajectory.detach(), node_deform

    def arap_loss_with_rot(self, t_samp_num=8):
        t_samp = torch.rand(t_samp_num).cuda()
        t_samp = t_samp[None, :, None].expand(self.node_num, t_samp_num, 1)  # M, T, 1
        node_deform = self.node_deform(t=t_samp)
        trajectory = self.nodes[:, None, :3].detach() + node_deform['d_xyz']  # M, T, 3
        trajectory_rot = node_deform['d_rotation'] if not self.d_rot_as_res else None
        arap_error, rot_error = arap_deformation_loss(trajectory=trajectory, node_radius=self.node_radius.detach(),
                                                      trajectory_rot=trajectory_rot, with_rot=not self.d_rot_as_res)
        return arap_error + rot_error

    def p2dR(self, p, p0=None, K=8, as_quat=True, mode='trajectory', t0=None):
        p = p.detach()
        nodes = self.nodes[..., :3].detach()
        if mode == 'trajectory':
            trajectory, t0_deform = self.get_trajectory(t_samp_num=4)
            t0_nodes = trajectory[:, 0] if p0 is None else p0
            trajectory = trajectory.reshape([trajectory.shape[0], -1])
            nn_dist, nn_idx, _ = pytorch3d.ops.knn_points(trajectory[None], trajectory[None], None, None, K=K + 1,
                                                          return_nn=False)
            nn_dist, nn_idx = nn_dist[0, :, 1:], nn_idx[0, :, 1:]
            nn_weight = torch.softmax(nn_dist / nn_dist.mean(), dim=-1)
            edges = torch.gather(t0_nodes[:, None].expand([nodes.shape[0], K, nodes.shape[-1]]), dim=0,
                                 index=nn_idx[..., None].expand([nodes.shape[0], K, nodes.shape[-1]])) - t0_nodes[:,
                                                                                                         None]
        elif mode == 'floyd':
            nn_weight, _, nn_idx = self.cal_nn_weight_floyd(x=p, t0=t0, cur_node=p0, K=K + 1, GraphK=4,
                                                            temperature=1e-1, cache_name='p2dR', XisNode=True)
            nn_weight, nn_idx = nn_weight[:, 1:], nn_idx[:, 1:]
            edges = torch.gather(p0[:, None].expand([nodes.shape[0], K, nodes.shape[-1]]), dim=0,
                                 index=nn_idx[..., None].expand([nodes.shape[0], K, nodes.shape[-1]])) - p0[:, None]
            t0_deform = None
        else:
            nn_dist, nn_idx, nn_nodes = pytorch3d.ops.knn_points(nodes[None], nodes[None], None, None, K=K + 1,
                                                                 return_nn=True)
            nn_dist, nn_idx, nn_nodes = nn_dist[0, :, 1:], nn_idx[0, :, 1:], nn_nodes[0, :, 1:]
            nn_weight = torch.softmax(nn_dist / nn_dist.mean(), dim=-1)
            if p0 is None:
                edges = nn_nodes - nodes[:, None]
            else:
                edges = torch.gather(p0[:, None].expand([nodes.shape[0], K, nodes.shape[-1]]), dim=0,
                                     index=nn_idx[..., None].expand([nodes.shape[0], K, nodes.shape[-1]])) - p0[:, None]
            t0_deform = None
        edges_t = torch.gather(p[:, None].expand([p.shape[0], K, p.shape[-1]]), dim=0,
                               index=nn_idx[..., None].expand([p.shape[0], K, p.shape[-1]])) - p[:, None]
        edges, edges_t = edges / (edges.norm(dim=-1, keepdim=True) + 1e-5), edges_t / (
                    edges_t.norm(dim=-1, keepdim=True) + 1e-5)
        W = torch.zeros([edges.shape[0], K, K], dtype=torch.float32, device=edges.device)
        W[:, range(K), range(K)] = nn_weight
        S = torch.einsum('nka,nkg,ngb->nab', edges, W, edges_t)
        U, _, V = svd(S)
        dR = torch.matmul(V, U.permute(0, 2, 1))
        if as_quat:
            dR = matrix_to_quaternion(dR)
        return dR, t0_deform

    def arap_loss(self, t=None, delta_t=0.05, t_samp_num=2):
        t = torch.rand([]).cuda() if t is None else t.squeeze() + delta_t * (torch.rand([]).cuda() - .5)
        t_samp = torch.rand(t_samp_num).cuda() * delta_t + t - .5 * delta_t
        t_samp = t_samp[None, :, None].expand(self.node_num, t_samp_num, 1)  # M, T, 1
        node_trans = self.node_deform(t=t_samp)['d_xyz']
        nodes_t = self.nodes[:, None, :3].detach() + node_trans  # M, T, 3
        hyper_nodes = nodes_t[:, 0]  # M, 3
        ii, jj, nn, weight = cal_connectivity_from_points(hyper_nodes, K=10)  # connectivity of control nodes
        error = cal_arap_error(nodes_t.permute(1, 0, 2), ii, jj, nn)
        return error

    def elastic_loss(self, t=None, delta_t=0.005, K=2, t_samp_num=8):
        # Calculate nodes translate
        t = torch.rand([]).cuda() if t is None else t.squeeze() + delta_t * (torch.rand([]).cuda() - .5)
        t_samp = torch.rand(t_samp_num).cuda() * delta_t + t - .5 * delta_t
        t_samp = t_samp[None, :, None].expand(self.node_num, t_samp_num, 1)
        node_trans = self.node_deform(t=t_samp)['d_xyz']
        nodes_t = self.nodes[:, None, :3].detach() + node_trans  # M, T, 3

        # Calculate weights of nodes NN
        nn_weight, _, nn_idx = self.cal_nn_weight(x=self.nodes[..., :3].detach(), feature=self.nodes[..., 3:], K=K + 1)
        nn_weight, nn_idx = nn_weight[:, 1:], nn_idx[:, 1:]  # M, K

        # Calculate edge deform loss
        edge_t = (nodes_t[nn_idx] - nodes_t[:, None]).norm(dim=-1)  # M, K, T
        edge_t_var = edge_t.var(dim=2)  # M, K
        edge_t_var = edge_t_var / (edge_t_var.detach() + 1e-5)
        arap_loss = (edge_t_var * nn_weight).sum(dim=1).mean()
        return arap_loss

    def acc_loss(self, t=None, delta_t=.005):
        # Calculate nodes translate
        t = torch.rand([]).cuda() if t is None else t.squeeze() + delta_t * (torch.rand([]).cuda() - .5)
        t = torch.stack([t - delta_t, t, t + delta_t])
        t = t[None, :, None].expand(self.node_num, 3, 1)
        node_trans = self.node_deform(t=t)['d_xyz']
        nodes_t = self.nodes[:, None, :3].detach() + node_trans  # M, 3, 3
        acc = (nodes_t[:, 0] + nodes_t[:, 2] - 2 * nodes_t[:, 1]).norm(dim=-1)  # M
        acc = acc / (acc.detach() + 1e-5)
        acc_loss = acc.mean()
        return acc_loss

    def geodesic_distance_floyd(self, cur_node, K=8):
        node_num = cur_node.shape[0]
        nn_dist, nn_idx, _ = pytorch3d.ops.knn_points(cur_node[None], cur_node[None], None, None, K=K + 1)
        nn_dist, nn_idx = nn_dist[0] ** .5, nn_idx[0]
        dist_mat = torch.inf * torch.ones([node_num, node_num], dtype=torch.float32, device=cur_node.device)
        dist_mat.scatter_(dim=1, index=nn_idx, src=nn_dist)
        dist_mat = torch.minimum(dist_mat, dist_mat.T)
        for i in range(nn_dist.shape[0]):
            dist_mat = torch.minimum((dist_mat[:, i, None] + dist_mat[None, i, :]), dist_mat)
        return dist_mat

    def _gather_weight_T(self, node_attr_T, nn_idx0_T, nn_idx1_T, nn_weight_T):
        """
        node_attr_T: (T,M,C) -> (T,N,C), KNN weighted.
        """
        picked = node_attr_T[nn_idx0_T, nn_idx1_T, :]  # (T,N,K,C)
        out = (picked * nn_weight_T).sum(dim=2)  # (T,N,C)
        return out

    def forward(self, x, t, feature, motion_mask, knn_feature=None, train_mask=None, iteration=0, is_training=True, node_trans_bias=None,
                node_scaling_bias=None, animation_d_values=None, **kwargs):
        assert t.dim() == 2
        N, K, M, T = x.shape[0], self.K, self.node_num, self.T

        # if t.dim() == 0:
        #     t = self.expand_time(t)
        x = x.detach()
        rot_bias = torch.tensor([1., 0, 0, 0]).float().to(x.device)
        rot_bias_T = rot_bias.view(1, 1, -1)
        motion_mask_T = motion_mask.unsqueeze(0).expand(T, -1, -1)

        # Calculate nn weights: [N, K]
        nn_weight, _, nn_idx = self.cal_nn_weight(
            x=x, feature=feature, gs_knn_feature=knn_feature,
            learning_metric=self.enable_learned_metric)
        nn_idx0_T = torch.arange(T, device=nn_idx.device).view(T, 1, 1).expand(T, N, K)
        nn_idx1_T = nn_idx.unsqueeze(0).expand(T, -1, -1)
        nn_weight_T = nn_weight.unsqueeze(0).expand(T, -1, -1).unsqueeze(-1)

        node_attrs = self.node_deform(t=t, mask=train_mask, **kwargs)

        # Animation
        if animation_d_values is not None:
            for key in animation_d_values:
                node_attrs[key] = animation_d_values[key]

        node_trans_T, node_rot_T, node_scale_T = node_attrs['d_xyz'], node_attrs['d_rotation'], node_attrs['d_scaling']

        # Obtain translation
        if self.local_frame:
            local_rot_T = node_attrs['local_rotation'] + rot_bias_T
            local_rot_matrix_T = quaternion_to_matrix(local_rot_T)
            nn_nodes = self.nodes[nn_idx, ..., :3].detach()
            nn_nodes_T = nn_nodes.unsqueeze(0).expand(T, -1, -1, -1)

            x_vec_T = (x[:, None, :] - nn_nodes).unsqueeze(0).expand(T, -1, -1, -1)  # (T,N,K,3)
            R_T = local_rot_matrix_T[nn_idx0_T, nn_idx1_T, ...]  # (T,N,K,3,3)
            trans_T = node_trans_T[nn_idx0_T, nn_idx1_T, :]

            Ax_T = torch.einsum('tnkab,tnkb->tnka', R_T, x_vec_T) + nn_nodes_T + trans_T  # (T,N,K,3)
            Ax_avg_T = (Ax_T * nn_weight_T).sum(dim=2)  # (T,N,3)
            translate_T = Ax_avg_T - x.unsqueeze(0)

        else:
            if self.use_dq:
                q_nodes_T = node_rot_T + rot_bias_T if not self.d_rot_as_res else node_rot_T  # (T,M,4)
                dq_nodes_T = self._rt_to_dq_T(q_nodes_T, node_trans_T)

                q_TN, t_TN = self._dq_knn_blend_T(
                    dq_nodes_T=dq_nodes_T,
                    nn_idx0_T=nn_idx0_T, nn_idx1_T=nn_idx1_T,
                    nn_weight_T=nn_weight_T
                )  # q_TN:(T,N,4), t_TN:(T,N,3)

                # x_warp_T = self._apply_se3_T(x, q_TN, t_TN)  # (T,N,3)
                # translate_T = (x_warp_T - x.unsqueeze(0))  # (T,N,3)
                picked_trans_T = node_trans_T[nn_idx0_T, nn_idx1_T, :]  # (T,N,K,3)
                translate_T = (picked_trans_T * nn_weight_T).sum(dim=2)  # (T,N,3)
            else:
                picked_trans_T = node_trans_T[nn_idx0_T, nn_idx1_T, :]  # (T,N,K,3)
                translate_T = (picked_trans_T * nn_weight_T).sum(dim=2)  # (T,N,3)

        translate_T = translate_T * motion_mask_T

        if not self.d_rot_as_res:
            raise ValueError("The public release requires d_rot_as_res=True.")
        else:
            if self.use_dq:
                rotation_T = q_TN * motion_mask_T  # (T,N,4)
            else:
                picked_rot_T = node_rot_T[nn_idx0_T, nn_idx1_T, :]  # (T,N,K,4)
                rotation_T = (picked_rot_T * nn_weight_T).sum(dim=2) * motion_mask_T

            scale_T = self._gather_weight_T(node_scale_T, nn_idx0_T, nn_idx1_T, nn_weight_T) * motion_mask_T  # (T,N,3)

            return_dict = {'d_xyz': translate_T, 'd_rotation': rotation_T, 'd_scaling': scale_T}

            if node_trans_bias is not None:
                with torch.no_grad():
                    cur_node_T = self.nodes[..., :3][None, :, :] + node_trans_T  # (T,M,3)
                    x_T = x[None, :, :].expand(T, -1, -1)
                    cur_gs_T = x_T + translate_T

                    cur_nn_weight_list, cur_nn_idx_list = [], []
                    for ti in range(T):
                        w_t, _, idx_t = self.cal_nn_weight(x=cur_gs_T[ti],
                                                           feature=None,
                                                           nodes=cur_node_T[ti],
                                                           K=8)
                        cur_nn_weight_list.append(w_t)
                        cur_nn_idx_list.append(idx_t)
                    cur_nn_weight_T = torch.stack(cur_nn_weight_list, dim=0)  # (T,N,K)
                    cur_nn_idx_T = torch.stack(cur_nn_idx_list, dim=0)

                    nodes_t = cur_node_T + node_trans_bias  # (T,M,3)
                    node_rot_bias_T, _ = self.p2dR(p=nodes_t, p0=cur_node_T, K=8, as_quat=True, mode='trajectory', t0=t)
                    d_rot_bias_T = (node_rot_bias_T[cur_nn_idx_T] * cur_nn_weight_T.unsqueeze(-1)).sum(dim=2)  # (T,N,4)
                    d_nn_node_rot_R_T = quaternion_to_matrix(node_rot_bias_T)[cur_nn_idx_T]  # (T,N,K,3,3)

                    gs_init_T = x_T + translate_T
                    rel_T = gs_init_T[:, :, None, :] - cur_node_T[cur_nn_idx_T]
                    gs_t_T = nodes_t[cur_nn_idx_T] + torch.einsum('tnkab,tnkb->tnka', d_nn_node_rot_R_T, rel_T)
                    gs_t_avg_T = (gs_t_T * cur_nn_weight_T.unsqueeze(-1)).sum(dim=2)
                    translate_T = gs_t_avg_T - x_T
                    return_dict['d_xyz'] = translate_T * motion_mask_T
                    return_dict['d_rotation_bias'] = ((node_rot_bias_T[cur_nn_idx_T] * cur_nn_weight_T.unsqueeze(
                        -1)).sum(dim=2) - rot_bias) * motion_mask_T + rot_bias

        if self.pred_opacity:
            node_opacity_T = node_attrs['d_opacity']
            d_opacity_T = (node_opacity_T[nn_idx0_T, nn_idx1_T, :] * nn_weight_T).sum(dim=2) * motion_mask_T  # (T,N,1)
            return_dict['d_opacity'] = d_opacity_T
        else:
            return_dict['d_opacity'] = None

        if self.pred_color:
            node_color_T = node_attrs['d_color']
            d_color_T = (node_color_T[nn_idx0_T, nn_idx1_T, :] * nn_weight_T).sum(dim=2) * motion_mask_T  # (T,N,3)
            return_dict['d_color'] = d_color_T
        else:
            return_dict['d_color'] = None

        self.reg_loss = 0.
        lambda_arap = landmark_interpolate(landmarks=self.lambda_arap_landmarks, steps=self.lambda_arap_steps,
                                           step=iteration)
        if self.training and lambda_arap > 0 and is_training:
            arap_loss = self.arap_loss()
            self.reg_loss = self.reg_loss + arap_loss * lambda_arap
        return return_dict

    @property
    def as_gaussians(self):
        if not hasattr(self, 'gs') or self.gs is None:
            print('Building Learnable Gaussians for Nodes!')
            from scene.gaussian_model import GaussianModel, BasicPointCloud, StandardGaussianModel
            pcd = BasicPointCloud(points=self.nodes[..., :3].detach(), colors=torch.zeros_like(self.nodes[..., :3]),
                                  normals=self.nodes[..., :3].detach())
            self.gs = StandardGaussianModel(sh_degree=0, all_the_same=True,
                                            with_motion_mask=False)  # blender datas are all dynamic
            self.gs.create_from_pcd(pcd=pcd, spatial_lr_scale=0., print_info=False)
            self.gs._scaling.data = torch.log(1e-2 * torch.ones_like(self.gs._scaling))
            self.gs._xyz.data = self.nodes[..., :3]
        return self.gs

    def init_gaussians(self, init_pcl, with_motion_mask):
        if not hasattr(self, 'gs') or self.gs is None:
            print('Initialize Learnable Gaussians for Nodes with Point Clouds!')
            from scene.gaussian_model import GaussianModel, BasicPointCloud, StandardGaussianModel
            pcd = BasicPointCloud(points=init_pcl.detach(), colors=torch.zeros_like(init_pcl),
                                  normals=torch.zeros_like(init_pcl))
            self.gs = StandardGaussianModel(sh_degree=0, all_the_same=True,
                                            with_motion_mask=with_motion_mask)  # blender datas are all dynamic
            self.gs.create_from_pcd(pcd=pcd, spatial_lr_scale=0., print_info=False)
        return self.gs

    def cal_node_importance(self, x: torch.Tensor, gs_knn_feature: torch.Tensor, K=None, weights=None, feature=None):
        device = self.nodes.device
        K = self.K if K is None else K

        nn_weight, _, nn_idxs = self.cal_nn_weight(
            x=x,  # kept for compatibility; no longer used directly for KNN
            K=K,
            feature=feature,
            gs_knn_feature=gs_knn_feature,  # KNN query source
            gs_kernel=True,
            learning_metric=self.enable_learned_metric
        )  # shapes: (N, K), (N, K)

        N = feature.shape[0]
        M = self.nodes.shape[0]
        node_code_dim = self.nodes.shape[1] - 3

        node_importance = torch.zeros(M, device=device)
        node_edge_count = torch.zeros(M, device=device)
        avg_affected_x = torch.zeros_like(self.nodes)  # (M, 3 + node_code_dim)

        if weights is None:
            weights = torch.ones(N, device=device)
        else:
            weights = weights.to(device)

        node_importance.index_add_(dim=0,
                                   index=nn_idxs.reshape(-1),
                                   source=(nn_weight * weights[:, None]).reshape(-1))
        node_edge_count.index_add_(dim=0,
                                   index=nn_idxs.reshape(-1),
                                   source=nn_weight.reshape(-1))

        q_mapped = torch.cat([x, feature], dim=-1)  # (N, node_code_dim)

        weighted = (nn_weight * weights[:, None]).reshape(-1, 1) * \
                   q_mapped[:, None, :].expand(N, K, node_code_dim).reshape(-1, node_code_dim)

        avg_affected_x[:, 3:].index_add_(dim=0,
                                         index=nn_idxs.reshape(-1),
                                         source=weighted)

        eps = 1e-8
        avg_affected_x[:, 3:] = avg_affected_x[:, 3:] / (node_importance[:, None] + eps)
        node_importance = node_importance / (node_edge_count + 1e-7)

        return node_importance, avg_affected_x, node_edge_count

    @torch.no_grad()
    def densify(self, max_grad, optimizer, x: torch.Tensor, knn_feature: torch.Tensor, x_grad: torch.Tensor, feature=None, K=None,
                use_gaussians_grad=False, force_dp=False):
        if not self.enable_dp and not force_dp:
            return
        if not self.inited:
            print('No need to densify nodes before initialization.')
            return
        if self.skinning:
            print('No need to densify for skinning type')
            return

        x_grad[x_grad.isnan()] = 0.
        K = self.K if K is None else K
        weights = x_grad.norm(dim=-1)

        # Calculate the avg importance and coor
        node_avg_xgradnorm, node_avg_x, node_edge_count = self.cal_node_importance(x=x, K=K, weights=weights,
                                                                                   feature=feature,
                                                                                   gs_knn_feature=knn_feature)

        # Picking pts to densify
        if use_gaussians_grad or not hasattr(self, 'nodes_accumulated_grad'):
            selected_pts_mask = torch.logical_and(node_avg_xgradnorm > max_grad,
                                                  node_avg_x.isnan().logical_not().all(dim=-1))
        else:
            avg_nodes_norm = self.nodes_accumulated_grad / self.denom
            selected_pts_mask = avg_nodes_norm > max_grad
            self.nodes_accumulated_grad.data = 0
            self.denom = 0

        # For visualization
        self.nodes_color_visualization = torch.ones_like(self.nodes[..., :3])

        min_keep = max(384, int(0.5 * self.node_num))
        pruned_pts_mask = node_edge_count == 0
        if pruned_pts_mask.sum() > (self.node_num - min_keep):
            bad_idx = torch.nonzero(pruned_pts_mask, as_tuple=False).view(-1)
            keep_n = self.node_num - min_keep
            pruned_pts_mask[:] = False
            pruned_pts_mask[bad_idx[:max(0, bad_idx.numel() - keep_n)]] = True


        if selected_pts_mask.sum() > 0 or pruned_pts_mask.sum() > 0:
            print(f'Add {selected_pts_mask.sum()} nodes and prune {pruned_pts_mask.sum()} nodes. ', end='')
        else:
            return

        # Densify
        if selected_pts_mask.sum() > 0 and self.node_num < 5000:
            new_nodes = node_avg_x[selected_pts_mask]
            new_node_radius = self._node_radius[selected_pts_mask]
            new_param_list = [new_nodes, new_node_radius]
            if self.with_node_weight:
                new_node_weight = self._node_weight[selected_pts_mask]
                new_param_list.append(new_node_weight)
            param_list = self.param_names
            param_idx = np.arange(len(param_list))
            for group in optimizer.param_groups:
                if group["name"] != 'nodes':
                    continue
                for i in param_idx:
                    stored_state = optimizer.state.get(group['params'][i], None)
                    extension_tensor = new_param_list[i]
                    if stored_state is not None:
                        stored_state["exp_avg"] = torch.cat(
                            (stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
                        stored_state["exp_avg_sq"] = torch.cat(
                            (stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)
                        del optimizer.state[group['params'][i]]
                        group["params"][i] = nn.Parameter(
                            torch.cat((group["params"][i], extension_tensor), dim=0).requires_grad_(True))
                        optimizer.state[group['params'][i]] = stored_state
                        setattr(self, param_list[i], group["params"][i])
                    else:
                        group["params"][i] = nn.Parameter(
                            torch.cat((group["params"][i], extension_tensor), dim=0).requires_grad_(True))
                        setattr(self, param_list[i], group["params"][i])
            self.nodes_color_visualization = torch.cat(
                [self.nodes_color_visualization, torch.ones_like(new_nodes[..., :3])], dim=0)
            self.nodes_color_visualization[-new_nodes.shape[0]:, 1:] = 0  # Set as red

        # Prune
        if pruned_pts_mask.shape[0] < self.nodes.shape[0]:
            pruned_pts_mask = torch.cat([pruned_pts_mask,
                                         torch.zeros([self.nodes.shape[0] - pruned_pts_mask.shape[0]]).to(
                                             pruned_pts_mask.device).to(pruned_pts_mask.dtype)])
        if pruned_pts_mask.sum() > 0:
            pruned_pts_mask = ~pruned_pts_mask
            if self.nodes_color_visualization.shape[0] != pruned_pts_mask.shape[0]:
                self.nodes_color_visualization = torch.ones(
                    (pruned_pts_mask.shape[0], 3),
                    dtype=self.nodes.dtype,
                    device=self.nodes.device,
                )
            self.nodes_color_visualization = self.nodes_color_visualization[pruned_pts_mask]
            optimizable_tensors = {}
            param_list = self.param_names
            param_idx = np.arange(len(param_list))
            for group in optimizer.param_groups:
                if group["name"] != 'nodes':
                    continue
                for i in param_idx:
                    stored_state = optimizer.state.get(group['params'][i], None)
                    if stored_state is not None:
                        stored_state["exp_avg"] = stored_state["exp_avg"][pruned_pts_mask]
                        stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][pruned_pts_mask]
                        del optimizer.state[group['params'][i]]
                        group["params"][i] = nn.Parameter((group["params"][i][pruned_pts_mask].requires_grad_(True)))
                        optimizer.state[group['params'][i]] = stored_state
                        optimizable_tensors[param_list[i]] = group["params"][i]
                    else:
                        group["params"][i] = nn.Parameter(group["params"][i][pruned_pts_mask].requires_grad_(True))
                        optimizable_tensors[param_list[i]] = group["params"][i]
            for key in optimizable_tensors:
                setattr(self, key, optimizable_tensors[key])
        else:
            pruned_pts_mask = ~pruned_pts_mask

        if not self.with_node_weight:
            self._node_weight = torch.zeros_like(self.nodes[..., :1])

        self.gs.densify_and_split(selected_pts_mask=selected_pts_mask, N=1, without_prune=True)
        self.gs.prune_points(~pruned_pts_mask)
        self.gs._xyz.data = self.nodes[..., :3]
        print(f'With {self.nodes.shape[0]} nodes left.')
