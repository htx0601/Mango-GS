#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import torch
import random
import shutil
from utils.loss_utils import l1_loss, ssim
from gaussian_renderer import render, render_batch
import sys
from scene import Scene, GaussianModel, DeformModel
from utils.general_utils import safe_state, get_linear_noise_func
import uuid
import tqdm
from argparse import ArgumentParser, Namespace
from arguments.__init__video import ModelParams, PipelineParams, OptimizationParams, apply_dataset_preset
from training_report import training_report
import math
import numpy as np
from collections import defaultdict
import torch.nn.functional as F

try:
    from torch.utils.tensorboard import SummaryWriter

    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False


def _checkpoint_iteration_dirs(root):
    if not os.path.isdir(root):
        return []
    out = []
    for name in os.listdir(root):
        if not name.startswith("iteration_"):
            continue
        try:
            iteration = int(name.split("_", 1)[1])
        except ValueError:
            continue
        path = os.path.join(root, name)
        if os.path.isdir(path):
            out.append((iteration, path))
    return out


def _depthwise_filter2d(images, kernel):
    channels = images.shape[1]
    weight = kernel.to(device=images.device, dtype=images.dtype).view(1, 1, *kernel.shape)
    weight = weight.repeat(channels, 1, 1, 1)
    return F.conv2d(images, weight, padding=kernel.shape[-1] // 2, groups=channels)


def _sobel_magnitude(images):
    kx = torch.tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]], device=images.device, dtype=images.dtype)
    ky = kx.t()
    gx = _depthwise_filter2d(images, kx)
    gy = _depthwise_filter2d(images, ky)
    return torch.sqrt(gx * gx + gy * gy + 1e-8)


def _laplacian(images):
    kernel = torch.tensor([[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]], device=images.device, dtype=images.dtype)
    return _depthwise_filter2d(images, kernel)


def _dynamic_mask_from_gt(gt_images, tau_quantile=0.75, blur_ks=3):
    if gt_images.shape[0] < 2:
        return None
    motion = torch.zeros((gt_images.shape[0], 1, gt_images.shape[2], gt_images.shape[3]), device=gt_images.device, dtype=gt_images.dtype)
    diffs = (gt_images[1:] - gt_images[:-1]).abs().mean(dim=1, keepdim=True)
    motion[1:] = torch.maximum(motion[1:], diffs)
    motion[:-1] = torch.maximum(motion[:-1], diffs)
    if blur_ks > 1:
        if blur_ks % 2 == 0:
            blur_ks += 1
        motion = F.avg_pool2d(motion, kernel_size=blur_ks, stride=1, padding=blur_ks // 2)
    flat = motion.flatten(1)
    if flat.shape[1] == 0:
        return None
    tau = torch.quantile(flat, tau_quantile, dim=1, keepdim=True).view(-1, 1, 1, 1)
    mask = (motion >= tau).detach()
    if not mask.any():
        return None
    return mask


def _masked_mean(value, mask):
    mask = mask.to(dtype=value.dtype)
    while mask.dim() < value.dim():
        mask = mask.expand(-1, value.shape[1], -1, -1)
    return (value * mask).sum() / (mask.sum() + 1e-8)


def dynamic_edge_loss(preds, gt_images, tau_quantile=0.75, blur_ks=3):
    mask = _dynamic_mask_from_gt(gt_images, tau_quantile=tau_quantile, blur_ks=blur_ks)
    if mask is None:
        return None, None
    pred_edges = _sobel_magnitude(preds)
    gt_edges = _sobel_magnitude(gt_images)
    edge_loss = _masked_mean((pred_edges - gt_edges).abs(), mask)
    pred_lap = _laplacian(preds)
    gt_lap = _laplacian(gt_images)
    lap_loss = _masked_mean((pred_lap - gt_lap).abs(), mask)
    return edge_loss, lap_loss


class MangoTrainer:
    def __init__(self, args, dataset, opt, pipe, testing_iterations, saving_iterations) -> None:
        self.dataset = dataset
        self.args = args
        self.opt = opt
        self.pipe = pipe
        self.testing_iterations = testing_iterations
        self.saving_iterations = saving_iterations

        if self.opt.progressive_train:
            self.opt.iterations_node_sampling = max(self.opt.iterations_node_sampling,
                                                    int(self.opt.progressive_stage_steps / self.opt.progressive_stage_ratio))
            self.opt.iterations_node_rendering = max(self.opt.iterations_node_rendering,
                                                     self.opt.iterations_node_sampling + 2000)
            print(
                f'Progressive train is on. Adjusting the iterations node sampling to {self.opt.iterations_node_sampling} and iterations node rendering {self.opt.iterations_node_rendering}')

        self.tb_writer = prepare_output_and_logger(dataset)
        self.deform = DeformModel(K=self.dataset.K, deform_type=self.dataset.deform_type, T=self.dataset.T,
                                  is_blender=self.dataset.is_blender, skinning=self.args.skinning,
                                  hyper_dim=self.dataset.hyper_dim, node_num=self.dataset.node_num,
                                  pred_opacity=self.dataset.pred_opacity, pred_color=self.dataset.pred_color,
                                  use_hash=self.dataset.use_hash, hash_time=self.dataset.hash_time,
                                  d_rot_as_res=self.dataset.d_rot_as_res and not self.dataset.d_rot_as_rotmat,
                                  local_frame=self.dataset.local_frame,
                                  progressive_brand_time=self.dataset.progressive_brand_time,
                                  with_arap_loss=not self.opt.no_arap_loss, max_d_scale=self.dataset.max_d_scale,
                                  enable_densify_prune=self.opt.node_enable_densify_prune,
                                  is_scene_static=dataset.is_scene_static,
                                  enable_learned_metric=self.dataset.enable_learned_metric,
                                  use_tcnn = self.opt.use_tcnn,)
        deform_loaded = self.deform.load_weights(dataset.model_path, iteration=-1)
        self.deform.train_setting(opt)

        gs_fea_dim = self.deform.deform.node_num if args.skinning and 'node' in self.deform.name else self.dataset.hyper_dim
        self.gaussians = GaussianModel(dataset.sh_degree, fea_dim=gs_fea_dim,
                                       with_motion_mask=self.dataset.gs_with_motion_mask)

        self.scene = Scene(dataset, self.gaussians, load_iteration=-1)
        self.gaussians.training_setup(opt)
        if 'node' in self.deform.name and not deform_loaded:
            if not self.dataset.is_blender:
                if self.opt.random_init_deform_gs:
                    num_pts = 100_000
                    print(f"Generating random point cloud ({num_pts})...")
                    xyz = torch.rand((num_pts, 3)).float().cuda() * 2 - 1
                    mean, scale = self.gaussians.get_xyz.mean(dim=0), self.gaussians.get_xyz.std(dim=0).mean() * 3
                    xyz = xyz * scale + mean
                    self.deform.deform.init(init_pcl=xyz, force_init=True, opt=self.opt,
                                            as_gs_force_with_motion_mask=self.dataset.as_gs_force_with_motion_mask,
                                            force_gs_keep_all=True)
                else:
                    print('Initialize nodes with COLMAP point cloud.')
                    self.deform.deform.init(init_pcl=self.gaussians.get_xyz, force_init=True, opt=self.opt,
                                            as_gs_force_with_motion_mask=self.dataset.as_gs_force_with_motion_mask,
                                            force_gs_keep_all=self.dataset.init_isotropic_gs_with_all_colmap_pcl)
            else:
                print('Initialize nodes with Random point cloud.')
                self.deform.deform.init(init_pcl=self.gaussians.get_xyz, force_init=True, opt=self.opt,
                                        as_gs_force_with_motion_mask=False, force_gs_keep_all=args.skinning)

        bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
        self.background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        self.iter_start = torch.cuda.Event(enable_timing=True)
        self.iter_end = torch.cuda.Event(enable_timing=True)
        self.iteration = 1 if self.scene.loaded_iter is None else self.scene.loaded_iter
        self.iteration_node_rendering = 1 if self.scene.loaded_iter is None else self.opt.iterations_node_rendering

        self.viewpoint_stack = None
        self.ema_loss_for_log = 0.0
        self.best_psnr = 0.0
        self.best_ssim = 0.0
        self.best_ms_ssim = 0.0
        self.best_lpips = np.inf
        self.best_alex_lpips = np.inf
        self.best_iteration = 0
        self.saved_checkpoint_iterations = set()
        self.progress_bar = tqdm.tqdm(range(opt.iterations), desc="Training progress")
        self.smooth_term = get_linear_noise_func(lr_init=0.1, lr_final=1e-15, lr_delay_mult=0.01, max_steps=20000)

    def _save_checkpoint(self, reason):
        print("\n[ITER {}] Saving {}".format(self.iteration, reason))
        self.scene.save(self.iteration)
        self.deform.save_weights(self.args.model_path, self.iteration)
        self.saved_checkpoint_iterations.add(self.iteration)
        self._prune_checkpoints()

    def _prune_checkpoints(self):
        keep_limit = getattr(self.args, "keep_recent_checkpoints", 0)
        if keep_limit <= 0:
            return

        keep = set(getattr(self.args, "keep_checkpoint_iterations", []) or [])
        if self.best_iteration > 0:
            keep.add(self.best_iteration)
        keep.add(self.iteration)
        keep.add(self.opt.iterations)

        recent = sorted(self.saved_checkpoint_iterations, reverse=True)[:keep_limit]
        keep.update(recent)

        roots = [
            os.path.join(self.args.model_path, "point_cloud"),
            os.path.join(self.args.model_path, "deform"),
        ]
        for root in roots:
            for iteration, path in _checkpoint_iteration_dirs(root):
                if iteration in keep:
                    continue
                try:
                    shutil.rmtree(path)
                    print("[checkpoint prune] Removed {}".format(path))
                except FileNotFoundError:
                    pass

    def get_placeholder_frame(self, fids):
        PlaceholderFrame = type("PlaceholderFrame", (), {})
        ph = PlaceholderFrame()
        ph.original_image = None
        ph.gt_alpha_mask = None
        ph.camera_center = None
        ph.fid = fids
        return ph

    def densify_and_prune(self, viewspace_point_tensor, visibility_filter, radii, T):
        vis_count = torch.stack(visibility_filter, dim=0).float().sum(dim=0)  # (N,)
        min_vis = max(2, int(math.ceil(0.3 * T)))
        vis_mask = (vis_count >= min_vis)

        per_frame_grads = []
        for vs in viewspace_point_tensor:
            if vs.grad is not None:
                g = torch.norm(vs.grad[vis_mask, :2], dim=-1, keepdim=True)  # (M,1)
            else:
                g = torch.zeros((vis_mask.sum(), 1), device=vs.device, dtype=vs.dtype)
            per_frame_grads.append(g)
        if vis_mask.sum().item() == 0:
            return
        grad_mags = torch.stack(per_frame_grads, dim=0)  # (T, M, 1)

        q = getattr(self.opt, "densify_grad_reduce_quantile", None)
        if q is not None and q < 0:
            q = None
        if q is None:
            grad_stats = grad_mags.max(dim=0).values  # (M,1)
        else:
            grad_stats = torch.quantile(grad_mags.squeeze(-1), q, dim=0, keepdim=True).transpose(0, 1)  # (M,1)

        radii_max = radii.max(dim=0).values
        if self.gaussians.max_radii2D.shape[0] == 0:
            self.gaussians.max_radii2D = torch.zeros_like(radii_max)
        self.gaussians.max_radii2D[vis_mask] = torch.max(
            self.gaussians.max_radii2D[vis_mask],
            radii_max[vis_mask]
        )

        if self.iteration < self.opt.densify_until_iter:
            self.gaussians.add_densification_stats(grad_stats, vis_mask)

            if self.iteration > self.opt.node_densify_from_iter and self.iteration % self.opt.node_densification_interval == 0 and self.iteration < self.opt.node_densify_until_iter and self.iteration > self.opt.warm_up or self.iteration == self.opt.node_force_densify_prune_step:
                # Nodes densify
                self.deform.densify(max_grad=self.opt.densify_node_grad_threshold, x=self.gaussians.get_xyz,
                                    knn_feature=self.gaussians.get_knn_feature,
                                    x_grad=self.gaussians.xyz_gradient_accum / self.gaussians.denom,
                                    feature=self.gaussians.feature,
                                    force_dp=(self.iteration == self.opt.node_force_densify_prune_step))

            if self.iteration > self.opt.densify_from_iter and self.iteration % self.opt.densification_interval == 0:
                size_threshold = 20 if self.iteration > self.opt.opacity_reset_interval else None
                self.gaussians.densify_and_prune(self.opt.densify_grad_threshold, 0.004, self.scene.cameras_extent,
                                                 size_threshold)

            if self.iteration % self.opt.opacity_reset_interval == 0 or (
                    self.dataset.white_background and self.iteration == self.opt.densify_from_iter):
                self.gaussians.reset_opacity()


    def train(self, iters=5000):
        if self.iteration_node_rendering < self.opt.iterations_node_rendering:
            self.iteration_node_rendering = 1
            for _ in tqdm.trange(self.opt.iterations_node_rendering, desc="Node Training"):
                self.train_node_init_step()

        self.viewpoint_stack = None
        self.ema_loss_for_log = 0.0

        if iters > 0:
            for _ in tqdm.trange(iters, desc="Gaussian Training"):
                if getattr(self, "stop_training", False):
                    break
                self.train_node_video_step()


    def train_node_init_step(self):
        # Pick a random Camera
        if not self.viewpoint_stack:
            viewpoint_stack = self.scene.getTrainCameras().copy()
            self.viewpoint_stack = viewpoint_stack

        viewpoint_cam = self.viewpoint_stack.pop(random.randint(0, len(self.viewpoint_stack) - 1))
        if self.dataset.load2gpu_on_the_fly:
            viewpoint_cam.load2device('cuda')
        time_input = viewpoint_cam.fid
        N = self.deform.deform.as_gaussians.get_xyz.shape[0]

        total_frame = len(self.scene.getTrainCameras())
        time_interval = 1 / total_frame

        if self.dataset.is_blender:
            noise = torch.zeros(1, device='cuda')
        else:
            noise = (torch.randn(1, 1, device='cuda')
                     * time_interval
                     * self.smooth_term(self.iteration_node_rendering))  # (1, 1)

        d_values = self.deform.deform.query_network(x=self.deform.deform.as_gaussians.get_xyz.detach(),
                                                    t=time_input + noise)
        d_xyz, d_opacity, d_color = d_values['d_xyz'] * self.deform.deform.as_gaussians.motion_mask, d_values[
                                                                                                         'd_opacity'] * self.deform.deform.as_gaussians.motion_mask if \
            d_values['d_opacity'] is not None else None, d_values[
                                                             'd_color'] * self.deform.deform.as_gaussians.motion_mask if \
            d_values['d_color'] is not None else None
        d_rot, d_scale = 0., 0.
        if self.iteration_node_rendering < self.opt.node_warm_up:
            d_xyz = d_xyz.detach()
        d_color = d_color.detach() if d_color is not None else None
        d_opacity = d_opacity.detach() if d_opacity is not None else None

        # Render
        random_bg_color = (self.opt.gt_alpha_mask_as_scene_mask or (
                self.opt.gt_alpha_mask_as_dynamic_mask and not self.deform.deform.as_gaussians.with_motion_mask)) and viewpoint_cam.gt_alpha_mask is not None
        render_pkg_re = render(viewpoint_cam, self.deform.deform.as_gaussians, self.pipe, self.background, d_xyz, d_rot,
                               d_scale, random_bg_color=random_bg_color, d_opacity=d_opacity, d_color=d_color,
                               d_rot_as_res=self.deform.d_rot_as_res)
        image, pred_depth, viewspace_point_tensor, visibility_filter, radii = (
            render_pkg_re["render"],
            render_pkg_re["depth"],
            render_pkg_re["viewspace_points"],
            render_pkg_re["visibility_filter"],
            render_pkg_re["radii"]
        )

        # Loss
        gt_image = viewpoint_cam.original_image.cuda()
        if random_bg_color:
            gt_alpha_mask = viewpoint_cam.gt_alpha_mask.cuda()
            gt_image = gt_image * gt_alpha_mask + render_pkg_re['bg_color'][:, None, None] * (1 - gt_alpha_mask)
        Ll1 = l1_loss(image, gt_image)
        loss = (1.0 - self.opt.lambda_dssim) * Ll1 + self.opt.lambda_dssim * (1.0 - ssim(image, gt_image))

        lambda_depth = float(getattr(self.opt, "lambda_depth", 0.0))
        if viewpoint_cam.depth is not None and lambda_depth > 0:
            gt_depth = viewpoint_cam.depth.cuda()
            valid_mask = (gt_depth > 1e-5).detach()
            loss_depth = ((pred_depth - gt_depth).abs() * valid_mask).sum() / (valid_mask.sum() + 1e-8)
            loss = (1 - lambda_depth) * loss + lambda_depth * loss_depth

        loss.backward()
        with torch.no_grad():
            # Progress bar
            self.ema_loss_for_log = 0.4 * loss.item() + 0.6 * self.ema_loss_for_log
            if self.iteration_node_rendering % 10 == 0:
                self.progress_bar.set_postfix({"Loss": f"{self.ema_loss_for_log:.{7}f}"})
                self.progress_bar.update(10)

            if self.iteration_node_rendering < self.opt.iterations_node_sampling:
                # Densification
                self.deform.deform.as_gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)
                if self.iteration_node_rendering % self.opt.densification_interval == 0 or self.iteration_node_rendering == self.opt.node_warm_up - 1:
                    size_threshold = 20 if self.iteration_node_rendering > self.opt.opacity_reset_interval else None
                    if self.dataset.is_blender:
                        grad_max = self.opt.densify_grad_threshold
                    else:
                        if self.deform.deform.as_gaussians.get_xyz.shape[
                            0] > self.deform.deform.node_num * self.opt.node_max_num_ratio_during_init:
                            grad_max = torch.inf
                        else:
                            grad_max = self.opt.densify_grad_threshold
                    self.deform.deform.as_gaussians.densify_and_prune(grad_max, 0.005, self.scene.cameras_extent,
                                                                      size_threshold)
                if self.iteration_node_rendering % self.opt.opacity_reset_interval == 0 or (
                        self.dataset.white_background and self.iteration_node_rendering == self.opt.densify_from_iter):
                    self.deform.deform.as_gaussians.reset_opacity()
            elif self.iteration_node_rendering == self.opt.iterations_node_sampling:

                src = self.deform.deform.as_gaussians  # Gaussians trained during the node-rendering phase.
                dst = self.gaussians  # Main rendering Gaussians.

                # Rebuild dst from src points when the point count no longer matches.
                need_recreate = (dst.get_xyz.shape[0] != src.get_xyz.shape[0])
                if need_recreate:
                    from scene.gaussian_model import BasicPointCloud
                    with torch.no_grad():
                        pcd = BasicPointCloud(
                            points=src.get_xyz.detach(),
                            colors=torch.zeros_like(src.get_xyz),
                            normals=src.get_xyz.detach()
                        )
                        # Keep dst compatible with the StandardGaussianModel create_from_pcd interface.
                        dst.create_from_pcd(pcd=pcd, spatial_lr_scale=0., print_info=False)

                with torch.no_grad():
                    xyz = src.get_xyz.detach().clone()
                    opacity = src._opacity.detach().clone()
                    scales = src._scaling.detach().clone()
                    rots = src._rotation.detach().clone()

                    f_dc = src._features_dc.detach().clone()  # [N, 1, 3]
                    # Copy SH rest features up to the destination feature dimension.
                    src_rest = src._features_rest.detach()  # [N, C_src, 3]
                    N, C_dst, _ = dst._features_rest.shape  # Use dst as the shape template.
                    _, C_src, _ = src_rest.shape
                    C_keep = min(C_src, C_dst)
                    f_rest = torch.zeros((N, C_dst, 3), device=src_rest.device, dtype=src_rest.dtype)
                    f_rest[:, :C_keep, :] = src_rest[:, :C_keep, :]

                    fea_dim_dst = getattr(dst, "fea_dim", 0)
                    fea_dim_src = getattr(src, "fea_dim", 0)
                    if fea_dim_dst > 0:
                        feat = torch.zeros((xyz.shape[0], fea_dim_dst), device=xyz.device, dtype=torch.float32)
                        if fea_dim_src > 0:
                            d = min(fea_dim_src, fea_dim_dst)
                            feat[:, :d] = src.feature.detach()[:, :d]
                    else:
                        feat = None

                dst._xyz = torch.nn.Parameter(xyz, requires_grad=True)
                dst._features_dc = torch.nn.Parameter(f_dc, requires_grad=True)
                dst._features_rest = torch.nn.Parameter(f_rest, requires_grad=True)
                dst._opacity = torch.nn.Parameter(opacity, requires_grad=True)
                dst._scaling = torch.nn.Parameter(scales, requires_grad=True)
                dst._rotation = torch.nn.Parameter(rots, requires_grad=True)
                if feat is not None:
                    dst.feature = torch.nn.Parameter(feat, requires_grad=True)

                dst.active_sh_degree = dst.max_sh_degree
                device = dst.get_xyz.device
                with torch.no_grad():
                    N = dst.get_xyz.shape[0]
                    dst.xyz_gradient_accum = torch.zeros((N, 1), device=device)
                    dst.denom = torch.zeros((N, 1), device=device)
                    dst.max_radii2D = torch.zeros((N,), device=device)

                dst.training_setup(self.opt)

                print(
                    f"[Sync Gaussians] Copied {src.get_xyz.shape[0]} points from deform.as_gaussians -> self.gaussians "
                    f"(SH cropped from src to match dst max_sh_degree={dst.max_sh_degree}, fea_dim={dst.fea_dim}). Optimizer reset."
                )

                strategy = self.opt.deform_downsamp_strategy
                if strategy == 'direct':
                    original_gaussians: GaussianModel = self.deform.deform.as_gaussians
                    self.deform.deform.init(opt=self.opt, init_pcl=original_gaussians.get_xyz, keep_all=True,
                                            force_init=True, reset_bbox=False, feature=self.gaussians.feature)
                    gaussians: GaussianModel = self.deform.deform.as_gaussians
                    gaussians._features_dc = torch.nn.Parameter(original_gaussians._features_dc)
                    gaussians._features_rest = torch.nn.Parameter(original_gaussians._features_rest)
                    gaussians._scaling = torch.nn.Parameter(original_gaussians._scaling)
                    gaussians._opacity = torch.nn.Parameter(original_gaussians._opacity)
                    gaussians._rotation = torch.nn.Parameter(original_gaussians._rotation)
                    if gaussians.fea_dim > 0:
                        gaussians.feature = torch.nn.Parameter(original_gaussians.feature)
                    print('Reset the optimizer of the deform model.')
                    self.deform.train_setting(self.opt)
                elif strategy == 'samp_hyper':
                    original_gaussians: GaussianModel = self.deform.deform.as_gaussians
                    if original_gaussians.get_xyz.shape[0] == 0:
                        print("[samp_hyper] Empty deform.as_gaussians after sync; falling back to render gaussians.")
                        original_gaussians = self.gaussians
                    time_num = 16
                    t_samp = torch.linspace(0, 1, time_num).cuda()
                    x = original_gaussians.get_xyz.detach()
                    trans_samp = []
                    for i in range(time_num):
                        time_input = t_samp[i:i + 1, None].expand_as(x[..., :1])
                        trans_samp.append(self.deform.deform.query_network(x=x, t=time_input)[
                                              'd_xyz'] * original_gaussians.motion_mask)
                    trans_samp = torch.stack(trans_samp, dim=1)
                    hyper_pcl = (trans_samp + original_gaussians.get_xyz[:, None]).reshape(
                        [original_gaussians.get_xyz.shape[0], -1])
                    dynamic_mask = original_gaussians.motion_mask[..., 0] > .5
                    if not self.opt.deform_downsamp_with_dynamic_mask:
                        dynamic_mask = torch.ones_like(dynamic_mask)
                    if dynamic_mask.sum().item() == 0:
                        print("[samp_hyper] Empty dynamic mask during node downsampling; falling back to all points.")
                        dynamic_mask = torch.ones_like(dynamic_mask)
                    idx = self.deform.deform.init(init_pcl=original_gaussians.get_xyz[dynamic_mask],
                                                  hyper_pcl=hyper_pcl[dynamic_mask], force_init=True, opt=self.opt,
                                                  reset_bbox=False, feature=self.gaussians.feature)
                    gaussians: GaussianModel = self.deform.deform.as_gaussians
                    gaussians._features_dc = torch.nn.Parameter(original_gaussians._features_dc[dynamic_mask][idx])
                    gaussians._features_rest = torch.nn.Parameter(original_gaussians._features_rest[dynamic_mask][idx])
                    gaussians._scaling = torch.nn.Parameter(original_gaussians._scaling[dynamic_mask][idx])
                    gaussians._opacity = torch.nn.Parameter(original_gaussians._opacity[dynamic_mask][idx])
                    gaussians._rotation = torch.nn.Parameter(original_gaussians._rotation[dynamic_mask][idx])
                    if gaussians.fea_dim > 0:
                        gaussians.feature = torch.nn.Parameter(original_gaussians.feature[dynamic_mask][idx])
                    gaussians.training_setup(self.opt)
                self.deform.deform.as_gaussians.optimizer.zero_grad(set_to_none=True)
                self.deform.optimizer.zero_grad()

            if self.iteration_node_rendering == self.opt.iterations_node_rendering - 1 and self.iteration_node_rendering > self.opt.iterations_node_sampling:
                # Just finish node training and has down sampled control nodes
                self.deform.deform.nodes.data[..., :3] = self.deform.deform.as_gaussians._xyz

            if not self.iteration_node_rendering == self.opt.iterations_node_sampling and not self.iteration_node_rendering == self.opt.iterations_node_rendering - 1:
                # Optimizer step
                self.deform.deform.as_gaussians.optimizer.step()
                self.deform.deform.as_gaussians.update_learning_rate(self.iteration_node_rendering)
                self.deform.deform.as_gaussians.optimizer.zero_grad(set_to_none=True)
                self.deform.update_learning_rate(self.iteration_node_rendering)
                self.deform.optimizer.step()
                self.deform.optimizer.zero_grad()

        self.deform.update(max(0, self.iteration_node_rendering - self.opt.node_warm_up))

        if self.dataset.load2gpu_on_the_fly:
            viewpoint_cam.load2device('cpu')

        self.iteration_node_rendering += 1


    def create_training_groups(self, T):
        # Pick a random Camera
        train_cams = self.scene.getTrainCameras().copy()
        def _fid_sort_key(cam):
            fid = getattr(cam, "fid", 0.0)
            if torch.is_tensor(fid):
                return float(fid.detach().reshape(-1)[0].cpu().item())
            return float(fid)
        # 1) Group cameras by view prefix.
        def get_prefix(cam):
            return self._camera_prefix(cam)
        buckets = defaultdict(list)
        for cam in train_cams:
            buckets[get_prefix(cam)].append(cam)
        # 2) Sort each view bucket by fid.
        for k in buckets:
            buckets[k] = sorted(buckets[k], key=_fid_sort_key)
        pos_neg_prob = self.opt.pos_net_prob if self.iteration < 60000 else 1
        pos_max_interval = self.opt.pos_max_interval if self.opt.pos_max_interval > T else T
        neg_max_interval = self.opt.neg_max_interval if self.opt.neg_max_interval > T else T-2
        pos_interval = [i for i in range(1, pos_max_interval + 1)]
        neg_interval = [-i for i in range(1, neg_max_interval + 1)]
        group_list = []

        # 3) Build temporal groups inside each view bucket.
        for prefix, cams in buckets.items():
            total_frame = len(cams)
            if total_frame < 2:
                continue
            for start_idx in range(total_frame - 1):
                # 3.1 Sample positive forward intervals or negative interpolation intervals.
                if random.random() < pos_neg_prob or T < 3:
                    interval = random.choice(pos_interval)
                else:
                    interval = random.choice(neg_interval)
                # 3.2 Positive interval: use real frames with a feasible stride.
                if interval > 0:
                    tmp_interval = interval
                    while tmp_interval > 0 and start_idx + (T - 1) * tmp_interval >= total_frame:
                        tmp_interval -= 1
                    if tmp_interval > 0:
                        indices = [start_idx + i * tmp_interval for i in range(T)]
                        group = [cams[i] for i in indices]
                        group_list.append(group)
                        continue
                    else:
                        # Fall back to the interpolation branch when no positive stride fits.
                        interval = random.randint(1 - T, -1)
                # 3.3 Negative interval: interpolate placeholder frames inside the current bucket.
                #     Positions marked in real_camera use real frames; the others use interpolated fid placeholders.
                if interval <= 0:
                    real_camera = [0] * T
                    # Keep real frame indices inside this bucket.
                    tmp_interval = interval
                    while tmp_interval > -T and start_idx + len(set(range(0, T, -tmp_interval))) - 1 >= total_frame:
                        tmp_interval -= 1
                    # Mark real-frame positions.
                    for i in range(0, T, -tmp_interval):
                        real_camera[i] = 1
                    real_positions = [i for i, flag in enumerate(real_camera) if flag]
                    pos_to_idx = {pos: idx for idx, pos in enumerate(real_positions)}
                    indices = [(start_idx + pos_to_idx[i] if real_camera[i] else -1) for i in range(T)]
                    # Linearly interpolate placeholder fids from adjacent real frames in the same view timeline.
                    fid_positions = real_positions[:2]  # The construction above guarantees at least two real frames.
                    fid0_raw = cams[indices[fid_positions[0]]].fid
                    fid1_raw = cams[indices[fid_positions[1]]].fid
                    fid0 = float(fid0_raw.detach().reshape(-1)[0].cpu().item()) if torch.is_tensor(fid0_raw) else float(fid0_raw)
                    fid1 = float(fid1_raw.detach().reshape(-1)[0].cpu().item()) if torch.is_tensor(fid1_raw) else float(fid1_raw)
                    step_count = fid_positions[1] - fid_positions[0]
                    fid_step = (fid1 - fid0) / step_count
                    if torch.is_tensor(fid0_raw):
                        fid_dtype, fid_device = fid0_raw.dtype, fid0_raw.device
                    else:
                        fid_dtype, fid_device = torch.float32, torch.device('cpu')
                    group = []
                    if torch.is_tensor(fid0_raw):
                        train_fids = [
                            torch.full_like(fid0_raw, float(fid0 + fid_step * i))
                            for i in range(T)
                        ]
                    else:
                        train_fids = [torch.tensor([fid0 + fid_step * i], dtype=fid_dtype, device=fid_device) for i in range(T)]
                    for idx, indice in enumerate(indices):
                        if indice >= 0:
                            group.append(cams[indice])
                        else:
                            # Create a placeholder frame with the interpolated fid.
                            group.append(self.get_placeholder_frame(train_fids[idx]))
                    group_list.append(group)
        cv_prob = float(getattr(self.opt, "cross_view_group_prob", 0.0))
        if cv_prob > 0 and len(buckets) > 1:
            group_list.extend(self.create_cross_view_training_groups(T, buckets, cv_prob))
        # 4) Combine groups from all view buckets.
        self.viewpoint_stack = group_list

    def _camera_prefix(self, cam):
        img_name = getattr(cam, "image_name", "")
        return img_name.split("_")[0] if "_" in img_name else "default"

    def _fid_value(self, fid):
        if torch.is_tensor(fid):
            return float(fid.detach().reshape(-1)[0].cpu().item())
        return float(fid)

    def create_cross_view_training_groups(self, T, buckets, cv_prob):
        cross_groups = []
        prefixes = [prefix for prefix, cams in buckets.items() if len(cams) >= T]
        if len(prefixes) < 2:
            return cross_groups
        base_groups = sum(max(0, len(cams) - T + 1) for cams in buckets.values())
        target_groups = max(1, int(base_groups * cv_prob))
        for _ in range(target_groups):
            ref_prefix = random.choice(prefixes)
            ref_cams = buckets[ref_prefix]
            start_idx = random.randint(0, len(ref_cams) - T)
            group = []
            for offset in range(T):
                ref_cam = ref_cams[start_idx + offset]
                target_fid = self._fid_value(ref_cam.fid)
                candidates = []
                for prefix in prefixes:
                    cams = buckets[prefix]
                    nearest = min(cams, key=lambda cam: abs(self._fid_value(cam.fid) - target_fid))
                    candidates.append(nearest)
                group.append(random.choice(candidates))
            if len({self._camera_prefix(cam) for cam in group}) > 1:
                cross_groups.append(group)
        return cross_groups

    def sample_cross_view_anchor_group(self, frame_group):
        if float(getattr(self.opt, "cross_view_anchor_prob", 0.0)) <= 0:
            return None
        if random.random() >= float(self.opt.cross_view_anchor_prob):
            return None
        train_cams = self.scene.getTrainCameras()
        if len(train_cams) == 0:
            return None
        buckets = defaultdict(list)
        for cam in train_cams:
            buckets[self._camera_prefix(cam)].append(cam)
        if len(buckets) < 2:
            return None
        for prefix in list(buckets.keys()):
            buckets[prefix] = sorted(buckets[prefix], key=lambda cam: self._fid_value(cam.fid))

        anchors = []
        tolerance = float(getattr(self.opt, "cross_view_time_tolerance", 1e-4))
        for base_frame in frame_group:
            if getattr(base_frame, "original_image", None) is None:
                return None
            target_fid = self._fid_value(base_frame.fid)
            base_prefix = self._camera_prefix(base_frame)
            candidates = []
            for prefix, cams in buckets.items():
                if prefix == base_prefix:
                    continue
                nearest = min(cams, key=lambda cam: abs(self._fid_value(cam.fid) - target_fid))
                if abs(self._fid_value(nearest.fid) - target_fid) <= tolerance:
                    candidates.append(nearest)
            if candidates:
                anchors.append(random.choice(candidates))
            else:
                return None
        return anchors if len(anchors) == len(frame_group) else None

    def render_anchor_loss(self, anchor_group):
        if not anchor_group:
            return None
        loaded = []
        if self.dataset.load2gpu_on_the_fly:
            for frame in anchor_group:
                frame.load2device('cuda')
                loaded.append(frame)
        try:
            fids = []
            for frame in anchor_group:
                fid_t = frame.fid if torch.is_tensor(frame.fid) else torch.tensor(frame.fid, device='cuda')
                fid_t = fid_t.to('cuda')
                if fid_t.dim() == 0:
                    fid_t = fid_t.unsqueeze(0)
                fids.append(fid_t)
            time_input = torch.stack(fids, dim=1).float().cuda()
            xyz = self.gaussians.get_xyz.detach()
            d_values = self.deform.step(
                xyz,
                time_input,
                train_mask=None,
                iteration=self.iteration,
                feature=self.gaussians.feature,
                motion_mask=self.gaussians.motion_mask,
                camera_center=[frame.camera_center for frame in anchor_group],
                knn_feature=self.gaussians.get_knn_feature.detach(),
            )
            render_group = render_batch(
                anchor_group,
                self.gaussians,
                self.pipe,
                self.background,
                d_values['d_xyz'],
                d_values['d_rotation'],
                d_values['d_scaling'],
                d_values.get('d_opacity', None),
                d_values.get('d_color', None),
                list(range(len(anchor_group))),
                random_bg_color=False,
                d_rot_as_res=self.deform.d_rot_as_res
            )
            preds = render_group["render"]
            gt_images = torch.stack([frame.original_image for frame in anchor_group], dim=0).cuda().to(preds.dtype)
            per_frame_l1 = (preds - gt_images).abs().mean(dim=(1, 2, 3))
            per_frame_ssim = torch.stack([ssim(preds[i:i + 1], gt_images[i:i + 1]) for i in range(preds.shape[0])], dim=0)
            lambda_dssim = getattr(self.opt, "lambda_dssim", 0.2)
            return ((1.0 - lambda_dssim) * per_frame_l1 + lambda_dssim * (1.0 - per_frame_ssim)).mean()
        finally:
            if self.dataset.load2gpu_on_the_fly:
                for frame in loaded:
                    frame.load2device('cpu')

    def _fid_tensor(self, fid, device='cuda'):
        fid_t = fid if torch.is_tensor(fid) else torch.tensor(fid, device=device)
        fid_t = fid_t.to(device)
        if fid_t.dim() == 0:
            fid_t = fid_t.unsqueeze(0)
        return fid_t

    def _load_frame_group_to_cuda(self, frame_group):
        if not self.dataset.load2gpu_on_the_fly:
            return
        for frame in frame_group:
            if frame.original_image is None:
                frame.fid = frame.fid.cuda()
            else:
                frame.load2device('cuda')

    def _release_frame_group_from_cuda(self, frame_group):
        if not self.dataset.load2gpu_on_the_fly:
            return
        for frame in frame_group:
            if frame.original_image is None:
                frame.fid = frame.fid.cpu()
            else:
                frame.load2device('cpu')

    def _real_frame_positions(self, frame_group):
        return [idx for idx, frame in enumerate(frame_group) if frame.original_image is not None]

    def _frame_fids(self, frame_group):
        return [self._fid_tensor(frame_group[i].fid) for i in range(self.deform.T)]

    def _image_reconstruction_loss(self, preds, gt_images):
        bs_real = preds.shape[0]
        per_frame_l1 = (preds - gt_images).abs().mean(dim=(1, 2, 3))
        per_frame_ssim = torch.stack([ssim(preds[i:i + 1], gt_images[i:i + 1]) for i in range(bs_real)], dim=0)
        lambda_dssim = getattr(self.opt, "lambda_dssim", 0.2)
        per_frame_loss = (1.0 - lambda_dssim) * per_frame_l1 + lambda_dssim * (1.0 - per_frame_ssim)

        Ll1 = torch.mean(per_frame_l1)
        use_topk = getattr(self.opt, "enable_topk", True) and not getattr(self.opt, "disable_topk_loss", False)
        topk_ratio = float(getattr(self.opt, "loss_topk_ratio", 0.6))
        if use_topk and bs_real > 2:
            k = max(2, round(topk_ratio * bs_real))
            topk_vals, _ = torch.topk(per_frame_loss, k=k, largest=True, sorted=False)
            loss_img = torch.mean(topk_vals)
        else:
            loss_img = per_frame_loss.mean()
        return loss_img * 0.9 + Ll1 * 0.1, Ll1

    def _apply_depth_loss(self, loss, pred_depths, frame_group, real_positions):
        lambda_depth = float(getattr(self.opt, "lambda_depth", 0.0))
        if frame_group[0].depth is None or lambda_depth <= 0:
            return loss
        gt_depths = torch.stack(
            [
                frame_group[i].depth if frame_group[i].depth.ndim == 3
                else frame_group[i].depth.unsqueeze(0)
                for i in real_positions
            ],
            dim=0,
        ).cuda().to(pred_depths.dtype)
        valid_mask = (gt_depths > 1e-5).detach()
        depth_l1 = ((pred_depths - gt_depths).abs() * valid_mask).sum() / (valid_mask.sum() + 1e-8)
        return (1 - lambda_depth) * loss + lambda_depth * depth_l1

    def _apply_dynamic_edge_loss(self, loss, preds, gt_images, bs_real):
        if (
            not getattr(self.opt, "with_dynamic_edge_loss", False)
            or self.iteration < int(getattr(self.opt, "dynamic_edge_start_iter", 12000))
            or bs_real <= 1
        ):
            return loss
        edge_loss, lap_loss = dynamic_edge_loss(
            preds,
            gt_images,
            tau_quantile=float(getattr(self.opt, "dynamic_edge_tau_quantile", 0.75)),
            blur_ks=int(getattr(self.opt, "dynamic_edge_mask_blur_ks", 3)),
        )
        if edge_loss is not None:
            loss = loss + float(getattr(self.opt, "lambda_dynamic_edge", 0.0)) * edge_loss
        if lap_loss is not None:
            loss = loss + float(getattr(self.opt, "lambda_dynamic_lap", 0.0)) * lap_loss
        return loss

    def _apply_motion_diff_loss(self, loss, preds, gt_images, bs_real, T):
        with_motion_loss = getattr(self.opt, "with_motion_diff_loss", False)
        if not (with_motion_loss and self.iteration > self.opt.warm_up + 20000 and bs_real > 1 and T > 1):
            return loss

        lambda_motion_diff = float(getattr(self.opt, "lambda_motion_diff", 0.2))
        lambda_motion_under = float(getattr(self.opt, "lambda_motion_under", 0.12))
        lambda_motion_dir = float(getattr(self.opt, "lambda_motion_dir", 0.012))
        motion_tau_q = float(getattr(self.opt, "motion_tau_quantile", 0.75))
        motion_blur_ks = int(getattr(self.opt, "motion_blur_ks", 3))

        I_tm1, I_t = gt_images[:-1], gt_images[1:]
        Ihat_tm1, Ihat_t = preds[:-1], preds[1:]

        with torch.no_grad():
            M = (I_t - I_tm1).abs().mean(1, keepdim=True)
            pad = motion_blur_ks // 2
            M_blur = torch.nn.functional.avg_pool2d(M, kernel_size=motion_blur_ks, stride=1, padding=pad)
            flat = M_blur.flatten(1)
            tau = torch.quantile(flat, motion_tau_q, dim=1, keepdim=True).view(-1, 1, 1, 1)
            beta = 0.1 * tau
            w = torch.sigmoid((M_blur - tau) / (beta + 1e-6)).detach()

        dI_gt = I_t - I_tm1
        dI_hat = Ihat_t - Ihat_tm1
        charb = lambda x: torch.sqrt(x * x + 1e-6)

        L_diff = (w * charb(dI_hat - dI_gt)).mean()
        amp_gt = dI_gt.abs().sum(dim=1, keepdim=True)
        amp_hat = dI_hat.abs().sum(dim=1, keepdim=True)
        L_under = (w * (amp_gt - amp_hat).clamp_min(0)).mean()

        num = (dI_hat * dI_gt).sum(dim=1, keepdim=True)
        den = dI_hat.norm(p=2, dim=1, keepdim=True) * dI_gt.norm(p=2, dim=1, keepdim=True) + 1e-6
        cos = num / den
        L_dir = (w * (1.0 - cos)).mean()

        return loss + lambda_motion_diff * L_diff + lambda_motion_under * L_under + lambda_motion_dir * L_dir

    def train_node_video_step(self, warmup=False):
        self.iter_start.record()

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if self.iteration % self.opt.oneupSHdegree_step == 0:
            self.gaussians.oneupSHdegree()

        # Pick a random Camera
        T = self.deform.T
        if not self.viewpoint_stack:
            self.create_training_groups(T)

        frame_group = self.viewpoint_stack.pop(random.randint(0, len(self.viewpoint_stack) - 1))
        self._load_frame_group_to_cuda(frame_group)
        real_positions = self._real_frame_positions(frame_group)

        total_frame = len(self.scene.getTrainCameras())
        time_interval = 1 / total_frame
        fids = self._frame_fids(frame_group)
        if self.deform.name == 'video_node':
            if self.iteration < self.opt.warm_up:
                d_xyz = [0.0] * T
                d_rotation = [0.0] * T
                d_scaling = [0.0] * T
                d_opacity = [0.0] * T
                d_color = [0.0] * T
            else:
                N = self.gaussians.get_xyz.shape[0]
                base_fids = torch.stack([fids[i] for i in range(T)], dim=1)

                mask = (torch.rand(T, device='cuda') < self.opt.mask_probability)  # shape: [T], bool
                if T == 1:
                    mask[:] = False
                else:
                    unmasked = (~mask).nonzero(as_tuple=False).squeeze(1)
                    if unmasked.numel() < 2:
                        # Unmask enough frames to keep at least two real supervision frames.
                        need = 2 - unmasked.numel()
                        masked_idx = mask.nonzero(as_tuple=False).squeeze(1)
                        if masked_idx.numel() >= need:
                            pick = masked_idx[torch.randperm(masked_idx.numel(), device='cuda')[:need]]
                            mask[pick] = False
                        else:
                            pick = torch.randperm(T, device='cuda')[:2]
                            mask[pick] = False

                train_mask = mask

                if not self.deform.deform.inited:
                    print('Notice that warping nodes are initialized with Gaussians!!!')
                    self.deform.deform.init(self.opt, self.gaussians.get_xyz.detach(), feature=self.gaussians.feature)

                if self.dataset.is_blender:
                    noise = torch.zeros(T, device='cuda')
                else:
                    noise = (torch.randn(1, T, device='cuda')
                             * time_interval
                             * self.smooth_term(self.iteration))  # (1, N)

                time_window = (base_fids + noise).float().cuda()

                d_values = self.deform.step(
                    self.gaussians.get_xyz.detach(),  # (N,3)
                    time_window,  # (T,)
                    train_mask=train_mask,
                    iteration=self.iteration,
                    feature=self.gaussians.feature,
                    motion_mask=self.gaussians.motion_mask,
                    camera_center=[frame_group[i].camera_center for i in range(T)],
                    knn_feature=self.gaussians.get_knn_feature.detach(),
                )

                d_xyz = d_values['d_xyz']
                d_rotation = d_values['d_rotation']
                d_scaling = d_values['d_scaling']
                d_opacity = d_values.get('d_opacity', None)
                d_color = d_values.get('d_color', None)

                if self.iteration < self.opt.warm_up:
                    d_xyz, d_rotation, d_scaling, d_opacity, d_color = d_xyz.detach(), d_rotation.detach(), d_scaling.detach(), d_opacity.detach() if d_opacity is not None else None, d_color.detach() if d_color is not None else None
                elif self.iteration < self.opt.dynamic_color_warm_up:
                    d_color = d_color.detach() if d_color is not None else None

        # Render
        random_bg = (not self.dataset.white_background and self.opt.random_bg_color
                     and self.opt.gt_alpha_mask_as_scene_mask
                     and frame_group[0].gt_alpha_mask is not None)

        render_group = render_batch(
            frame_group,
            self.gaussians,
            self.pipe,
            self.background,
            d_xyz, d_rotation, d_scaling,
            d_opacity, d_color,
            real_positions,
            random_bg_color=random_bg,
            d_rot_as_res=self.deform.d_rot_as_res
        )

        images, pred_depths, viewspace_point_tensor, visibility_filter, radii = (
            render_group["render"],
            render_group["depth"],
            render_group["viewspace_points"],
            render_group["visibility_filter"],
            render_group["radii"]
        )

        preds = images  # (B,C,H,W), B=len(real_positions)
        gt_images = torch.stack([frame_group[i].original_image for i in real_positions], dim=0).cuda().to(preds.dtype)
        bs_real = preds.shape[0]

        loss, Ll1 = self._image_reconstruction_loss(preds, gt_images)
        loss = self._apply_depth_loss(loss, pred_depths, frame_group, real_positions)

        lambda_cv = float(getattr(self.opt, "lambda_cross_view_anchor", 0.0))
        if lambda_cv > 0 and self.iteration > self.opt.warm_up and self.deform.name == 'video_node':
            anchor_loss = self.render_anchor_loss(self.sample_cross_view_anchor_group(frame_group))
            if anchor_loss is not None:
                loss = loss + lambda_cv * anchor_loss

        loss = self._apply_dynamic_edge_loss(loss, preds, gt_images, bs_real)

        loss = self._apply_motion_diff_loss(loss, preds, gt_images, bs_real, T)

        if self.iteration > self.opt.warm_up:
            loss = loss + self.deform.reg_loss

        loss.backward()

        self.iter_end.record()

        with torch.no_grad():
            # Progress bar
            self.ema_loss_for_log = 0.4 * loss.item() + 0.6 * self.ema_loss_for_log
            if self.iteration % 10 == 0:
                self.progress_bar.set_postfix({"Loss": f"{self.ema_loss_for_log:.{7}f}"})
                self.progress_bar.update(10)
            if self.iteration == self.opt.iterations:
                self.progress_bar.close()


            saved_this_iteration = False
            if self.iteration in self.saving_iterations or self.iteration == self.opt.warm_up - 1:
                self._save_checkpoint("Gaussians")
                saved_this_iteration = True
            # Log and save
            cur_psnr, cur_ssim, cur_lpips, cur_ms_ssim, cur_alex_lpips, cur_tlp = training_report(self.tb_writer,
                                                                                         self.iteration,
                                                                                         Ll1, loss, l1_loss,
                                                                                         self.iter_start.elapsed_time(
                                                                                             self.iter_end),
                                                                                         self.testing_iterations,
                                                                                         self.scene, render_batch,
                                                                                         (self.pipe,
                                                                                          self.background),
                                                                                         self.deform,
                                                                                         self.dataset.load2gpu_on_the_fly,
                                                                                         self.progress_bar)

            if not hasattr(self, "best_tlp"):
                self.best_tlp = float('inf')

            if self.iteration in self.testing_iterations:
                if cur_psnr.item() > self.best_psnr:
                    self.best_psnr = cur_psnr.item()
                    self.best_iteration = self.iteration
                    self.best_ssim = cur_ssim.item()
                    self.best_ms_ssim = cur_ms_ssim.item()
                    self.best_lpips = cur_lpips.item()
                    self.best_alex_lpips = cur_alex_lpips.item()
                    self.best_tlp = cur_tlp.item()
                    if not saved_this_iteration and not getattr(self.args, "no_save_best", False):
                        self._save_checkpoint("new best Gaussians")
                        saved_this_iteration = True
                if (
                    self.args.target_test_psnr_stop > 0
                    and cur_psnr.item() >= self.args.target_test_psnr_stop
                ):
                    if not saved_this_iteration:
                        self._save_checkpoint("target PSNR checkpoint")
                    print(
                        "\n[ITER {}] Target test PSNR reached: current PSNR={:.5f}, "
                        "target={:.5f}. Stopping early.".format(
                            self.iteration,
                            cur_psnr.item(),
                            self.args.target_test_psnr_stop,
                        )
                    )
                    self.stop_training = True
                    if self.progress_bar is not None:
                        self.progress_bar.close()
                    return
                elif (
                    self.args.collapse_guard_psnr_drop > 0
                    and self.best_psnr >= self.args.collapse_guard_min_best_psnr
                ):
                    psnr_drop = self.best_psnr - cur_psnr.item()
                    if psnr_drop >= self.args.collapse_guard_psnr_drop:
                        print(
                            "\n[ITER {}] Metric collapse detected: current PSNR={:.5f}, "
                            "best PSNR={:.5f} at iter {}, drop={:.5f}. Stopping early and "
                            "keeping the best saved checkpoint.".format(
                                self.iteration,
                                cur_psnr.item(),
                                self.best_psnr,
                                self.best_iteration,
                                psnr_drop,
                            )
                        )
                        self.stop_training = True
                        if self.progress_bar is not None:
                            self.progress_bar.close()
                        return


            if self.iteration > self.opt.node_densify_from_iter and self.iteration % self.opt.node_densification_interval == 0 and self.iteration < self.opt.node_densify_until_iter and self.iteration > self.opt.warm_up or self.iteration == self.opt.node_force_densify_prune_step:
                # Nodes densify
                self.deform.densify(max_grad=self.opt.densify_grad_threshold, x=self.gaussians.get_xyz, knn_feature=None, x_grad=self.gaussians.xyz_gradient_accum / self.gaussians.denom, feature=self.gaussians.feature, force_dp=(self.iteration == self.opt.node_force_densify_prune_step))

            self.densify_and_prune(viewspace_point_tensor, visibility_filter, radii, T)

            if self.iteration_node_rendering == self.opt.iterations_node_rendering - 1 and self.iteration_node_rendering > self.opt.iterations_node_sampling:
                # Just finish node training and has down sampled control nodes
                self.deform.deform.nodes.data[..., :3] = self.deform.deform.as_gaussians._xyz

            # Optimizer step
            if self.iteration < self.opt.iterations:
                self.gaussians.optimizer.step()
                self.gaussians.update_learning_rate(self.iteration)
                self.gaussians.optimizer.zero_grad(set_to_none=True)
                self.deform.optimizer.step()
                self.deform.optimizer.zero_grad()
                self.deform.update_learning_rate(self.iteration)

        self.deform.update(max(0, self.iteration - self.opt.warm_up))

        self._release_frame_group_from_cuda(frame_group)

        tlp_str = 'nan' if (not torch.is_tensor(cur_tlp) or not torch.isfinite(cur_tlp)) else ('%.5f' % cur_tlp.item())
        self.progress_bar.set_description(
            "Best PSNR={} in Iteration {}, SSIM={}, LPIPS={}, MS-SSIM={}, Alex-LPIPS={}, TLP={}".format(
                '%.5f' % self.best_psnr, self.best_iteration, '%.5f' % self.best_ssim, '%.5f' % self.best_lpips,
                '%.5f' % self.best_ms_ssim, '%.5f' % self.best_alex_lpips, tlp_str))
        self.iteration += 1


def prepare_output_and_logger(args):
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str = os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])

    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok=True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer


def main():
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)

    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument('--no_save_best', action='store_true', default=False,
                        help="Do not save extra checkpoints when test PSNR improves; metrics are still tracked.")
    parser.add_argument('--keep_recent_checkpoints', type=int, default=0,
                        help="If >0, prune old checkpoint folders after each save while keeping best/current/final and this many recent saves.")
    parser.add_argument('--keep_checkpoint_iterations', nargs="+", type=int, default=[],
                        help="Checkpoint iterations that should never be pruned when keep_recent_checkpoints is enabled.")
    parser.add_argument('--collapse_guard_psnr_drop', type=float, default=15.0,
                        help="Stop after a test PSNR drop this far below best PSNR; set <=0 to disable.")
    parser.add_argument('--collapse_guard_min_best_psnr', type=float, default=25.0,
                        help="Only enable collapse guard after best test PSNR reaches this value.")
    parser.add_argument('--target_test_psnr_stop', type=float, default=0.0,
                        help="Stop after a test evaluation reaches this PSNR; set <=0 to disable.")
    parser.add_argument("--profile_config", type=str, default="",
                        help="Path to release profile directory. Defaults to configs/profiles.")
    parser.add_argument("--profile_dataset", type=str, default="",
                        help="Optional dataset key override for release profiles, e.g. n3v or hypernerf.")
    parser.add_argument("--profile_scene", type=str, default="",
                        help="Optional scene key override for release profiles.")
    parser.add_argument("--no_profile", action="store_true",
                        help="Disable automatic global/dataset/scene release profiles.")
    parser.add_argument("--test_iterations", nargs="+", type=int,
                        default=[1, 3000, 5000, 6000, 7_000] + list(range(8000, 100_001, 1000)))
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 10_000, 20_000, 30_000, 40000, 80000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--deform-type", type=str, default='mango_node')

    args = parser.parse_args(sys.argv[1:])
    args = apply_dataset_preset(args)
    args.save_iterations.append(args.iterations)
    args.save_iterations = sorted(set(args.save_iterations))

    if not args.model_path.endswith(args.deform_type):
        args.model_path = os.path.join(os.path.dirname(os.path.normpath(args.model_path)), os.path.basename(os.path.normpath(args.model_path)) + f'_{args.deform_type}')

    print("Optimizing " + args.model_path)
    safe_state(args.quiet)

    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    trainer = MangoTrainer(
        args=args,
        dataset=lp.extract(args),
        opt=op.extract(args),
        pipe=pp.extract(args),
        testing_iterations=args.test_iterations,
        saving_iterations=args.save_iterations,
    )

    remaining_iterations = args.iterations
    if trainer.scene.loaded_iter is not None:
        remaining_iterations = max(0, args.iterations - trainer.scene.loaded_iter + 1)
        print(
            "Resuming from iteration {}; running {} remaining iterations to target {}.".format(
                trainer.scene.loaded_iter, remaining_iterations, args.iterations
            )
        )

    trainer.train(remaining_iterations)

    # All done
    print("\nTraining complete.")


if __name__ == "__main__":
    main()
