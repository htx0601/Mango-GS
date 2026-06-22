#!/usr/bin/env python3

from __future__ import annotations

import argparse
import os
import time
from collections import defaultdict

import imageio
import numpy as np
import torch
import torchvision
from pytorch_msssim import ms_ssim

from arguments.__init__video import ModelParams, OptimizationParams, PipelineParams, apply_dataset_preset, get_combined_args
from gaussian_renderer import GaussianModel, render_batch
from scene import DeformModel
from scene.dataset_readers import readCamerasFromNpy, readNerfiesInfo, sceneLoadTypeCallbacks
from utils.camera_utils import cameraList_from_camInfos
from utils.general_utils import safe_state
from utils.image_utils import alex_lpips, lpips, psnr, ssim as ssim_func


def load_test_cameras(dataset, args):
    source = dataset.source_path
    if os.path.exists(os.path.join(source, "poses_bounds.npy")):
        cam_infos = readCamerasFromNpy(
            source,
            "poses_bounds.npy",
            split="test",
            hold_id=[0],
            num_images=getattr(args, "n3v_num_images", 0),
            video_indices=[0],
            intrinsics_mode=getattr(args, "n3v_intrinsics_mode", "scaled"),
            camera_id_mode=getattr(args, "n3v_camera_id_mode", "position"),
        )
    elif os.path.exists(os.path.join(source, "points3d.ply")):
        cam_infos = readNerfiesInfo(source, True).test_cameras
    elif os.path.exists(os.path.join(source, "sparse")) or os.path.exists(os.path.join(source, "colmap_sparse")):
        cam_infos = sceneLoadTypeCallbacks["Colmap"](source, dataset.images, dataset.eval).test_cameras
    else:
        raise RuntimeError(f"Unsupported render source: {source}")
    return cameraList_from_camInfos(cam_infos, 1.0, args)


def _batched_window(sorted_views, start_idx, window_size):
    if len(sorted_views) == 0:
        return [], []

    group = list(sorted_views[start_idx:start_idx + window_size])
    real_positions = list(range(len(group)))
    if len(group) < window_size:
        group.extend([group[-1]] * (window_size - len(group)))
    return group, real_positions


def _center_window(sorted_views, center_idx, window_size):
    if len(sorted_views) == 0:
        return [], 0
    if len(sorted_views) >= window_size:
        start = center_idx - window_size // 2
        start = max(0, min(start, len(sorted_views) - window_size))
        group = sorted_views[start:start + window_size]
        return group, center_idx - start

    group = list(sorted_views)
    group.extend([sorted_views[-1]] * (window_size - len(group)))
    return group, center_idx


def _iter_video_windows(sorted_views, window_size, mode):
    if mode == "block":
        for start in range(0, len(sorted_views), window_size):
            group, real_positions = _batched_window(sorted_views, start, window_size)
            yield group, real_positions, [start + pos for pos in real_positions]
    elif mode == "center":
        for frame_idx in range(len(sorted_views)):
            group, target_pos = _center_window(sorted_views, frame_idx, window_size)
            yield group, [target_pos], [frame_idx]
    else:
        raise ValueError(f"Unknown video window mode: {mode}")


def _pad_video_frames(frames, macro_block_size=16):
    if macro_block_size <= 1:
        return frames, (0, 0)
    if frames.ndim != 4:
        raise ValueError(f"Expected video frames with shape (N, H, W, C), got {frames.shape}")

    _, height, width, _ = frames.shape
    pad_h = (-height) % macro_block_size
    pad_w = (-width) % macro_block_size
    if pad_h == 0 and pad_w == 0:
        return frames, (0, 0)

    padded = np.pad(frames, ((0, 0), (0, pad_h), (0, pad_w), (0, 0)), mode="edge")
    return padded, (pad_h, pad_w)


def _write_video(path, frames, fps, quality=8):
    frames, (pad_h, pad_w) = _pad_video_frames(frames)
    if pad_h or pad_w:
        print(
            f"[video] padded frames from {frames.shape[2] - pad_w}x{frames.shape[1] - pad_h} "
            f"to {frames.shape[2]}x{frames.shape[1]} before encoding"
        )
    imageio.mimwrite(path, frames, fps=fps, quality=quality, macro_block_size=1)


def render_test_video(
    model_path,
    load2gpu_on_the_fly,
    iteration,
    views,
    gaussians,
    pipeline,
    background,
    deform,
    video_fps=10,
    video_window_mode="block",
):
    out_dir = os.path.join(model_path, "test", f"video_{iteration}")
    renders_dir = os.path.join(out_dir, "renders")
    gts_dir = os.path.join(out_dir, "gt")
    depth_dir = os.path.join(out_dir, "depth")
    os.makedirs(renders_dir, exist_ok=True)
    os.makedirs(gts_dir, exist_ok=True)
    os.makedirs(depth_dir, exist_ok=True)

    to8b = lambda x: (255 * np.clip(x, 0, 1)).astype(np.uint8)

    buckets = defaultdict(list)
    for view in views:
        img_name = getattr(view, "image_name", "")
        prefix = img_name.split("_")[0] if "_" in img_name else "default"
        buckets[prefix].append(view)

    render_modes = ["block", "center"] if video_window_mode == "both" else [video_window_mode]

    for prefix, vlist in buckets.items():
        sorted_views = sorted(vlist, key=lambda vv: float(vv.fid.item()))
        for mode_name in render_modes:
            mode_suffix = "" if mode_name == "block" else f"_{mode_name}"
            sub_renders = os.path.join(renders_dir, prefix + mode_suffix)
            sub_gts = os.path.join(gts_dir, prefix + mode_suffix)
            sub_depth = os.path.join(depth_dir, prefix + mode_suffix)
            os.makedirs(sub_renders, exist_ok=True)
            os.makedirs(sub_gts, exist_ok=True)
            os.makedirs(sub_depth, exist_ok=True)

            frames = []
            psnr_list, ssim_list, lpips_list = [], [], []
            ms_ssim_list, alex_lpips_list = [], []
            total_render_secs = 0.0
            total_render_frames = 0

            for group, real_positions, frame_indices in _iter_video_windows(sorted_views, deform.T, mode_name):
                if not real_positions:
                    continue
                if load2gpu_on_the_fly:
                    for group_view in group:
                        group_view.load2device()

                fids = torch.tensor(
                    [group_view.fid.item() for group_view in group],
                    device="cuda",
                    dtype=torch.float32,
                ).unsqueeze(0)

                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                t0 = time.perf_counter()
                dvals = deform.step(
                    gaussians.get_xyz.detach(),
                    fids,
                    feature=gaussians.feature,
                    motion_mask=gaussians.motion_mask,
                    camera_center=[group_view.camera_center for group_view in group],
                )
                pkg = render_batch(
                    group,
                    gaussians,
                    pipeline,
                    background,
                    dvals["d_xyz"],
                    dvals["d_rotation"],
                    dvals["d_scaling"],
                    dvals.get("d_opacity", None),
                    dvals.get("d_color", None),
                    real_positions=real_positions,
                )
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                total_render_secs += time.perf_counter() - t0
                total_render_frames += len(real_positions)

                for out_i, (pos, frame_idx) in enumerate(zip(real_positions, frame_indices)):
                    view = group[pos]
                    img = torch.clamp(pkg["render"][out_i], 0.0, 1.0)
                    gt = torch.clamp(view.original_image.to(img.device), 0.0, 1.0)
                    dep = pkg["depth"][out_i] / (pkg["depth"][out_i].max() + 1e-5)
                    psnr_list.append(psnr(img[None], gt[None]).mean())
                    ssim_list.append(ssim_func(img[None], gt[None], data_range=1.0).mean())
                    lpips_list.append(lpips(img[None], gt[None]).mean())
                    ms_ssim_list.append(ms_ssim(img[None], gt[None], data_range=1.0).mean())
                    alex_lpips_list.append(alex_lpips(img[None], gt[None]).mean())
                    torchvision.utils.save_image(img, os.path.join(sub_renders, f"{frame_idx:05d}.png"))
                    torchvision.utils.save_image(gt, os.path.join(sub_gts, f"{frame_idx:05d}.png"))
                    torchvision.utils.save_image(dep, os.path.join(sub_depth, f"{frame_idx:05d}.png"))
                    frames.append(to8b(img[:3].permute(1, 2, 0).detach().cpu().numpy()))

            if frames:
                video_np = np.stack(frames, axis=0)
                video_path = os.path.join(sub_renders, f"test_{prefix}{mode_suffix}.mp4")
                _write_video(video_path, video_np, fps=video_fps, quality=8)
                print(f"[video mode:{mode_name}] saved to {video_path}")

            if psnr_list:
                psnr_test = torch.stack(psnr_list).mean()
                ssim_test = torch.stack(ssim_list).mean()
                lpips_test = torch.stack(lpips_list).mean()
                ms_ssim_test = torch.stack(ms_ssim_list).mean()
                alex_lpips_test = torch.stack(alex_lpips_list).mean()
                fps_render = total_render_frames / total_render_secs if total_render_secs > 0 else float("nan")
                print(
                    f"\n[video mode:{mode_name}] {prefix} Iter {iteration} | PSNR {psnr_test:.4f} "
                    f"SSIM {ssim_test:.4f} LPIPS {lpips_test:.4f} MS-SSIM {ms_ssim_test:.4f} "
                    f"ALEX-LPIPS {alex_lpips_test:.4f}"
                )
                print(
                    f"[video mode:{mode_name}] Rendered {total_render_frames} frames in {total_render_secs:.3f}s "
                    f"-> FPS (deform+render) = {fps_render:.2f}"
                )


def main() -> int:
    parser = argparse.ArgumentParser(description="Render Mango-GS test images and videos.")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    OptimizationParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--mode", default="video", choices=["video"], help="Only video rendering is supported in the public release.")
    parser.add_argument("--video_fps", default=10, type=int)
    parser.add_argument("--video_window_mode", default="block", choices=["block", "center", "both"])
    parser.add_argument("--profile_config", type=str, default="")
    parser.add_argument("--profile_dataset", type=str, default="")
    parser.add_argument("--profile_scene", type=str, default="")
    parser.add_argument("--no_profile", action="store_true")
    args = get_combined_args(parser)
    args = apply_dataset_preset(args)
    if not args.model_path.endswith(args.deform_type):
        args.model_path = os.path.join(
            os.path.dirname(os.path.normpath(args.model_path)),
            os.path.basename(os.path.normpath(args.model_path)) + f"_{args.deform_type}",
        )
    safe_state(args.quiet)
    dataset = model.extract(args)
    pipe = pipeline.extract(args)

    deform = DeformModel(
        K=dataset.K,
        deform_type=dataset.deform_type,
        T=dataset.T,
        is_blender=dataset.is_blender,
        skinning=dataset.skinning,
        hyper_dim=dataset.hyper_dim,
        node_num=dataset.node_num,
        pred_opacity=dataset.pred_opacity,
        pred_color=dataset.pred_color,
        use_hash=dataset.use_hash,
        hash_time=dataset.hash_time,
        d_rot_as_res=dataset.d_rot_as_res,
        local_frame=dataset.local_frame,
        progressive_brand_time=dataset.progressive_brand_time,
        max_d_scale=dataset.max_d_scale,
        enable_learned_metric=dataset.enable_learned_metric,
    )

    with torch.no_grad():
        deform.load_weights(dataset.model_path, iteration=args.iteration)
        gs_fea_dim = deform.deform.node_num if dataset.skinning and "node" in deform.name else dataset.hyper_dim
        gaussians = GaussianModel(dataset.sh_degree, fea_dim=gs_fea_dim, with_motion_mask=dataset.gs_with_motion_mask)
        gaussians.load_ply(
            os.path.join(dataset.model_path, "point_cloud", f"iteration_{args.iteration}", "point_cloud.ply")
        )
        views = load_test_cameras(dataset, args)
        background = torch.tensor([1, 1, 1] if dataset.white_background else [0, 0, 0], dtype=torch.float32, device="cuda")
        render_test_video(
            dataset.model_path,
            dataset.load2gpu_on_the_fly,
            args.iteration,
            views,
            gaussians,
            pipe,
            background,
            deform,
            video_fps=args.video_fps,
            video_window_mode=args.video_window_mode,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
