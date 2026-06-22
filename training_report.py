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
from collections import defaultdict
import torch
import torch.nn.functional as F
from scene import Scene
import uuid
from utils.image_utils import psnr, lpips, alex_lpips
from utils.image_utils import ssim as ssim_func
from piq import LPIPS
lpips = LPIPS()
from argparse import Namespace
from pytorch_msssim import ms_ssim

try:
    from torch.utils.tensorboard import SummaryWriter

    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False


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


def _is_temporal_deform(deform) -> bool:
    name = getattr(deform, "name", "")
    return any(token in name for token in ("video", "node")) and int(getattr(deform, "T", 1)) > 1


def _lpips_resize_pair(image_a, image_b, min_size=64):
    height, width = image_a.shape[-2:]
    if min(height, width) >= min_size:
        return image_a, image_b
    scale = min_size / max(1, min(height, width))
    size = (max(min_size, int(round(height * scale))), max(min_size, int(round(width * scale))))
    return (
        F.interpolate(image_a, size=size, mode="bilinear", align_corners=False),
        F.interpolate(image_b, size=size, mode="bilinear", align_corners=False),
    )


def temporal_lpips_delta(pred_images, gt_images):
    if pred_images.shape[0] < 2 or gt_images.shape[0] < 2:
        return None
    vals = []
    for frame_idx in range(1, pred_images.shape[0]):
        pred_cur, pred_prev = _lpips_resize_pair(pred_images[frame_idx:frame_idx + 1], pred_images[frame_idx - 1:frame_idx])
        gt_cur, gt_prev = _lpips_resize_pair(gt_images[frame_idx:frame_idx + 1], gt_images[frame_idx - 1:frame_idx])
        vals.append(torch.abs(lpips(pred_cur, pred_prev) - lpips(gt_cur, gt_prev)).reshape(()))
    return torch.stack(vals).mean() if vals else None


def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene: Scene, renderFunc, renderArgs, deform, load2gpu_on_the_fly, progress_bar=None):
    def _fid_sort_key(cam):
        fid = getattr(cam, "fid", 0.0)
        if torch.is_tensor(fid):
            return float(fid.detach().reshape(-1)[0].cpu().item())
        return float(fid)

    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

    test_psnr = 0.0
    test_ssim = 0.0
    test_lpips = 1e10
    test_ms_ssim = 0.0
    test_alex_lpips = 1e10
    test_tlp = 1e10
    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        is_multiframe = _is_temporal_deform(deform)
        T = getattr(deform, 'T', 1)

        train_eval_cameras = scene.getTrainCameras()
        if len(train_eval_cameras) > 0:
            if is_multiframe:
                # For multi-view video datasets such as N3V, sampling every few global
                # indices breaks per-view temporal continuity, which leaves some buckets
                # with fewer than T frames and causes empty evaluation groups.
                def _prefix(cam):
                    name = getattr(cam, "image_name", "")
                    return name.split("_")[0] if "_" in name else "default"

                grouped_train = defaultdict(list)
                for cam in train_eval_cameras:
                    grouped_train[_prefix(cam)].append(cam)

                selected_train = []
                max_views = 3
                frames_per_view = max(T, min(2 * T, 12))
                for _, cams in sorted(grouped_train.items()):
                    cams = sorted(cams, key=_fid_sort_key)
                    if len(cams) < T:
                        continue
                    selected_train.extend(cams[:frames_per_view])
                    max_views -= 1
                    if max_views == 0:
                        break
                train_eval_cameras = selected_train
            else:
                train_eval_cameras = [train_eval_cameras[idx % len(train_eval_cameras)] for idx in range(10, 100, 5)]

        validation_configs = ({'name': 'test',
                               'cameras': scene.getTestCameras()},
                              {'name': 'train',
                               'cameras': train_eval_cameras})
        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                # images = torch.tensor([], device="cuda")
                # gts = torch.tensor([], device="cuda")
                psnr_list, ssim_list, lpips_list, l1_list = [], [], [], []
                ms_ssim_list, alex_lpips_list = [], []
                tlp_list = []

                if is_multiframe:
                    def _prefix(cam):
                        name = getattr(cam, "image_name", "")
                        return name.split("_")[0] if "_" in name else "default"

                    buckets = defaultdict(list)
                    for c in config['cameras']:
                        buckets[_prefix(c)].append(c)

                    for vname, cams in buckets.items():
                        sorted_cams = sorted(cams, key=_fid_sort_key)

                        groups = [sorted_cams[i:i + T] for i in range(0, len(sorted_cams) - T + 1, T)]

                        for idx, viewpoints in enumerate(groups):
                            if load2gpu_on_the_fly:
                                for cam in viewpoints:
                                    cam.load2device()
                            xyz = scene.gaussians.get_xyz.detach()

                            # Normalize fid shape/device.
                            fids = []
                            for c in viewpoints:
                                fid = c.fid
                                fid_t = fid if torch.is_tensor(fid) else torch.tensor(fid, device='cuda')
                                fid_t = fid_t.to('cuda')
                                if fid_t.dim() == 0:
                                    fid_t = fid_t.unsqueeze(0)
                                fids.append(fid_t)
                            fids = torch.stack(fids, dim=0)
                            N = xyz.shape[0]
                            time_input = torch.stack([fids[i] for i in range(T)], dim=1)  # (1,T)

                            d_values = deform.step(
                                xyz, time_input,
                                feature=scene.gaussians.feature,
                                motion_mask=scene.gaussians.motion_mask,
                                camera_center=[c.camera_center for c in viewpoints],
                                knn_feature=scene.gaussians.get_knn_feature,
                                is_training=False
                            )

                            pkg = renderFunc(
                                viewpoints, scene.gaussians, *renderArgs,
                                d_xyz_stack=d_values['d_xyz'],  # (N,T,3)
                                d_rotation_stack=d_values['d_rotation'],  # (N,T,4)
                                d_scaling_stack=d_values['d_scaling'],  # (N,T,3)
                                d_opacity_stack=d_values.get('d_opacity'),
                                d_color_stack=d_values.get('d_color')
                            )

                            images = torch.clamp(pkg["render"], 0, 1)  # (T,C,H,W)
                            gt_images = torch.stack([c.original_image.to("cuda") for c in viewpoints], dim=0).clamp(0, 1)
                            if tb_writer and (idx % 10 == 0):
                                for f, cam in enumerate(viewpoints):
                                    img = images[f]
                                    tb_writer.add_images(
                                        f"{config['name']}_view_{vname}/{cam.image_name}/frame_{f:02d}/render",
                                        img.unsqueeze(0), global_step=iteration)

                                if testing_iterations and iteration == testing_iterations[0]:
                                        gt = gt_images[f]
                                        tb_writer.add_images(
                                            f"{config['name']}_view_{vname}/{cam.image_name}/frame_{f:02d}/ground_truth",
                                            gt.unsqueeze(0), global_step=iteration)

                            for img, gt in zip(images, gt_images):
                                l1_list.append(l1_loss(img[None], gt[None]).mean())
                                psnr_list.append(psnr(img[None], gt[None]).mean())
                                ssim_list.append(ssim_func(img[None], gt[None], data_range=1.).mean())
                                lpips_list.append(lpips(img[None], gt[None]).mean())
                                ms_ssim_list.append(ms_ssim(img[None], gt[None], data_range=1.).mean())
                                alex_lpips_list.append(alex_lpips(img[None], gt[None]).mean())

                            tlp_val = temporal_lpips_delta(images, gt_images)
                            if tlp_val is not None:
                                tlp_list.append(tlp_val)

                else:
                    for idx, viewpoint in enumerate(config['cameras']):
                        if load2gpu_on_the_fly:
                            viewpoint.load2device()
                        fid = viewpoint.fid
                        xyz = scene.gaussians.get_xyz

                        if deform.name == 'mlp':
                            time_input = fid.unsqueeze(0).expand(xyz.shape[0], -1)
                        elif deform.name == 'node':
                            time_input = deform.deform.expand_time(fid)
                        else:
                            time_input = 0

                        d_values = deform.step(xyz.detach(), time_input, feature=scene.gaussians.feature,
                                               is_training=False, motion_mask=scene.gaussians.motion_mask,
                                               camera_center=viewpoint.camera_center)
                        d_xyz, d_rotation, d_scaling, d_opacity, d_color = d_values['d_xyz'], d_values['d_rotation'], \
                        d_values['d_scaling'], d_values['d_opacity'], d_values['d_color']

                        image = torch.clamp(
                            renderFunc(viewpoint, scene.gaussians, *renderArgs, d_xyz=d_xyz, d_rotation=d_rotation,
                                       d_scaling=d_scaling, d_opacity=d_opacity, d_color=d_color,
                                       d_rot_as_res=deform.d_rot_as_res)["render"], 0.0, 1.0)
                        gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)

                        l1_list.append(l1_loss(image[None], gt_image[None]).mean())
                        psnr_list.append(psnr(image[None], gt_image[None]).mean())
                        ssim_list.append(ssim_func(image[None], gt_image[None], data_range=1.).mean())
                        lpips_list.append(lpips(image[None], gt_image[None]).mean())
                        ms_ssim_list.append(ms_ssim(image[None], gt_image[None], data_range=1.).mean())
                        alex_lpips_list.append(alex_lpips(image[None], gt_image[None]).mean())

                        # images = torch.cat((images, image.unsqueeze(0)), dim=0)
                        # gts = torch.cat((gts, gt_image.unsqueeze(0)), dim=0)

                        if load2gpu_on_the_fly:
                            viewpoint.load2device('cpu')
                        if tb_writer and (idx % 5 == 0):
                            tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name),
                                                 image[None], global_step=iteration)
                            if iteration == testing_iterations[0]:
                                tb_writer.add_images(
                                    config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name),
                                    gt_image[None], global_step=iteration)

                if not l1_list:
                    if progress_bar is None:
                        print(f"\n[ITER {iteration}] Skipping {config['name']} evaluation: no complete frame groups.")
                    else:
                        progress_bar.set_description(
                            f"\n[ITER {iteration}] Skipping {config['name']} evaluation: no complete frame groups."
                        )
                    continue

                l1_test = torch.stack(l1_list).mean()
                psnr_test = torch.stack(psnr_list).mean()
                ssim_test = torch.stack(ssim_list).mean()
                lpips_test = torch.stack(lpips_list).mean()
                ms_ssim_test = torch.stack(ms_ssim_list).mean()
                alex_lpips_test = torch.stack(alex_lpips_list).mean()


                if len(tlp_list) > 0:
                    tlp_test = torch.stack(tlp_list).mean()
                else:
                    tlp_test = torch.tensor(float('nan'), device=psnr_test.device)

                if config['name'] == 'test' or len(validation_configs[0]['cameras']) == 0:
                    test_psnr = psnr_test
                    test_ssim = ssim_test
                    test_lpips = lpips_test
                    test_ms_ssim = ms_ssim_test
                    test_alex_lpips = alex_lpips_test
                    test_tlp = tlp_test

                msg = ("\n[ITER {}] Evaluating {}: "
                       "L1 {} PSNR {} SSIM {} LPIPS {} MS SSIM {} ALEX_LPIPS {} TLP {}").format(
                    iteration, config['name'], l1_test, psnr_test, ssim_test,
                    lpips_test, ms_ssim_test, alex_lpips_test, tlp_test)
                if progress_bar is None:
                    print(msg)
                else:
                    progress_bar.set_description(msg)

                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - ssim', test_ssim, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - lpips', test_lpips, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - ms-ssim', test_ms_ssim, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - alex-lpips', test_alex_lpips, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - tlp', tlp_test, iteration)

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity.detach().cpu().numpy(), iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()

    return test_psnr, test_ssim, test_lpips, test_ms_ssim, test_alex_lpips, test_tlp
