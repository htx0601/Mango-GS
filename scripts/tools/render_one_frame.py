#!/usr/bin/env python3

from __future__ import annotations

import argparse
import os
import sys

import torch
import torchvision

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from arguments.__init__video import ModelParams, OptimizationParams, PipelineParams, apply_dataset_preset
from gaussian_renderer import GaussianModel, render_batch
from scene import DeformModel, Scene
from utils.general_utils import safe_state
from utils.system_utils import resolveCheckpointIteration


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_path", required=True)
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--iteration", type=int, default=-1)
    parser.add_argument("--out_path", required=True)
    parser.add_argument("--deform_type", default="mango_node")
    parser.add_argument("--resolution", type=int, default=2)
    parser.add_argument("--W", type=int, default=800)
    parser.add_argument("--H", type=int, default=800)
    parser.add_argument("--profile_config", default="")
    args = parser.parse_args()

    param_parser = argparse.ArgumentParser()
    lp = ModelParams(param_parser)
    op = OptimizationParams(param_parser)
    pp = PipelineParams(param_parser)
    param_parser.add_argument("--W", type=int, default=args.W)
    param_parser.add_argument("--H", type=int, default=args.H)
    param_parser.add_argument("--profile_config", default=args.profile_config)
    argv = [
        "--source_path",
        args.source_path,
        "--model_path",
        args.model_path,
        "--deform_type",
        args.deform_type,
        "--eval",
        "--resolution",
        str(args.resolution),
        "--white_background",
    ]
    if args.profile_config:
        argv.extend(["--profile_config", args.profile_config])
    train_args = param_parser.parse_args(argv)
    train_args = apply_dataset_preset(train_args)
    if not train_args.model_path.endswith(train_args.deform_type):
        train_args.model_path = os.path.join(
            os.path.dirname(os.path.normpath(train_args.model_path)),
            os.path.basename(os.path.normpath(train_args.model_path)) + f"_{train_args.deform_type}",
        )
    if "/n3v/" in args.source_path.replace("\\", "/"):
        train_args.n3v_camera_id_mode = "position"
        train_args.load2gpu_on_the_fly = True
        train_args.n3v_num_images = 1

    safe_state(True)
    dataset = lp.extract(train_args)
    opt = op.extract(train_args)
    pipe = pp.extract(train_args)
    del opt
    args.iteration = resolveCheckpointIteration(dataset.model_path, args.iteration)

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
    if not deform.load_weights(dataset.model_path, iteration=args.iteration):
        raise RuntimeError(f"Could not load deform weights: {dataset.model_path}")

    gs_fea_dim = deform.deform.node_num if dataset.skinning and "node" in deform.name else dataset.hyper_dim
    gaussians = GaussianModel(dataset.sh_degree, fea_dim=gs_fea_dim, with_motion_mask=dataset.gs_with_motion_mask)
    scene = Scene(dataset, gaussians, load_iteration=args.iteration, shuffle=False)

    views = scene.getTestCameras() or scene.getTrainCameras()
    if not views:
        raise RuntimeError("No cameras available to render")

    group = list(views[: deform.T])
    if len(group) < deform.T:
        group.extend([group[-1]] * (deform.T - len(group)))
    if dataset.load2gpu_on_the_fly:
        for view in group:
            view.load2device("cuda")

    fids = []
    for view in group:
        fid = view.fid if torch.is_tensor(view.fid) else torch.tensor(view.fid, device="cuda")
        fid = fid.to("cuda")
        if fid.dim() == 0:
            fid = fid.unsqueeze(0)
        fids.append(fid.reshape(1))
    time_input = torch.stack(fids, dim=1).float().cuda()

    dvals = deform.step(
        gaussians.get_xyz.detach(),
        time_input,
        feature=gaussians.feature,
        motion_mask=gaussians.motion_mask,
        camera_center=[view.camera_center for view in group],
        knn_feature=gaussians.get_knn_feature.detach(),
    )
    background = torch.tensor([1, 1, 1] if dataset.white_background else [0, 0, 0], dtype=torch.float32, device="cuda")
    pkg = render_batch(
        group,
        gaussians,
        pipe,
        background,
        dvals["d_xyz"],
        dvals["d_rotation"],
        dvals["d_scaling"],
        dvals.get("d_opacity", None),
        dvals.get("d_color", None),
        list(range(len(group))),
        d_rot_as_res=deform.d_rot_as_res,
    )

    os.makedirs(os.path.dirname(args.out_path), exist_ok=True)
    torchvision.utils.save_image(torch.clamp(pkg["render"][0], 0, 1), args.out_path)
    print(args.out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
