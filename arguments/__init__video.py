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

from argparse import ArgumentParser, Namespace
import sys
import os
import json
from pathlib import Path
from typing import Dict, List, Tuple


class GroupParams:
    pass


class ParamGroup:
    def __init__(self, parser: ArgumentParser, name: str, fill_none=False):
        group = parser.add_argument_group(name)
        for key, value in vars(self).items():
            shorthand = False
            if key.startswith("_"):
                shorthand = True
                key = key[1:]
            t = type(value)
            value = value if not fill_none else None
            # if shorthand:
            #     if t == bool:
            #         group.add_argument("--" + key, ("-" + key[0:1]), default=value, action="store_true")
            #     else:
            #         group.add_argument("--" + key, ("-" + key[0:1]), default=value, type=t)
            # else:
            if t == bool:
                group.add_argument("--" + key, default=value, action="store_true")
            else:
                group.add_argument("--" + key, default=value, type=t)

    def extract(self, args):
        group = GroupParams()
        for arg in vars(args).items():
            if arg[0] in vars(self) or ("_" + arg[0]) in vars(self):
                setattr(group, arg[0], arg[1])
        return group


class ModelParams(ParamGroup):
    def __init__(self, parser, sentinel=False):
        self.sh_degree = 3
        self.K = 3
        self._source_path = ""
        self._model_path = ""
        self._images = "images"
        self._resolution = -1
        self._white_background = False
        self.data_device = "cuda"
        self.eval = False
        self.load2gpu_on_the_fly = False
        self.n3v_protocol = "full_heldout"
        self.n3v_num_images = 0
        self.n3v_intrinsics_mode = "scaled"
        self.n3v_camera_id_mode = "position"
        self.is_blender = False
        self.deform_type = 'mango_node'
        self.T = 4
        self.skinning = False
        self.hyper_dim = 8
        self.node_num = 2048
        self.enable_learned_metric = False
        self.pred_opacity = False
        self.pred_color = False
        self.use_hash = False
        self.hash_time = False
        self.d_rot_as_rotmat = False
        self.d_rot_as_res = True
        self.local_frame = False
        self.progressive_brand_time = False
        self.gs_with_motion_mask = False
        self.init_isotropic_gs_with_all_colmap_pcl = False
        self.as_gs_force_with_motion_mask = False  # Only for scenes with both static and dynamic parts and without alpha mask
        self.max_d_scale = -1.
        self.is_scene_static = False
        super().__init__(parser, "Loading Parameters", sentinel)

    def extract(self, args):
        g = super().extract(args)
        g.source_path = os.path.abspath(g.source_path)
        if not g.model_path.endswith(g.deform_type):
            g.model_path = os.path.join(os.path.dirname(os.path.normpath(g.model_path)), os.path.basename(os.path.normpath(g.model_path)) + f'_{g.deform_type}')
        return g


class PipelineParams(ParamGroup):
    def __init__(self, parser):
        self.convert_SHs_python = False
        self.compute_cov3D_python = False
        self.debug = False
        super().__init__(parser, "Pipeline Parameters")


class OptimizationParams(ParamGroup):
    def __init__(self, parser):
        self.iterations = 80_000
        self.warm_up = 1000
        self.dynamic_color_warm_up = 20_000
        self.position_lr_init = 0.00016
        self.position_lr_final = 0.0000016
        self.position_lr_delay_mult = 0.01
        self.position_lr_max_steps = 30_000
        self.deform_lr_max_steps = 45_000
        self.feature_lr = 0.0025
        self.opacity_lr = 0.05
        self.scaling_lr = 0.001
        self.rotation_lr = 0.001
        self.percent_dense = 0.01
        self.lambda_dssim = 0.2
        self.lambda_temp = 0.1
        self.densification_interval = 100
        self.opacity_reset_interval = 3000
        self.densify_from_iter = 3000
        self.densify_until_iter = 50_000
        self.densify_grad_threshold = 0.00018
        self.densify_grad_reduce_quantile = -1.0
        self.densify_node_grad_threshold = 0.0001
        self.oneupSHdegree_step = 1000
        self.random_bg_color = False

        self.deform_lr_scale = 1.
        self.deform_downsamp_strategy = 'samp_hyper'
        self.deform_downsamp_with_dynamic_mask = False
        self.node_enable_densify_prune = False
        self.node_densification_interval = 2500
        self.node_densify_from_iter = 10000
        self.node_densify_until_iter = 25_000
        self.node_force_densify_prune_step = 10_000
        self.node_max_num_ratio_during_init = 16
        self.use_tcnn = False

        self.random_init_deform_gs = False
        self.node_warm_up = 6000
        self.iterations_node_sampling = 10000
        self.iterations_node_rendering = 12000

        self.progressive_train = False
        self.progressive_train_node = False
        self.progressive_stage_ratio = .2  # The ratio of the number of images added per stage
        self.progressive_stage_steps = 3000  # The training steps of each stage

        self.lambda_optical_landmarks = [1e-1, 1e-1,   1e-3,        0]
        self.lambda_optical_steps =     [0,    15_000, 25_000, 25_001]

        self.lambda_motion_mask_landmarks = [5e-1,      1e-2,      0]
        self.lambda_motion_mask_steps =     [0,       10_000, 10_001]
        self.no_motion_mask_loss = False  # Camera pose may be inaccurate and should model the whole scene motion

        self.gt_alpha_mask_as_scene_mask = False
        self.gt_alpha_mask_as_dynamic_mask = False
        self.no_arap_loss = True  # For large scenes arap is too slow

        self.with_temporal_smooth_loss = False
        self.with_motion_diff_loss = False
        self.lambda_motion_diff = 0.2
        self.lambda_motion_under = 0.12
        self.lambda_motion_dir = 0.012
        self.motion_tau_quantile = 0.75
        self.motion_blur_ks = 3
        self.with_dynamic_edge_loss = False
        self.lambda_dynamic_edge = 0.0
        self.lambda_dynamic_lap = 0.0
        self.dynamic_edge_start_iter = 12000
        self.dynamic_edge_tau_quantile = 0.75
        self.dynamic_edge_mask_blur_ks = 3
        self.lambda_depth = 0.0
        self.enable_topk = True
        self.disable_topk_loss = False
        self.loss_topk_ratio = 0.6

        self.mask_probability = 0.15
        self.pos_net_prob = 0.7
        self.pos_max_interval = 3
        self.neg_max_interval = 3
        self.cross_view_group_prob = 0.0
        self.cross_view_anchor_prob = 0.0
        self.lambda_cross_view_anchor = 0.0
        self.cross_view_anchor_num = 1
        self.cross_view_time_tolerance = 1e-4

        super().__init__(parser, "Optimization Parameters")


def get_combined_args(parser: ArgumentParser):
    cmdlne_string = sys.argv[1:]
    cfgfile_string = "Namespace()"
    args_cmdline = parser.parse_args(cmdlne_string)

    if getattr(args_cmdline, "deform_type", None) is None:
        args_cmdline.deform_type = "mango_node"
    if not args_cmdline.model_path.endswith(args_cmdline.deform_type):
        args_cmdline.model_path = os.path.join(os.path.dirname(os.path.normpath(args_cmdline.model_path)), os.path.basename(os.path.normpath(args_cmdline.model_path)) + f'_{args_cmdline.deform_type}')

    try:
        cfgfilepath = os.path.join(args_cmdline.model_path, "cfg_args")
        print("Looking for config file in", cfgfilepath)
        with open(cfgfilepath) as cfg_file:
            print("Config file found: {}".format(cfgfilepath))
            cfgfile_string = cfg_file.read()
    except (FileNotFoundError, TypeError):
        print("Config file not found; using command-line and profile defaults.")
    args_cfgfile = eval(cfgfile_string)

    merged_dict = vars(args_cfgfile).copy()
    for k, v in vars(args_cmdline).items():
        if v != None:
            merged_dict[k] = v
    return Namespace(**merged_dict)


def _arg_was_set(name: str) -> bool:
    flags = {f"--{name}", f"--{name.replace('_', '-')}"}
    return any(arg == flag or arg.startswith(flag + "=") for arg in sys.argv[1:] for flag in flags)


def _profiles_root(args) -> Path:
    explicit = getattr(args, "profile_config", "") or os.environ.get("MANGO_GS_PROFILE_CONFIG", "")
    if explicit:
        path = Path(explicit).expanduser().resolve()
        return path.parent / "profiles" if path.is_file() else path
    repo_root = Path(__file__).resolve().parents[1]
    return repo_root / "configs" / "profiles"


def _load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _load_release_profiles(args, dataset: str, scene: str) -> Tuple[List[str], Dict]:
    root = _profiles_root(args)
    params = {}
    chain = []

    global_profile = _load_json(root / "global.json")
    if global_profile:
        params.update(global_profile.get("params", {}))
        chain.append("global")

    dataset_profile = _load_json(root / "datasets" / f"{dataset}.json")
    if dataset_profile:
        params.update(dataset_profile.get("params", {}))
        chain.append(dataset)

    scene_profile = _load_json(root / "scenes" / dataset / f"{scene}.json")
    if scene_profile:
        params.update(scene_profile.get("params", {}))
        chain.append(f"{dataset}/{scene}")

    return chain, params


def _detect_dataset_and_scene(source_path: str) -> Tuple[str, str]:
    normalized = os.path.abspath(source_path or "").replace("\\", "/")
    lowered = normalized.lower()
    parts = [part for part in normalized.split("/") if part]
    scene = parts[-1] if parts else ""
    n3v_scenes = {
        "coffee_martini",
        "flame_salmon_1",
        "cook_spinach",
        "cut_roasted_beef",
        "flame_steak",
        "sear_steak",
    }
    hypernerf_scenes = {
        "broom2",
        "vrig-3dprinter",
        "vrig-chicken",
        "vrig-peel-banana",
    }
    if scene in n3v_scenes:
        return "n3v", scene
    if scene in hypernerf_scenes:
        return "hypernerf", scene
    if "/n3v/" in lowered or lowered.endswith("/n3v"):
        return "n3v", scene
    if "/hypernerf/" in lowered or lowered.endswith("/hypernerf") or "/vrig/" in lowered:
        return "hypernerf", scene
    return "default", scene


def _profile_dataset_and_scene(args) -> Tuple[str, str]:
    dataset = getattr(args, "profile_dataset", "") or ""
    scene = getattr(args, "profile_scene", "") or ""
    detected_dataset, detected_scene = _detect_dataset_and_scene(getattr(args, "source_path", "") or "")
    return dataset or detected_dataset, scene or detected_scene


def _set_profile_value(args, name: str, value):
    if _arg_was_set(name):
        return False
    setattr(args, name, value)
    return True


def apply_dataset_preset(args):
    if getattr(args, "no_profile", False):
        args.applied_profile = "disabled"
        return args

    dataset, scene = _profile_dataset_and_scene(args)
    chain, params = _load_release_profiles(args, dataset, scene)

    applied = []
    for name, value in params.items():
        if _set_profile_value(args, name, value):
            applied.append(name)

    args.profile_dataset = dataset
    args.profile_scene = scene
    args.applied_profile = " > ".join(chain)
    args.profile_applied_params = ",".join(sorted(applied))
    if chain:
        print(f"Applied Mango-GS profile: {args.applied_profile}")

    return args
