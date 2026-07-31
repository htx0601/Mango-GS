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

from errno import EEXIST
from os import makedirs, path
import os


def mkdir_p(folder_path):
    # Creates a directory. equivalent to using mkdir -p on the command line
    try:
        makedirs(folder_path)
    except OSError as exc:  # Python >2.5
        if exc.errno == EEXIST and path.isdir(folder_path):
            pass
        else:
            raise


def searchForMaxIteration(folder):
    if not os.path.exists(folder):
        return None
    saved_iters = [int(fname.split("_")[-1]) for fname in os.listdir(folder) if "_" in fname]
    return max(saved_iters) if saved_iters != [] else None


def resolveCheckpointIteration(model_path, requested_iteration):
    flat_point_cloud = os.path.join(model_path, "point_cloud.ply")
    flat_deform = os.path.join(model_path, "deform.pth")
    if os.path.isfile(flat_point_cloud) and os.path.isfile(flat_deform):
        return "release"

    if requested_iteration != -1:
        return requested_iteration

    point_iteration = searchForMaxIteration(os.path.join(model_path, "point_cloud"))
    deform_iteration = searchForMaxIteration(os.path.join(model_path, "deform"))
    if point_iteration is None or deform_iteration is None:
        raise FileNotFoundError(f"No complete checkpoint found in {model_path}")
    if point_iteration != deform_iteration:
        raise RuntimeError(
            f"Checkpoint mismatch in {model_path}: "
            f"point_cloud={point_iteration}, deform={deform_iteration}"
        )
    return point_iteration


def resolvePointCloudPath(model_path, checkpoint):
    flat_path = os.path.join(model_path, "point_cloud.ply")
    if os.path.isfile(flat_path):
        return flat_path
    return os.path.join(model_path, "point_cloud", f"iteration_{checkpoint}", "point_cloud.ply")
