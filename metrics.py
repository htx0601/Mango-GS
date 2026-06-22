from __future__ import annotations

import json
import os
from argparse import ArgumentParser
from pathlib import Path

import torch
import torchvision.transforms.functional as tf
from PIL import Image
from tqdm import tqdm

from lpipsPyTorch import lpips
from utils.image_utils import psnr
from utils.loss_utils import ssim


def read_images(renders_dir: Path, gt_dir: Path):
    renders = []
    gts = []
    image_names = []
    for fname in sorted(os.listdir(renders_dir)):
        if not fname.lower().endswith((".png", ".jpg", ".jpeg")):
            continue
        render_path = renders_dir / fname
        gt_path = gt_dir / fname
        if not gt_path.exists():
            raise FileNotFoundError(f"Missing GT image for render {render_path}: {gt_path}")
        render = Image.open(render_path)
        gt = Image.open(gt_path)
        renders.append(tf.to_tensor(render).unsqueeze(0)[:, :3, :, :].cuda())
        gts.append(tf.to_tensor(gt).unsqueeze(0)[:, :3, :, :].cuda())
        image_names.append(fname)
    if not renders:
        raise RuntimeError(f"No render images found in {renders_dir}")
    return renders, gts, image_names


def _method_dirs(test_dir: Path):
    if not test_dir.exists():
        raise FileNotFoundError(f"Missing test directory: {test_dir}")
    methods = []
    for method_dir in sorted(path for path in test_dir.iterdir() if path.is_dir()):
        if method_dir.name.startswith(("ours_", "video_")):
            methods.append(method_dir)
    if not methods:
        raise RuntimeError(f"No supported render methods found in {test_dir}; expected ours_* or video_*")
    return methods


def _image_group_dirs(method_dir: Path):
    renders_root = method_dir / "renders"
    gt_root = method_dir / "gt"
    if not renders_root.exists() or not gt_root.exists():
        raise FileNotFoundError(f"Missing renders/gt folders under {method_dir}")

    groups = []
    if any(path.is_file() and path.suffix.lower() in {".png", ".jpg", ".jpeg"} for path in renders_root.iterdir()):
        groups.append((renders_root, gt_root, "."))

    for render_dir in sorted(path for path in renders_root.rglob("*") if path.is_dir()):
        if any(path.is_file() and path.suffix.lower() in {".png", ".jpg", ".jpeg"} for path in render_dir.iterdir()):
            rel = render_dir.relative_to(renders_root)
            groups.append((render_dir, gt_root / rel, str(rel)))

    if not groups:
        raise RuntimeError(f"No render image groups found under {renders_root}")
    return groups


def evaluate_scene(scene_dir: Path):
    print("Scene:", scene_dir)
    test_dir = scene_dir / "test"
    full_dict = {}
    per_view_dict = {}

    for method_dir in _method_dirs(test_dir):
        method = method_dir.name
        print("Method:", method)
        all_ssims = []
        all_psnrs = []
        all_lpipss = []

        for renders_dir, gt_dir, group_name in _image_group_dirs(method_dir):
            renders, gts, image_names = read_images(renders_dir, gt_dir)
            group_ssims = []
            group_psnrs = []
            group_lpipss = []

            for idx in tqdm(range(len(renders)), desc=f"Metric evaluation {method}/{group_name}"):
                group_ssims.append(ssim(renders[idx], gts[idx]))
                group_psnrs.append(psnr(renders[idx], gts[idx]))
                group_lpipss.append(lpips(renders[idx], gts[idx], net_type="vgg"))

            group_key = method if group_name == "." else f"{method}/{group_name}"
            full_dict[group_key] = {
                "SSIM": torch.tensor(group_ssims).mean().item(),
                "PSNR": torch.tensor(group_psnrs).mean().item(),
                "LPIPS": torch.tensor(group_lpipss).mean().item(),
            }
            per_view_dict[group_key] = {
                "SSIM": {name: value for value, name in zip(torch.tensor(group_ssims).tolist(), image_names)},
                "PSNR": {name: value for value, name in zip(torch.tensor(group_psnrs).tolist(), image_names)},
                "LPIPS": {name: value for value, name in zip(torch.tensor(group_lpipss).tolist(), image_names)},
            }
            all_ssims.extend(group_ssims)
            all_psnrs.extend(group_psnrs)
            all_lpipss.extend(group_lpipss)

        if all_psnrs:
            print("  SSIM : {:>12.7f}".format(torch.tensor(all_ssims).mean(), ".5"))
            print("  PSNR : {:>12.7f}".format(torch.tensor(all_psnrs).mean(), ".5"))
            print("  LPIPS: {:>12.7f}".format(torch.tensor(all_lpipss).mean(), ".5"))
            print("")

    with open(scene_dir / "results.json", "w") as fp:
        json.dump(full_dict, fp, indent=True)
    with open(scene_dir / "per_view.json", "w") as fp:
        json.dump(per_view_dict, fp, indent=True)


def evaluate(model_paths) -> int:
    failures = []
    for scene_dir in model_paths:
        try:
            evaluate_scene(Path(scene_dir))
        except Exception as exc:
            failures.append((scene_dir, exc))
            print(f"Unable to compute metrics for model {scene_dir}: {exc}")

    if failures and len(failures) == len(model_paths):
        return 1
    return 0


if __name__ == "__main__":
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)

    parser = ArgumentParser(description="Evaluate rendered Mango-GS images.")
    parser.add_argument("--model_paths", "-m", required=True, nargs="+", type=str, default=[])
    args = parser.parse_args()
    raise SystemExit(evaluate(args.model_paths))
