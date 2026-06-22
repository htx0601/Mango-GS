#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from PIL import Image
from piq import LPIPS


def _load_rgb(path: Path, device: torch.device) -> torch.Tensor:
    return TF.to_tensor(Image.open(path).convert("RGB")).unsqueeze(0).to(device)


def _resize_for_lpips(image_a: torch.Tensor, image_b: torch.Tensor, min_size: int = 64) -> tuple[torch.Tensor, torch.Tensor]:
    height, width = image_a.shape[-2:]
    if min(height, width) >= min_size:
        return image_a, image_b
    scale = min_size / max(1, min(height, width))
    size = (max(min_size, int(round(height * scale))), max(min_size, int(round(width * scale))))
    return (
        F.interpolate(image_a, size=size, mode="bilinear", align_corners=False),
        F.interpolate(image_b, size=size, mode="bilinear", align_corners=False),
    )


def _candidate_bases(model_path: Path, iteration: str) -> list[Path]:
    return [
        model_path / "test" / f"video_{iteration}",
        model_path / "test" / f"ours_{iteration}",
    ]


def compute_tpips_tlp(model_path: Path, iteration: str, device: str = "cuda") -> float | None:
    torch_device = torch.device(device if device == "cuda" and torch.cuda.is_available() else "cpu")
    metric = LPIPS().to(torch_device)
    vals: list[torch.Tensor] = []

    for base in _candidate_bases(model_path, iteration):
        renders_root = base / "renders"
        gts_root = base / "gt"
        if not renders_root.exists() or not gts_root.exists():
            continue

        render_dirs = [path for path in renders_root.rglob("*") if path.is_dir()]
        if any(renders_root.glob("*.png")):
            render_dirs.append(renders_root)

        for render_dir in sorted(set(render_dirs)):
            rel = render_dir.relative_to(renders_root)
            gt_dir = gts_root / rel
            if not gt_dir.exists():
                continue
            render_paths = sorted(path for path in render_dir.glob("*.png") if not path.name.endswith("_v1.png"))
            pairs = [(render_path, gt_dir / render_path.name) for render_path in render_paths]
            pairs = [(render_path, gt_path) for render_path, gt_path in pairs if gt_path.exists()]
            if len(pairs) < 2:
                continue

            pred_prev = _load_rgb(pairs[0][0], torch_device)
            gt_prev = _load_rgb(pairs[0][1], torch_device)
            for pred_path, gt_path in pairs[1:]:
                pred = _load_rgb(pred_path, torch_device)
                gt = _load_rgb(gt_path, torch_device)
                pred_cur, pred_old = _resize_for_lpips(pred, pred_prev)
                gt_cur, gt_old = _resize_for_lpips(gt, gt_prev)
                vals.append(torch.abs(metric(pred_cur, pred_old) - metric(gt_cur, gt_old)).detach().reshape(()).cpu())
                pred_prev, gt_prev = pred, gt

    if not vals:
        return None
    return float(torch.stack(vals).mean().item())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("model_path")
    parser.add_argument("iteration")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    value = compute_tpips_tlp(Path(args.model_path), str(args.iteration), args.device)
    print("nan" if value is None else f"{value:.8f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
