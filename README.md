# Mango-GS

Official release code for **Mango-GS: Enhancing Spatio-Temporal Consistency in Dynamic Scenes Reconstruction using Multi-Frame Node-Guided 4D Gaussian Splatting**.

- Paper: [Mango-GS: Enhancing Spatio-Temporal Consistency in Dynamic Scenes Reconstruction using Multi-Frame Node-Guided 4D Gaussian Splatting](https://arxiv.org/abs/2603.11543)
- Project page: [https://htx0601.github.io/Mango-GS/](https://htx0601.github.io/Mango-GS/)
- Code: [https://github.com/htx0601/Mango-GS](https://github.com/htx0601/Mango-GS)

This repository contains the training, rendering, metric, profile, and CUDA extension code used by the public release. Pretrained weights and datasets are distributed separately.

## Repository Layout

```text
.
|-- train.py                         # Training entry point
|-- trainer.py                       # Main training loop
|-- training_report.py               # Training-time metric logging
|-- render.py                        # Test rendering and video export
|-- metrics.py                       # PSNR / SSIM / LPIPS evaluation
|-- configs/profiles/                # Global, dataset, and scene profiles
|-- docs/index.html                  # GitHub Pages project page
|-- docs/release_profiles.md         # Profile table and parameter summary
|-- scripts/
|   |-- train_scene.sh               # Train one scene on one GPU
|   |-- queue_train.sh               # TSV-driven multi-GPU training queue
|   |-- render_scene.sh              # Render one scene
|   |-- render_one_frame.sh          # Render one preview frame
|   |-- eval_metrics.sh              # PSNR / SSIM / LPIPS / TPIPS
|   `-- tools/                       # Helper scripts used by public commands
|-- scene/, gaussian_renderer/, mango/, utils/
|-- submodules/
|   |-- diff-gaussian-rasterization/
|   `-- simple-knn/
`-- weights/                         # Place downloaded pretrained weights here
```

## Installation

Create a CUDA-enabled Python environment. The release has been tested with Python 3.8 and PyTorch `2.4.1+cu121`.

```bash
conda create -n mango-gs python=3.8 -y
conda activate mango-gs
```

Install PyTorch for your CUDA version. For CUDA 12.1:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

Install Python dependencies and CUDA extensions:

```bash
pip install -r requirements.txt
pip install -e submodules/diff-gaussian-rasterization
pip install -e submodules/simple-knn
```

If PyTorch3D cannot be installed from `requirements.txt` on your platform, install a wheel or source build matching your PyTorch/CUDA version, then rerun the extension installation commands.

All shell scripts use `python` from the active environment. In non-interactive shells, pin the interpreter explicitly:

```bash
PYTHON=/path/to/conda/env/bin/python scripts/train_scene.sh ...
```

## Data Preparation

The scripts do not require a fixed dataset root. Pass the scene directory as `source_path`.

N3V scenes should contain `poses_bounds.npy` and per-camera frame folders:

```text
datasets/n3v/cook_spinach/
  poses_bounds.npy
  cam00/images/
  cam01/images/
  ...
```

HyperNeRF/Nerfies scenes should follow the standard processed layout:

```text
datasets/hypernerf/vrig/vrig-peel-banana/
  dataset.json
  scene.json
  camera/
  rgb/
  points3d.ply
```

## Release Profiles

Training and rendering parameters are selected automatically from `source_path`.

Profiles are applied in this order:

```text
configs/profiles/global.json
  -> configs/profiles/datasets/<dataset>.json
  -> configs/profiles/scenes/<dataset>/<scene>.json
  -> explicit command-line arguments
```

The released profiles cover the N3V and HyperNeRF scenes listed in `docs/release_profiles.md`. Use `--profile_dataset` and `--profile_scene` only when the scene name cannot be inferred from the path.

## Train N3V

Example: train `cook_spinach` on GPU 0.

```bash
scripts/train_scene.sh n3v cook_spinach \
  /data/datasets/n3v/cook_spinach \
  outputs \
  0
```

The output directory is:

```text
outputs/n3v_cook_spinach_mango_node/
```

To train a shorter debugging run:

```bash
scripts/train_scene.sh n3v cook_spinach \
  /data/datasets/n3v/cook_spinach \
  outputs \
  0 \
  --iterations 12000
```

## Train HyperNeRF

Example: train `vrig-peel-banana` on GPU 1.

```bash
scripts/train_scene.sh hypernerf vrig-peel-banana \
  /data/datasets/hypernerf/vrig/vrig-peel-banana \
  outputs \
  1
```

The output directory is:

```text
outputs/hypernerf_vrig-peel-banana_mango_node/
```

## Multi-GPU Queue

Create a TSV file:

```text
dataset	scene	source_path	model_root	gpu_or_auto	extra_args
n3v	cook_spinach	/data/datasets/n3v/cook_spinach	outputs	auto
hypernerf	vrig-peel-banana	/data/datasets/hypernerf/vrig/vrig-peel-banana	outputs	auto	--iterations 12000
```

Run the queue on GPUs `0,1,2,3` with at most four active jobs:

```bash
scripts/queue_train.sh jobs.tsv 0,1,2,3 4
```

Logs are written under `queue_logs/`.

## Render And Validate N3V

Render test images and a video from a trained or downloaded checkpoint:

```bash
scripts/render_scene.sh n3v cook_spinach \
  /data/datasets/n3v/cook_spinach \
  outputs/n3v_cook_spinach \
  0 \
  20
```

The last argument is video FPS. Rendered outputs are saved under:

```text
outputs/n3v_cook_spinach_mango_node/test/video_<checkpoint>/
```

Compute image and temporal metrics:

```bash
scripts/eval_metrics.sh outputs/n3v_cook_spinach 0
```

Render one preview frame:

```bash
scripts/render_one_frame.sh n3v cook_spinach \
  /data/datasets/n3v/cook_spinach \
  outputs/n3v_cook_spinach \
  0 \
  previews/cook_spinach.png
```

## Render And Validate HyperNeRF

```bash
scripts/render_scene.sh hypernerf vrig-peel-banana \
  /data/datasets/hypernerf/vrig/vrig-peel-banana \
  outputs/hypernerf_vrig-peel-banana \
  0 \
  10
```

```bash
scripts/eval_metrics.sh outputs/hypernerf_vrig-peel-banana 0
```

```bash
scripts/render_one_frame.sh hypernerf vrig-peel-banana \
  /data/datasets/hypernerf/vrig/vrig-peel-banana \
  outputs/hypernerf_vrig-peel-banana \
  0 \
  previews/vrig-peel-banana.png
```

## Pretrained Weights

Pretrained weights are hosted on
[Hugging Face](https://huggingface.co/htx0601/Mango-GS). Download them directly
into `weights/`:

```bash
hf download htx0601/Mango-GS --local-dir weights
```

Expected layout:

```text
weights/
  n3v/
    cook_spinach_mango_node/
      cfg_args
      point_cloud.ply
      deform.pth
  hypernerf/
    vrig-peel-banana_mango_node/
      cfg_args
      point_cloud.ply
      deform.pth
```

Commands may use the base path, for example `weights/n3v/cook_spinach`; the code resolves it to `weights/n3v/cook_spinach_mango_node`.

## Scene Profiles

N3V:

| Scene | T | K | node_num |
|---|---:|---:|---:|
| coffee_martini | 4 | 3 | 2048 |
| flame_salmon_1 | 4 | 3 | 2048 |
| cook_spinach | 4 | 5 | 4096 |
| cut_roasted_beef | 4 | 5 | 4096 |
| flame_steak | 4 | 5 | 4096 |
| sear_steak | 4 | 5 | 4096 |

HyperNeRF:

| Scene | T | K | node_num |
|---|---:|---:|---:|
| broom2 | 6 | 3 | 2048 |
| vrig-3dprinter | 6 | 3 | 2048 |
| vrig-chicken | 6 | 3 | 2048 |
| vrig-peel-banana | 8 | 3 | 4096 |

## Citation

If you use this code or the released models, please cite:

```bibtex
@inproceedings{huang2026mangogs,
  title     = {Mango-GS: Enhancing Spatio-Temporal Consistency in Dynamic Scenes Reconstruction using Multi-Frame Node-Guided 4D Gaussian Splatting},
  author    = {Huang, Tingxuan and Zhu, Haowei and Yong, Jun-hai and Pan, Hao and Wang, Bin},
  booktitle = {International Conference on Learning Representations},
  year      = {2026}
}
```

## License

This code is released under the MIT License. See `LICENSE`.
