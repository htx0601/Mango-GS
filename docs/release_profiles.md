# Release Profiles

The public profile hierarchy lives under `configs/profiles/`.

Profiles are applied in this order:

1. `configs/profiles/global.json`
2. `configs/profiles/datasets/<dataset>.json`
3. `configs/profiles/scenes/<dataset>/<scene>.json`
4. Explicit command-line arguments

The release defaults keep the paper-compatible path enabled: top-k is on, temporal masking is on, and extra diagnostic losses are off. The scene overrides only change structural parameters such as `T`, `K`, and `node_num`.

Profiles are loaded by `train.py`, `render.py`, and the public shell scripts. The dataset and scene are inferred from `source_path`; explicit command-line arguments always take priority.

## N3V

| Scene | T | K | node_num |
|---|---:|---:|---:|
| coffee_martini | 4 | 3 | 2048 |
| flame_salmon_1 | 4 | 3 | 2048 |
| cook_spinach | 4 | 5 | 4096 |
| cut_roasted_beef | 4 | 5 | 4096 |
| flame_steak | 4 | 5 | 4096 |
| sear_steak | 4 | 5 | 4096 |

Dataset defaults: `resolution=2`, `W=800`, `H=800`, scaled intrinsics, positional camera ids, `densify_until_iter=18000`, `node_densify_until_iter=18000`, `densify_grad_threshold=0.0001`, and `opacity_reset_interval=300000`.

## HyperNeRF

| Scene | T | K | node_num |
|---|---:|---:|---:|
| broom2 | 6 | 3 | 2048 |
| vrig-3dprinter | 6 | 3 | 2048 |
| vrig-chicken | 6 | 3 | 2048 |
| vrig-peel-banana | 8 | 3 | 4096 |

Dataset defaults: `resolution=2`, `W=800`, `H=800`, `densify_until_iter=10000`, `node_densify_until_iter=10000`, `densify_grad_threshold=0.00012`, and `opacity_reset_interval=3000`.
