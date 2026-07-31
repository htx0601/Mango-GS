# Mango-GS Weights

Download the released files into this folder.

- Hugging Face: https://huggingface.co/htx0601/Mango-GS

The repository contains the ten inference checkpoints selected for the public
release. Model parameters are recorded in `manifest.json`; the same hierarchy
is implemented by `configs/profiles/`.

Expected layout:

```text
weights/
  manifest.json
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

Commands use the base path without the deform suffix. For example,
`weights/n3v/cook_spinach` resolves to
`weights/n3v/cook_spinach_mango_node`.
