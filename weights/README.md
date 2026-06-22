# Mango-GS Weights

Download the released weights and extract them into this folder.

- Google Drive: TBD
- Baidu Netdisk: TBD

Expected layout:

```text
weights/
  n3v/
    cook_spinach_mango_node/
      cfg_args
      point_cloud/iteration_<iter>/point_cloud.ply
      deform/iteration_<iter>/...
  hypernerf/
    vrig-peel-banana_mango_node/
      cfg_args
      point_cloud/iteration_<iter>/point_cloud.ply
      deform/iteration_<iter>/...
```

Mango-GS appends the deform suffix to model folders. Commands can be called with a base path such as `weights/n3v/cook_spinach`; it resolves to `weights/n3v/cook_spinach_mango_node`.
