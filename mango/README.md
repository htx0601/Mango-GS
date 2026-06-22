# Mango-GS Modules

This package exposes the paper-specific implementation:

- `time_utils.py`: public Mango-GS temporal deformation and control-node utilities.
- `MangoNodeWarp`: decoupled control nodes with learned affinity.
- `MangoDeformNetwork`: multi-frame temporal deformation network with temporal attention.

The original 3D Gaussian Splatting renderer, rasterizer, scene loaders, and
Gaussian parameter storage remain in their original modules.
