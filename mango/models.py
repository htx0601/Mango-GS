"""Mango-GS model components.

The public class names use the paper terminology while retaining the original
state-dict layout for compatibility with existing experiments.
"""

from .time_utils import (
    MultiFrameControlNodeWarp,
    MultiFrameDeformNetwork,
)


class MangoNodeWarp(MultiFrameControlNodeWarp):
    """Decoupled multi-frame control-node warp used by Mango-GS."""


class MangoDeformNetwork(MultiFrameDeformNetwork):
    """Temporal Transformer deformation network used by Mango-GS nodes."""


__all__ = [
    "MangoDeformNetwork",
    "MangoNodeWarp",
    "MultiFrameControlNodeWarp",
    "MultiFrameDeformNetwork",
]
