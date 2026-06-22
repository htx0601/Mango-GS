"""Backward-compatible imports for Mango-GS deformation utilities.

The paper-specific temporal deformation, learned metric, and control-node
implementations now live in :mod:`mango.time_utils`. This module preserves the
legacy import path used by the original research scripts and older checkpoints.
"""

from mango.time_utils import *  # noqa: F401,F403
