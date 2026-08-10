"""
grtri -- reproducibility package for the Grover--Rudolph Triangulum campaigns
============================================================================

Self-contained reference implementation accompanying the manuscript

    "Hardware characterisation of exact Grover--Rudolph state preparation on a
     3-qubit NMR quantum computer"

It depends only on NumPy (plus SciPy for the statistical scripts) and requires
neither hardware nor the SpinQ vendor SDK, so that every claim in the paper
that can be checked without the device can in fact be checked.

Authoritative campaign code lives at
https://github.com/afalco/gr-triangulum-characterisation
"""

from .angles import GRAngles, gr_angles, theta_from_phi
from .backends import Backend, IdealBackend, NoisyBackend, SpinQitBackend
from .descriptors import Descriptors, descriptors
from .distributions import (
    DIST_IDS,
    DIST_TYPES,
    build_distributions,
    checksum,
    load_canonical,
    verify_against_canonical,
)
from .metrics import (
    contrast,
    fidelity,
    l2_distance,
    marginals,
    mean_of_metric,
    metric_of_mean,
    shannon_entropy,
    tv_distance,
)
from .simulator import LADDERS, STAGES, gate_counts, simulate, statevector

__version__ = "1.0.0"

__all__ = [
    "GRAngles", "gr_angles", "theta_from_phi",
    "Backend", "IdealBackend", "NoisyBackend", "SpinQitBackend",
    "Descriptors", "descriptors",
    "DIST_IDS", "DIST_TYPES", "build_distributions", "checksum",
    "load_canonical", "verify_against_canonical",
    "tv_distance", "l2_distance", "fidelity", "marginals",
    "shannon_entropy", "contrast", "mean_of_metric", "metric_of_mean",
    "STAGES", "LADDERS", "simulate", "statevector", "gate_counts",
    "__version__",
]
