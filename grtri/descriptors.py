"""
grtri.descriptors
=================
Structural descriptors of a target distribution and of its Grover--Rudolph
compilation tree.

These are the covariates used in the manuscript to explain the ordering of
FULL-stage hardware fidelity across the benchmark suite.

WARNING -- two different "UCRy" descriptors are in circulation
--------------------------------------------------------------
The manuscript (Table 5 and Section "Predictors of FULL-stage fidelity")
defines the **UCRy range**

    Delta_UCRy = max_j phi_{2|j} - min_j phi_{2|j}

whereas the shipped analysis script
`experiments/campaign_v2/05_characterisation_analysis.py` computes a different
quantity under the name `max_ucry_dev`,

    max_ucry_dev = max_j | phi_{2|j} - 90 |.

They are not equal (for D1: 38.9 vs 19.5).  Both are provided here under
distinct names so that any downstream analysis states which one it used.  The
values printed in Table 5 of the manuscript correspond to `ucry_range`.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from .angles import gr_angles
from .metrics import contrast, shannon_entropy

__all__ = ["Descriptors", "descriptors"]


@dataclass(frozen=True)
class Descriptors:
    """Scalar structural descriptors of a 3-qubit target distribution."""

    shannon_entropy_bits: float
    contrast: float
    level1_deviation: float      # Delta_1, manuscript Table 5
    ucry_range: float            # Delta_UCRy, manuscript Table 5
    max_ucry_dev: float          # as computed by 05_characterisation_analysis.py

    def as_dict(self) -> dict[str, float]:
        return asdict(self)


def descriptors(p: np.ndarray | list[float]) -> Descriptors:
    """
    Compute all structural descriptors for a target probability vector.

    Angles are commanded hardware angles (`phi`) in degrees, matching the
    convention in which Table 5 of the manuscript is stated.
    """
    p = np.asarray(p, dtype=float)
    ang = gr_angles(p)

    level1 = np.array(list(ang.phi_level1.values()))
    level2 = np.array(list(ang.phi_level2.values()))

    return Descriptors(
        shannon_entropy_bits=shannon_entropy(p),
        contrast=contrast(p),
        level1_deviation=float(np.max(np.abs(level1 - 90.0))),
        ucry_range=float(level2.max() - level2.min()),
        max_ucry_dev=float(np.max(np.abs(level2 - 90.0))),
    )
