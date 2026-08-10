"""
grtri.angles
============
Grover--Rudolph dyadic angle map and the two Gray-code ladder realisations of
the level-2 uniformly controlled `Ry` block.

Angle conventions (important)
-----------------------------
The manuscript distinguishes two angles per node `w` of the dyadic tree:

    theta_w   : the *theoretical* splitting angle, cos^2(theta_w) = p_{w0}/p_w,
                so theta_w in [0, 90] degrees;
    phi_w     : the *commanded hardware* rotation, phi_w = 2 * theta_w.

The campaign code in `gr-triangulum-characterisation` uses the name `theta*`
for what the manuscript calls `phi` (its `safe_acos` helper already multiplies
by two).  This module follows the **hardware convention** -- every angle
returned here is a commanded `Ry` angle in degrees, i.e. `phi` -- because that
is what is compared against the circuit figures and against Table 5 of the
manuscript.  Use `theta_from_phi` if you need the theoretical angles.

Reference
---------
Falco, Falco-Pomares & Matthies (2026), arXiv:2601.17930, Appendix A;
Moettoenen, Vartiainen, Bergholm & Salomaa, QIC 5, 467 (2005) for the
Gray-code ladder decomposition of uniformly controlled rotations.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

__all__ = [
    "GRAngles",
    "gr_angles",
    "theta_from_phi",
    "WALSH_HADAMARD",
    "LADDER_CONTROLS",
]


# The 4x4 Walsh--Hadamard matrix used to invert the uniformly controlled
# rotation into a Gray-code ladder of plain rotations and CNOTs.
WALSH_HADAMARD = np.array(
    [
        [1, 1, 1, 1],
        [1, -1, 1, -1],
        [1, 1, -1, -1],
        [1, -1, -1, 1],
    ],
    dtype=float,
) / 4.0

# Control qubit applied *after* each of the four sub-rotations of the level-2
# UCRy block acting on q2.  The two ladders are exactly equivalent as unitaries
# and differ only in the traversal order of the Gray code.
LADDER_CONTROLS: dict[str, tuple[int, int, int, int]] = {
    "A": (1, 0, 1, 0),   # Gray code 00 -> 10 -> 11 -> 01
    "B": (0, 1, 0, 1),   # Gray code 00 -> 01 -> 11 -> 10
}

# Ordering in which the four leaf angles are fed to the Walsh--Hadamard
# transform, keyed by ladder.  Keys refer to the (q0, q1) control pattern.
_WHT_ORDER: dict[str, tuple[str, str, str, str]] = {
    "A": ("00", "11", "10", "01"),
    "B": ("00", "11", "01", "10"),
}


def _split_angle(numerator: float, denominator: float) -> float:
    """
    Commanded hardware angle ``phi = 2 * arccos(sqrt(numerator/denominator))``
    in degrees.

    Returns 0.0 when the parent mass is numerically zero, which keeps the
    construction well defined on distributions with empty subtrees.
    """
    if denominator < 1e-15:
        return 0.0
    ratio = float(np.clip(numerator / denominator, 0.0, 1.0))
    return 2.0 * float(np.degrees(np.arccos(np.sqrt(ratio))))


def theta_from_phi(phi_deg: float) -> float:
    """Convert a commanded hardware angle back to the theoretical GR angle."""
    return 0.5 * phi_deg


@dataclass(frozen=True)
class GRAngles:
    """
    Complete angle set for a 3-qubit Grover--Rudolph preparation.

    All angles are commanded hardware `Ry` angles in degrees (`phi`).

    Attributes
    ----------
    phi_root
        Level-0 rotation on q0.
    phi_level1
        Level-1 rotations on q1, keyed by the value of q0 ("0", "1").
    phi_level2
        Level-2 leaf rotations on q2, keyed by the (q0, q1) bit pattern.
    ladder_angles
        The four Walsh--Hadamard-transformed sub-rotations actually executed on
        q2, keyed by ladder ("A", "B").
    """

    phi_root: float
    phi_level1: dict[str, float]
    phi_level2: dict[str, float]
    ladder_angles: dict[str, np.ndarray] = field(repr=False)

    # -- convenience views -------------------------------------------------

    @property
    def theta_root(self) -> float:
        return theta_from_phi(self.phi_root)

    def as_flat_dict(self) -> dict[str, float]:
        """Flat mapping suitable for CSV export."""
        out = {"phi_root": self.phi_root}
        for k, v in sorted(self.phi_level1.items()):
            out[f"phi_1|{k}"] = v
        for k, v in sorted(self.phi_level2.items()):
            out[f"phi_2|{k}"] = v
        for ladder, arr in sorted(self.ladder_angles.items()):
            for i, a in enumerate(arr):
                out[f"ladder_{ladder}_{i}"] = float(a)
        return out


def gr_angles(p: np.ndarray | list[float]) -> GRAngles:
    """
    Compute the full Grover--Rudolph angle set for an 8-entry probability
    vector, together with both Gray-code ladder realisations.

    Parameters
    ----------
    p
        Probability vector of length 8, indexed as ``k = 4*q0 + 2*q1 + q2``
        (q0 is the most significant bit).  Must be non-negative and sum to 1.

    Returns
    -------
    GRAngles

    Notes
    -----
    This reproduces `gr_angles` in `_spinqit_backend.py` of the campaign
    repository exactly; the only differences are naming and packaging.
    """
    p = np.asarray(p, dtype=float)
    if p.shape != (8,):
        raise ValueError(f"expected a length-8 vector, got shape {p.shape}")
    if np.any(p < -1e-12):
        raise ValueError("probability vector has negative entries")
    if not np.isclose(p.sum(), 1.0, atol=1e-9):
        raise ValueError(f"probability vector sums to {p.sum()!r}, not 1")

    # Partial masses of the dyadic tree.
    s_left = p[0] + p[1] + p[2] + p[3]
    s_right = 1.0 - s_left
    s_ll, s_lr = p[0] + p[1], p[2] + p[3]
    s_rl, s_rr = p[4] + p[5], p[6] + p[7]

    phi_root = _split_angle(s_left, 1.0)
    phi_level1 = {
        "0": _split_angle(s_ll, s_left),
        "1": _split_angle(s_rl, s_right),
    }
    phi_level2 = {
        "00": _split_angle(p[0], s_ll),
        "01": _split_angle(p[2], s_lr),
        "10": _split_angle(p[4], s_rl),
        "11": _split_angle(p[6], s_rr),
    }

    ladder_angles: dict[str, np.ndarray] = {}
    for ladder, order in _WHT_ORDER.items():
        alpha = np.array([phi_level2[k] for k in order]) / 2.0
        ladder_angles[ladder] = 2.0 * (WALSH_HADAMARD @ alpha)

    return GRAngles(
        phi_root=phi_root,
        phi_level1=phi_level1,
        phi_level2=phi_level2,
        ladder_angles=ladder_angles,
    )
