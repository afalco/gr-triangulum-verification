"""
grtri.distributions
===================
The canonical seven-distribution benchmark suite D0--D6.

The vectors are reproduced bit-for-bit from
`experiments/campaign_v2/00_generate_distributions.py` of the campaign
repository, including the two Dirichlet draws with their fixed seeds, and are
checked against the SHA-256 digests stored alongside the campaign data.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

__all__ = [
    "DIST_IDS",
    "DIST_TYPES",
    "build_distributions",
    "checksum",
    "load_canonical",
    "verify_against_canonical",
]

DIST_IDS = ("D0", "D1", "D2", "D3", "D4", "D5", "D6")

DIST_TYPES = {
    "D0": "Uniform",
    "D1": "Symmetric",
    "D2": "Monotone decreasing",
    "D3": "Bimodal",
    "D4": "Unimodal / bell",
    "D5": "Random sparse",
    "D6": "Random generic",
}

_DATA = Path(__file__).resolve().parents[1] / "data"


def _dirichlet(alpha: float, seed: int, n: int = 8,
               clip_min: float = 1e-4) -> np.ndarray:
    """Dirichlet draw with fixed seed, clipped away from zero and renormalised."""
    rng = np.random.default_rng(seed)
    p = rng.dirichlet(np.full(n, alpha))
    p = np.clip(p, clip_min, None)
    return p / p.sum()


def build_distributions() -> dict[str, np.ndarray]:
    """Regenerate the canonical suite from its definition."""
    return {
        "D0": np.full(8, 0.125),
        "D1": np.array([0.05, 0.10, 0.15, 0.20, 0.20, 0.15, 0.10, 0.05]),
        "D2": np.array([0.35, 0.25, 0.15, 0.10, 0.07, 0.04, 0.03, 0.01]),
        "D3": np.array([0.02, 0.03, 0.05, 0.40, 0.40, 0.05, 0.03, 0.02]),
        "D4": np.array([0.02, 0.05, 0.12, 0.31, 0.31, 0.12, 0.05, 0.02]),
        "D5": _dirichlet(alpha=0.5, seed=42),
        "D6": _dirichlet(alpha=1.0, seed=137),
    }


def checksum(vec) -> str:
    """SHA-256 of the vector serialised at 15 decimal places (campaign recipe)."""
    payload = json.dumps([f"{v:.15f}" for v in list(vec)], separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def load_canonical(path: Path | None = None) -> dict[str, dict]:
    """Load the vendored `campaign_distributions.json`."""
    path = path or (_DATA / "campaign_distributions.json")
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def verify_against_canonical(path: Path | None = None) -> dict[str, bool]:
    """
    Regenerate the suite and compare every checksum with the stored file.

    Returns a mapping ``dist_id -> bool``.  A `False` anywhere means the
    distribution actually executed on hardware cannot be reproduced from the
    generator, which invalidates the campaign provenance chain.
    """
    stored = load_canonical(path)
    fresh = build_distributions()
    return {
        d: checksum(fresh[d].tolist()) == stored[d]["checksum_sha256"]
        for d in DIST_IDS
    }
