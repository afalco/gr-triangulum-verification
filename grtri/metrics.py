"""
grtri.metrics
=============
Distribution-level comparison metrics used throughout the manuscript
(Section "Metrics for distribution-level comparison") and the campaign code.

All functions take length-8 probability vectors indexed as
``k = 4*q0 + 2*q1 + q2``.

A note on two averaging conventions
-----------------------------------
The manuscript reports metrics under two different conventions, which are *not*
interchangeable:

* **mean of per-run metrics** -- used by the gate-level campaign tables
  (e.g. "fidelity 0.909 +/- 0.018 over 50 runs");
* **metric of the averaged distribution** -- used by the pulse-level campaign
  tables (e.g. "Control A, fidelity 0.9618").

Use `mean_of_metric` and `metric_of_mean` to make the choice explicit rather
than implicit.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np

__all__ = [
    "tv_distance",
    "l2_distance",
    "fidelity",
    "marginals",
    "shannon_entropy",
    "contrast",
    "mean_of_metric",
    "metric_of_mean",
]


def _as_vec(p) -> np.ndarray:
    v = np.asarray(p, dtype=float)
    if v.shape != (8,):
        raise ValueError(f"expected a length-8 vector, got shape {v.shape}")
    return v


def tv_distance(p, q) -> float:
    """Total variation distance, ``0.5 * sum |p_k - q_k|``."""
    return float(0.5 * np.sum(np.abs(_as_vec(p) - _as_vec(q))))


def l2_distance(p, q) -> float:
    """Euclidean distance between the two probability vectors."""
    return float(np.sqrt(np.sum((_as_vec(p) - _as_vec(q)) ** 2)))


def fidelity(p, q) -> float:
    """Classical fidelity ``(sum_k sqrt(p_k q_k))^2``."""
    a, b = _as_vec(p), _as_vec(q)
    return float(np.sum(np.sqrt(np.clip(a, 0, None) * np.clip(b, 0, None))) ** 2)


def marginals(p) -> np.ndarray:
    """
    Single-qubit marginals ``P(q_j = 1)`` for j = 0, 1, 2.

    Uses ``k = 4*q0 + 2*q1 + q2``, i.e. q0 is the most significant bit -- the
    same convention as `per_qubit_marginals` in the campaign repository.

    Marginals are *linear* in the distribution, so the marginal of an averaged
    distribution always equals the average of the per-run marginals.  Any
    reported pair that violates this identity indicates a convention mismatch
    upstream.
    """
    v = _as_vec(p)
    return np.array([
        float(sum(v[k] for k in range(8) if (k >> (2 - j)) & 1))
        for j in range(3)
    ])


def shannon_entropy(p) -> float:
    """Shannon entropy in bits, ignoring zero entries."""
    v = _as_vec(p)
    nz = v[v > 0]
    return float(-np.sum(nz * np.log2(nz)))


def contrast(p) -> float:
    """Ratio of largest to smallest non-zero probability."""
    nz = _as_vec(p)[_as_vec(p) > 0]
    return float(nz.max() / nz.min())


def mean_of_metric(metric: Callable[[np.ndarray, np.ndarray], float],
                   target, runs: Sequence) -> tuple[float, float]:
    """
    Mean and (population) standard deviation of a metric evaluated run by run.

    This is the convention of the gate-level campaign tables.
    """
    vals = [metric(target, r) for r in runs]
    return float(np.mean(vals)), float(np.std(vals))


def metric_of_mean(metric: Callable[[np.ndarray, np.ndarray], float],
                   target, runs: Sequence) -> float:
    """
    Metric evaluated on the averaged distribution.

    This is the convention of the pulse-level campaign tables.
    """
    mean_dist = np.mean(np.vstack([_as_vec(r) for r in runs]), axis=0)
    return metric(target, mean_dist)
