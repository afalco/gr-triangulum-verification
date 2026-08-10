"""
grtri.backends
==============
Execution backends for the compiled Grover--Rudolph circuits.

Two implementations are provided:

* `IdealBackend` -- the dependency-free exact state-vector simulator of
  `grtri.simulator`.  Runs anywhere, needs no hardware and no vendor SDK.
* `SpinQitBackend` -- a thin adapter over the vendor SDK, delegating to the
  campaign module `_spinqit_backend.py`.  Importing it requires `spinqit` and
  a reachable Triangulum device.

`NoisyBackend` is also provided for pipeline smoke-testing; it is a crude
depolarising surrogate and must never be used to produce results.
"""

from __future__ import annotations

import abc
import os

import numpy as np

from .angles import gr_angles
from .simulator import simulate

__all__ = ["Backend", "IdealBackend", "NoisyBackend", "SpinQitBackend",
           "spinq_connection_from_env"]


class Backend(abc.ABC):
    """Abstract execution backend returning a length-8 probability vector."""

    name: str = "abstract"

    @abc.abstractmethod
    def run(self, p: np.ndarray, stage: str = "FULL",
            ladder: str = "A") -> np.ndarray:
        """Execute the compiled circuit for target `p` and return probabilities."""


class IdealBackend(Backend):
    """Exact, noiseless, infinite-shot reference."""

    name = "ideal"

    def __init__(self, identity_tail: bool = False):
        self.identity_tail = identity_tail

    def run(self, p, stage="FULL", ladder="A") -> np.ndarray:
        return simulate(p, stage, ladder, identity_tail=self.identity_tail)


class NoisyBackend(Backend):
    """
    Depolarising surrogate for smoke-testing the analysis pipeline only.

    Mirrors the `--dry-run` path of `03_run_campaign.py`, which mixes the ideal
    output with a Dirichlet perturbation.  Not a physical model of the device.
    """

    name = "noisy-surrogate"

    def __init__(self, p_depol: float = 0.05, seed: int | None = None):
        self.p_depol = p_depol
        self._rng = np.random.default_rng(seed)

    def run(self, p, stage="FULL", ladder="A") -> np.ndarray:
        ideal = simulate(p, stage, ladder)
        noise = self._rng.dirichlet(np.full(8, 50.0))
        out = (1.0 - self.p_depol) * ideal + self.p_depol * noise
        return out / out.sum()


def spinq_connection_from_env() -> dict[str, str]:
    """
    Read Triangulum connection parameters from the environment.

    Credentials must never be committed.  The campaign uses:
    ``SPINQ_IP``, ``SPINQ_PORT``, ``SPINQ_ACCOUNT``, ``SPINQ_PASSWORD``.
    """
    missing = [k for k in ("SPINQ_IP", "SPINQ_PORT", "SPINQ_ACCOUNT",
                           "SPINQ_PASSWORD") if not os.environ.get(k)]
    if missing:
        raise RuntimeError(
            "missing environment variables for the SpinQ backend: "
            + ", ".join(missing)
        )
    return {
        "ip": os.environ["SPINQ_IP"],
        "port": os.environ["SPINQ_PORT"],
        "account": os.environ["SPINQ_ACCOUNT"],
        "password": os.environ["SPINQ_PASSWORD"],
    }


class SpinQitBackend(Backend):
    """
    Adapter over the vendor SDK, delegating to the campaign backend module.

    This class deliberately does *not* reimplement the SpinQit calls: the
    authoritative implementation is
    `experiments/campaign_v2/_spinqit_backend.py` in the campaign repository,
    and duplicating it here would create two sources of truth for the compiled
    circuit actually executed on the device.

    Parameters
    ----------
    campaign_dir
        Path to `experiments/campaign_v2` of the campaign repository.
    shots
        Repetitions per run.  The campaign default is 4096.
    """

    name = "spinqit-nmr"

    def __init__(self, campaign_dir: str, shots: int = 4096,
                 conn: dict[str, str] | None = None):
        import sys
        sys.path.insert(0, str(campaign_dir))
        try:
            import _spinqit_backend as vendor      # noqa: F401
        except ImportError as exc:                  # pragma: no cover
            raise RuntimeError(
                "could not import the campaign SpinQit backend. Install the "
                "vendor SDK (`pip install spinqit`) and point `campaign_dir` "
                "at experiments/campaign_v2 of gr-triangulum-characterisation."
            ) from exc
        self._vendor = vendor
        self.shots = shots
        self.conn = conn or spinq_connection_from_env()

    def run(self, p, stage="FULL", ladder="A") -> np.ndarray:
        angles = self._vendor.gr_angles(np.asarray(p, dtype=float))
        return self._vendor.run_hardware(angles, stage, ladder,
                                         self.conn, self.shots)

    def bare_state_check(self) -> np.ndarray:
        """Bare-state health check used by the campaign every 100 runs."""
        return self._vendor.run_bare_hardware(self.conn, self.shots)

    def angles_agree_with_reference(self, p) -> bool:
        """
        Assert that the vendor angle map and the reference implementation in
        `grtri.angles` agree, guarding against silent drift between the two.
        """
        ours = gr_angles(p)
        theirs = self._vendor.gr_angles(np.asarray(p, dtype=float))
        return bool(
            np.isclose(ours.phi_root, theirs["theta0"])
            and np.isclose(ours.phi_level1["0"], theirs["theta1_0"])
            and np.isclose(ours.phi_level1["1"], theirs["theta1_1"])
            and np.allclose(ours.ladder_angles["A"], theirs["ladder_angles_A"])
            and np.allclose(ours.ladder_angles["B"], theirs["ladder_angles_B"])
        )
