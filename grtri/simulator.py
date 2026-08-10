"""
grtri.simulator
===============
Exact 3-qubit state-vector simulator for the compiled Grover--Rudolph circuits,
implemented in NumPy only.

Why this exists
---------------
The campaign pipeline simulates through `spinqit.get_basic_simulator()`, which
requires the vendor SDK.  This module reimplements the *same gate sequence*
with a dependency-free state-vector evolution, so that a reviewer can verify
the central correctness claim of the manuscript -- that the ideal circuit
reproduces the target law to machine precision -- without any hardware, any
vendor SDK, and any sampling noise.

Gate sequences are transcribed verbatim from `_build_circuit` in
`experiments/campaign_v2/_spinqit_backend.py` of the campaign repository.

Conventions
-----------
* Qubit order: ``k = 4*q0 + 2*q1 + q2`` (q0 most significant), matching both
  the manuscript and `per_qubit_marginals` in the campaign code.
* ``Ry(phi) = exp(-i * phi/2 * Y)`` with `phi` in degrees, so ``Ry(720) = I``.
  The compiled circuits use `720 - x` in place of `-x`, which is why several
  commanded angles in the manuscript figures exceed 360 degrees.
"""

from __future__ import annotations

import numpy as np

from .angles import GRAngles, LADDER_CONTROLS, gr_angles

__all__ = ["STAGES", "LADDERS", "statevector", "simulate", "gate_counts"]

STAGES = ("L0", "L01", "FULL")
LADDERS = ("A", "B")

_N_QUBITS = 3
_DIM = 2 ** _N_QUBITS


# --------------------------------------------------------------------------
# Elementary gate application on a length-8 state vector
# --------------------------------------------------------------------------

def _apply_1q(state: np.ndarray, gate: np.ndarray, qubit: int) -> np.ndarray:
    """
    Apply a 2x2 gate to `qubit` (0 = most significant).

    With ``k = 4*q0 + 2*q1 + q2`` and C-order reshaping, the tensor index
    ``(i, j, l)`` corresponds to ``(q0, q1, q2)``, so qubit `q` is axis `q`.
    """
    axis = qubit
    tensor = state.reshape((2,) * _N_QUBITS)
    tensor = np.moveaxis(tensor, axis, 0)
    shape = tensor.shape
    tensor = (gate @ tensor.reshape(2, -1)).reshape(shape)
    return np.moveaxis(tensor, 0, axis).reshape(_DIM)


def _apply_cx(state: np.ndarray, control: int, target: int) -> np.ndarray:
    """Apply CNOT with the given control and target qubit indices."""
    if control == target:
        raise ValueError("control and target must differ")
    out = state.copy()
    c_bit = _N_QUBITS - 1 - control
    t_bit = _N_QUBITS - 1 - target
    for k in range(_DIM):
        if (k >> c_bit) & 1:
            j = k ^ (1 << t_bit)
            if j > k:
                out[k], out[j] = state[j], state[k]
    return out


def _ry(phi_deg: float) -> np.ndarray:
    """Ry(phi) = exp(-i phi/2 Y), phi in degrees."""
    h = np.radians(phi_deg) / 2.0
    return np.array([[np.cos(h), -np.sin(h)],
                     [np.sin(h), np.cos(h)]], dtype=complex)


_X = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex)


# --------------------------------------------------------------------------
# Compiled circuit
# --------------------------------------------------------------------------

IDENTITY_TAIL_DEG = (0.06, 719.94)
"""
The two commanded angles of the 'identity-safe tail' applied to q0 at the end of
every stage in the manuscript circuit figures.  They sum to 720 degrees and are
therefore an identity in the ideal model, but each is realised as a physical
pulse of finite accuracy.  Experiment 1 of the pulse-level campaign
(Section 9 of the manuscript) removes exactly these gates.

Note that `_spinqit_backend.py` in the campaign repository does **not** emit
this tail, so the shipped compiler produces 1/11/19 gates for L0/L01/FULL
whereas the manuscript figures quote 3/13/21.  Set ``identity_tail=True`` to
reproduce the figures.
"""


def _gate_list(angles: GRAngles, stage: str, ladder: str,
               identity_tail: bool = False) -> list[tuple]:
    """
    Return the compiled circuit as a list of gate tuples.

    Each tuple is either ``("ry", qubit, phi_deg)``, ``("x", qubit)`` or
    ``("cx", control, target)``.

    Parameters
    ----------
    identity_tail
        If True, append the two identity-equivalent `Ry` gates on q0 shown in
        the manuscript circuit figures.  Default False, matching the shipped
        campaign compiler.
    """
    if stage not in STAGES:
        raise ValueError(f"unknown stage {stage!r}, expected one of {STAGES}")
    if ladder not in LADDERS:
        raise ValueError(f"unknown ladder {ladder!r}, expected one of {LADDERS}")

    def finish(gates: list[tuple]) -> list[tuple]:
        if identity_tail:
            gates = gates + [("ry", 0, IDENTITY_TAIL_DEG[0]),
                             ("ry", 0, IDENTITY_TAIL_DEG[1])]
        return gates

    ops: list[tuple] = [("ry", 0, angles.phi_root)]
    if stage == "L0":
        return finish(ops)

    # Level 1: single-control UCRy on q1 conditioned on q0.
    for control_value in ("0", "1"):
        half = angles.phi_level1[control_value] / 2.0
        if control_value == "0":
            ops += [("ry", 1, half), ("x", 0), ("cx", 0, 1),
                    ("ry", 1, 720.0 - half), ("cx", 0, 1), ("x", 0)]
        else:
            ops += [("ry", 1, half), ("cx", 0, 1),
                    ("ry", 1, 720.0 - half), ("cx", 0, 1)]
    if stage == "L01":
        return finish(ops)

    # Level 2: two-control UCRy on q2 as a Gray-code ladder.
    sub_angles = angles.ladder_angles[ladder]
    controls = LADDER_CONTROLS[ladder]
    for phi, ctrl in zip(sub_angles, controls):
        ops += [("ry", 2, float(phi)), ("cx", ctrl, 2)]
    return finish(ops)


def gate_counts(stage: str, ladder: str = "A",
                identity_tail: bool = False) -> dict[str, int]:
    """Gate counts of a compiled stage, as quoted in the manuscript figures."""
    dummy = gr_angles(np.full(8, 0.125))
    ops = _gate_list(dummy, stage, ladder, identity_tail=identity_tail)
    counts = {"Ry": 0, "X": 0, "CNOT": 0}
    for op in ops:
        counts[{"ry": "Ry", "x": "X", "cx": "CNOT"}[op[0]]] += 1
    counts["total"] = sum(counts.values())
    return counts


def statevector(angles: GRAngles, stage: str, ladder: str = "A",
                identity_tail: bool = False) -> np.ndarray:
    """Evolve |000> through the compiled circuit and return the state vector."""
    state = np.zeros(_DIM, dtype=complex)
    state[0] = 1.0
    for op in _gate_list(angles, stage, ladder, identity_tail=identity_tail):
        if op[0] == "ry":
            state = _apply_1q(state, _ry(op[2]), op[1])
        elif op[0] == "x":
            state = _apply_1q(state, _X, op[1])
        else:
            state = _apply_cx(state, op[1], op[2])
    return state


def simulate(p: np.ndarray | list[float], stage: str = "FULL",
             ladder: str = "A", identity_tail: bool = False) -> np.ndarray:
    """
    Ideal computational-basis output distribution of the compiled circuit for
    target `p`.

    This is the noiseless, infinite-shot reference against which every hardware
    run in the campaign is compared.
    """
    angles = gr_angles(p)
    psi = statevector(angles, stage, ladder, identity_tail=identity_tail)
    probs = np.abs(psi) ** 2
    return probs / probs.sum()
