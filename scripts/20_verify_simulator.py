#!/usr/bin/env python3
"""
20_verify_simulator.py
======================
Verifies the central correctness claim of the manuscript:

    "The ideal simulator reproduces the target Grover--Rudolph distributions up
     to machine precision."   (Section "Exact behaviour in simulation")

Checks
------
1. FULL-stage output equals the target exactly, for all seven distributions and
   both Gray-code ladders.
2. Ladders A and B produce *identical* output distributions -- the ideal-model
   equivalence that the hardware A/B comparison tests.
3. The truncated stages L0 and L01 reproduce the partial-support distributions
   plotted in the staged-histogram figure.
4. The identity-safe tail is an identity in the ideal model, and including it
   reproduces the gate counts quoted in the circuit figures (3 / 13 / 21).
5. The compiled level-2 sub-rotations reproduce the angles printed in the
   circuit figures for D1.

Needs no hardware and no vendor SDK.

Usage
-----
    python scripts/20_verify_simulator.py
"""

from __future__ import annotations

import numpy as np

from _common import Report

from grtri import (
    DIST_IDS,
    build_distributions,
    gate_counts,
    gr_angles,
    simulate,
)

EPS = 1e-12


def main() -> int:
    rep = Report("Ideal simulator: exactness of the compiled GR circuits")
    dists = build_distributions()

    # ---------------------------------------------------------------- 1
    rep.section("FULL stage reproduces the target to machine precision")
    worst = 0.0
    for dist_id in DIST_IDS:
        p = dists[dist_id]
        for ladder in ("A", "B"):
            q = simulate(p, "FULL", ladder)
            err = float(np.abs(q - p).max())
            worst = max(worst, err)
            rep.check(err < EPS, f"{dist_id} FULL-{ladder} exact",
                      f"max|q-p| = {err:.2e}")
    rep.info(f"worst deviation over all 14 circuits: {worst:.3e}")

    # ---------------------------------------------------------------- 2
    rep.section("Ladders A and B are identical in the ideal model")
    for dist_id in DIST_IDS:
        p = dists[dist_id]
        diff = float(np.abs(simulate(p, "FULL", "A")
                            - simulate(p, "FULL", "B")).max())
        rep.check(diff < EPS, f"{dist_id}: ladder A == ladder B",
                  f"max diff = {diff:.2e}")

    # ---------------------------------------------------------------- 3
    rep.section("Truncated stages reproduce the partial-support targets")
    d1 = dists["D1"]
    expected_l0 = np.array([0.5, 0, 0, 0, 0.5, 0, 0, 0])
    expected_l01 = np.array([0.15, 0, 0.35, 0, 0.35, 0, 0.15, 0])
    rep.check(np.allclose(simulate(d1, "L0", "A"), expected_l0, atol=EPS),
              "D1 L0 matches the staged-histogram reference")
    rep.check(np.allclose(simulate(d1, "L01", "A"), expected_l01, atol=EPS),
              "D1 L01 matches the staged-histogram reference")

    for dist_id in DIST_IDS:
        p = dists[dist_id]
        q0 = simulate(p, "L0", "A")
        s_left = p[:4].sum()
        rep.check(
            np.isclose(q0[0], s_left, atol=1e-12)
            and np.isclose(q0[4], 1.0 - s_left, atol=1e-12)
            and np.isclose(q0[[1, 2, 3, 5, 6, 7]].sum(), 0.0, atol=1e-12),
            f"{dist_id} L0 splits the mass correctly",
            f"P(000)={q0[0]:.6f} vs {s_left:.6f}",
        )

    # ---------------------------------------------------------------- 4
    rep.section("Identity-safe tail and gate counts")
    for dist_id in DIST_IDS:
        p = dists[dist_id]
        diff = float(np.abs(simulate(p, "FULL", "A", identity_tail=False)
                            - simulate(p, "FULL", "A",
                                       identity_tail=True)).max())
        rep.check(diff < EPS, f"{dist_id}: identity tail is an identity",
                  f"max diff = {diff:.2e}")

    published = {"L0": 3, "L01": 13, "FULL": 21}
    for stage, total in published.items():
        with_tail = gate_counts(stage, "A", identity_tail=True)
        without = gate_counts(stage, "A", identity_tail=False)
        rep.check(with_tail["total"] == total,
                  f"{stage} gate count with tail matches the circuit figure",
                  f"{with_tail['total']} vs {total}")
        rep.info(f"{stage}: shipped compiler emits {without['total']} gates "
                 f"({without['Ry']} Ry, {without['CNOT']} CNOT, "
                 f"{without['X']} X) -- no identity tail")
    rep.warn("_spinqit_backend.py does not emit the identity-safe tail, so the "
             "shipped compiler and the manuscript figures disagree on gate counts")

    # ---------------------------------------------------------------- 5
    rep.section("Level-2 sub-rotations against the D1 circuit figure")
    ang = gr_angles(dists["D1"])
    figure_a = np.array([90.0, 5.63, 0.0, 13.84])     # Ry(720) drawn for 0
    figure_b = np.array([90.0, 13.84, 0.0, 5.63])
    rep.check(np.allclose(ang.ladder_angles["A"], figure_a, atol=5e-3),
              "ladder A sub-rotations match the figure",
              np.array2string(ang.ladder_angles["A"], precision=2))
    rep.check(np.allclose(ang.ladder_angles["B"], figure_b, atol=5e-3),
              "ladder B sub-rotations match the figure",
              np.array2string(ang.ladder_angles["B"], precision=2))
    rep.check(np.isclose(ang.phi_level1["0"] / 2.0, 56.79, atol=5e-3)
              and np.isclose(ang.phi_level1["1"] / 2.0, 33.21, atol=5e-3),
              "level-1 half-angles match the figure (56.79 and 33.21 deg)")
    rep.check(np.isclose(ang.phi_root, 90.0, atol=1e-9),
              "root rotation is Ry(90 deg) for D1")

    return rep.finish(exit_on_failure=False)


if __name__ == "__main__":
    raise SystemExit(main())
