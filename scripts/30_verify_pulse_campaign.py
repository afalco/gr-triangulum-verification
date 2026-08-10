#!/usr/bin/env python3
"""
30_verify_pulse_campaign.py
===========================
Recomputes every derived quantity of the pulse-level campaign (campaign 3,
Section 9 of the manuscript) from the one primary table that carries actual
measurements -- the averaged probability distributions.

What this establishes
---------------------
* The fidelity / TV / l2 table and the ladder-difference table are exactly
  reproducible from the distributions.  PASS is expected.
* The single-qubit marginal table is NOT reproducible from the same
  distributions.  FAIL is expected, and is the point of this script: marginals
  are linear functionals of the joint law, so
  ``marginal(mean distribution) == mean(marginals)`` must hold identically.

Exit code is non-zero while the marginal inconsistency stands.

Usage
-----
    python scripts/30_verify_pulse_campaign.py
"""

from __future__ import annotations

import csv

import numpy as np

from _common import DATA, Report

from grtri import build_distributions, fidelity, l2_distance, marginals, tv_distance

PULSE = DATA / "pulse_campaign_v3"
TOL = 5e-4          # tables are printed to four decimals
TOL_MARG = 2e-3     # marginals are printed to three decimals


def _read(path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _dist_rows() -> dict[tuple[str, str], np.ndarray]:
    out = {}
    keys = ["p_000", "p_001", "p_010", "p_011",
            "p_100", "p_101", "p_110", "p_111"]
    for row in _read(PULSE / "distributions_avg.csv"):
        vec = np.array([float(row[k]) for k in keys])
        out[(row["config"], row["ladder"])] = vec
    return out


def main() -> int:
    rep = Report("Pulse-level campaign (campaign 3): internal consistency")
    target = build_distributions()["D1"]
    dist = _dist_rows()

    # ---------------------------------------------------------------- 0
    rep.section("Averaged distributions are normalised")
    for (cfg, lad), vec in dist.items():
        rep.check(abs(vec.sum() - 1.0) < 2e-4,
                  f"{cfg} {lad} sums to 1", f"sum = {vec.sum():.6f}")

    # ---------------------------------------------------------------- 1
    rep.section("Table: fidelity / TV / l2 of the averaged distribution")
    print(f"\n  {'cfg':<9}{'lad':<5}"
          f"{'Fid':>9}{'Fid*':>9}{'TV':>9}{'TV*':>9}{'l2':>9}{'l2*':>9}")
    print("  " + "-" * 59)
    for row in _read(PULSE / "metrics.csv"):
        key = (row["config"], row["ladder"])
        vec = dist[key]
        f, t, l = (fidelity(target, vec), tv_distance(target, vec),
                   l2_distance(target, vec))
        print(f"  {row['config']:<9}{row['ladder']:<5}"
              f"{f:>9.4f}{float(row['fidelity']):>9.4f}"
              f"{t:>9.4f}{float(row['tv']):>9.4f}"
              f"{l:>9.4f}{float(row['l2']):>9.4f}")
    print("  (columns marked * are the published values)\n")

    for row in _read(PULSE / "metrics.csv"):
        vec = dist[(row["config"], row["ladder"])]
        label = f"{row['config']} {row['ladder']}"
        rep.check(abs(fidelity(target, vec) - float(row["fidelity"])) < TOL,
                  f"{label}: fidelity reproduces")
        rep.check(abs(tv_distance(target, vec) - float(row["tv"])) < TOL,
                  f"{label}: TV reproduces")
        rep.check(abs(l2_distance(target, vec) - float(row["l2"])) < TOL,
                  f"{label}: l2 reproduces")

    # ---------------------------------------------------------------- 2
    rep.section("Table: ladder A vs B divergence")
    for row in _read(PULSE / "ladder_diff.csv"):
        cfg = row["config"]
        a, b = dist[(cfg, "A")], dist[(cfg, "B")]
        rep.check(abs(tv_distance(a, b) - float(row["tv_ab"])) < TOL,
                  f"{cfg}: TV(A,B) reproduces",
                  f"{tv_distance(a, b):.4f} vs {row['tv_ab']}")
        rep.check(abs(l2_distance(a, b) - float(row["l2_ab"])) < TOL,
                  f"{cfg}: l2(A,B) reproduces",
                  f"{l2_distance(a, b):.4f} vs {row['l2_ab']}")
        rep.check(abs(fidelity(a, b) - float(row["fidelity_ab"])) < TOL,
                  f"{cfg}: Fid(A,B) reproduces (classical fidelity)",
                  f"{fidelity(a, b):.4f} vs {row['fidelity_ab']}")

    # ---------------------------------------------------------------- 3
    rep.section("Table: single-qubit marginals  [EXPECTED TO FAIL]")
    rep.info("Marginals are linear in the distribution, so the marginal of the")
    rep.info("averaged distribution must equal the average of the marginals.")
    print(f"\n  {'cfg':<9}{'lad':<5}{'recomputed (q0,q1,q2)':<28}"
          f"{'published':<24}{'note'}")
    print("  " + "-" * 78)
    for row in _read(PULSE / "marginals.csv"):
        key = (row["config"], row["ladder"])
        got = marginals(dist[key])
        pub = np.array([float(row[f"p_q{j}_1"]) for j in range(3)])
        swapped = np.allclose(got[[2, 1, 0]], pub, atol=TOL_MARG)
        note = ("q0<->q2 swap" if swapped
                else "no relabelling reconciles" if not np.allclose(
                    got, pub, atol=TOL_MARG) else "")
        print(f"  {row['config']:<9}{row['ladder']:<5}"
              f"{np.array2string(got, precision=3):<28}"
              f"{np.array2string(pub, precision=3):<24}{note}")
    print()

    for row in _read(PULSE / "marginals.csv"):
        key = (row["config"], row["ladder"])
        got = marginals(dist[key])
        pub = np.array([float(row[f"p_q{j}_1"]) for j in range(3)])
        rep.check(np.allclose(got, pub, atol=TOL_MARG),
                  f"{row['config']} {row['ladder']}: marginals reproduce",
                  f"max|diff| = {np.abs(got - pub).max():.4f}")

    rep.section("Consequence for the stated conclusion")
    exp2 = np.vstack([dist[("Exp2", "A")], dist[("Exp2", "B")]])
    recomputed = np.vstack([marginals(v) for v in exp2])
    inside = np.all((recomputed >= 0.48) & (recomputed <= 0.52))
    rep.info("Section 9 claims that after global pulse optimisation all "
             "marginals lie in [0.48, 0.52].")
    rep.info(f"Recomputed Exp.2 marginals: "
             f"{np.array2string(recomputed, precision=3)}")
    rep.check(inside,
              "recomputed Exp.2 marginals support the [0.48, 0.52] claim")

    # ---------------------------------------------------------------- 4
    rep.section("Cross-campaign consistency")
    stability = {r["config"]: r for r in _read(PULSE / "stability.csv")}
    ctrl = float(stability["Control"]["fidelity_mean"])
    gate_level_d1_full = 0.909
    rep.info(f"Control mean fidelity (10 runs, per-run mean): {ctrl:.4f}")
    rep.info(f"Gate-level D1 FULL (50 runs, per-run mean):    "
             f"{gate_level_d1_full:.4f}")
    rep.info("Same circuit, same target, same device -- the gap must be "
             "explained in the paper (session and calibration state).")
    if abs(ctrl - gate_level_d1_full) > 0.02:
        rep.warn("Control vs gate-level D1 FULL differ by "
                 f"{abs(ctrl - gate_level_d1_full):.3f} in mean fidelity")

    exp2_std = float(stability["Exp2"]["fidelity_std"])
    shots = 4096
    shot_noise_scale = 1.0 / np.sqrt(shots)
    rep.info(f"Exp.2 fidelity std = {exp2_std:.5f}; "
             f"1/sqrt({shots}) = {shot_noise_scale:.5f}")
    if exp2_std < 0.1 * shot_noise_scale:
        rep.warn("Exp.2 run-to-run spread is far below the naive shot-noise "
                 "scale; the number of shots and the averaging scheme must be "
                 "documented")

    return rep.finish(exit_on_failure=False)


if __name__ == "__main__":
    raise SystemExit(main())
