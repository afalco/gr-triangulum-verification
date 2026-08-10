#!/usr/bin/env python3
"""
60_verify_from_runs.py
======================
Regenerates the published tables of the extended campaign directly from the
run-level record, `data/campaign_v2_runs/runs_flat_v2.csv` (700 rows).

Why this script is the important one
------------------------------------
The other scripts cross-check published summaries against each other and
against the compilation. This one starts from primary data and rebuilds the
summaries, which is the check a referee would actually want. It reproduces:

* the campaign design (4 groups per distribution, 25 runs each, 700 total);
* the D1 stage table of the manuscript;
* the stage x distribution table, including the monotone progression and the
  fact that L0 performance does not predict FULL performance;
* the FULL-stage per-ladder table and the pooled fidelity range;
* the campaign-level readout diagnostics of the error-budget section;
* the single-qubit marginals, which are the quantity that was wrong in earlier
  drafts.

Usage
-----
    python scripts/60_verify_from_runs.py
"""

from __future__ import annotations

import csv
import statistics as st

import numpy as np

from _common import DATA, Report

from grtri import DIST_IDS, marginals

RUNS = DATA / "campaign_v2_runs" / "runs_flat_v2.csv"
STAGES = ("L0", "L01", "FULL")
BASIS = ["exp_000", "exp_001", "exp_010", "exp_011",
         "exp_100", "exp_101", "exp_110", "exp_111"]


def load():
    with open(RUNS, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def mean_sd(rows, col):
    v = [float(r[col]) for r in rows]
    return st.mean(v), (st.stdev(v) if len(v) > 1 else 0.0)


def main() -> int:
    rep = Report("Published tables regenerated from the 700-run record")
    rows = load()

    # ---------------------------------------------------------------- design
    rep.section("Campaign design")
    rep.check(len(rows) == 700, "700 runs in the record", f"{len(rows)}")
    groups = sorted({(r["stage"], r["ladder"]) for r in rows})
    rep.check(groups == [("FULL", "A"), ("FULL", "B"), ("L0", "A"), ("L01", "A")],
              "four groups: L0-A, L01-A, FULL-A, FULL-B", str(groups))
    for d in DIST_IDS:
        n = {s: len([r for r in rows if r["dist_id"] == d and r["stage"] == s])
             for s in STAGES}
        rep.check(n == {"L0": 25, "L01": 25, "FULL": 50},
                  f"{d}: 25/25/50 runs by stage", str(n))

    # ------------------------------------------------- stage x distribution
    rep.section("Stage x distribution table")
    tbl = {}
    print(f"\n  {'':5}" + "".join(f"{s:>20}" for s in STAGES))
    for d in DIST_IDS:
        line = f"  {d:5}"
        for s in STAGES:
            sel = [r for r in rows if r["dist_id"] == d and r["stage"] == s]
            m, sd = mean_sd(sel, "fid_tgt")
            tbl[(d, s)] = m
            line += f"{m:>13.3f}+-{sd:.3f}"
        print(line)
    print()

    for d in DIST_IDS:
        rep.check(tbl[(d, "L0")] < tbl[(d, "L01")] < tbl[(d, "FULL")],
                  f"{d}: monotone L0 < L01 < FULL",
                  f"{tbl[(d,'L0')]:.3f} {tbl[(d,'L01')]:.3f} {tbl[(d,'FULL')]:.3f}")

    try:
        from scipy import stats as sps
    except ImportError:
        rep.warn("scipy not installed; rank correlations skipped")
    else:
        l0 = [tbl[(d, "L0")] for d in DIST_IDS]
        l01 = [tbl[(d, "L01")] for d in DIST_IDS]
        full = [tbl[(d, "FULL")] for d in DIST_IDS]
        rho_l0, p_l0 = sps.spearmanr(l0, full)
        rho_l01, _ = sps.spearmanr(l01, full)
        rep.info(f"Spearman(L0, FULL)  = {rho_l0:+.3f}  p = {p_l0:.3f}")
        rep.info(f"Spearman(L01, FULL) = {rho_l01:+.3f}")
        rep.check(abs(rho_l0) < 0.1 and p_l0 > 0.9,
                  "L0 performance does not predict FULL performance")
        rep.check(rho_l01 > 0.8, "L01 already orders the targets as FULL does")
        worst_l0 = min(DIST_IDS, key=lambda d: tbl[(d, "L0")])
        best_full = max(DIST_IDS, key=lambda d: tbl[(d, "FULL")])
        rep.check(worst_l0 == "D6" and best_full == "D6",
                  "D6 is worst at L0 and best at FULL",
                  f"L0 {tbl[('D6','L0')]:.3f} -> FULL {tbl[('D6','FULL')]:.3f}")

    # ------------------------------------------------------- D1 stage table
    rep.section("D1 stage table of the manuscript")
    for stage, fid, tv in (("L0", 0.486, 0.651), ("L01", 0.772, 0.385),
                           ("FULL", 0.909, 0.268)):
        sel = [r for r in rows if r["dist_id"] == "D1" and r["stage"] == stage]
        mf, _ = mean_sd(sel, "fid_tgt")
        mt, _ = mean_sd(sel, "tv_tgt")
        rep.check(abs(mf - fid) < 1e-3, f"D1 {stage} fidelity", f"{mf:.3f} vs {fid}")
        rep.check(abs(mt - tv) < 1.5e-3, f"D1 {stage} TV", f"{mt:.3f} vs {tv}")

    # ------------------------------------------------------ pooled FULL range
    rep.section("FULL-stage pooled fidelity range")
    pooled = {d: mean_sd([r for r in rows if r["dist_id"] == d
                          and r["stage"] == "FULL"], "fid_tgt")[0]
              for d in DIST_IDS}
    lo, hi = min(pooled, key=pooled.get), max(pooled, key=pooled.get)
    rep.check(lo == "D3" and abs(pooled["D3"] - 0.871) < 1e-3,
              "lowest is D3 = 0.871", f"{pooled[lo]:.3f}")
    rep.check(hi == "D6" and abs(pooled["D6"] - 0.964) < 1e-3,
              "highest is D6 = 0.964", f"{pooled[hi]:.3f}")
    best = max(float(r["fid_tgt"]) for r in rows if r["stage"] == "FULL")
    n_full = len([r for r in rows if r["stage"] == "FULL"])
    rep.check(abs(best - 0.989) < 1e-3 and n_full == 350,
              "best of the 350 FULL runs is 0.989",
              f"{best:.3f} of {n_full}")

    # ------------------------------------------------- readout diagnostics
    rep.section("Campaign-level readout diagnostics")
    l0rows = [r for r in rows if r["stage"] == "L0"]
    rep.check(len(l0rows) == 175, "175 L0 runs", f"{len(l0rows)}")
    for q, target, tol in ((1, 0.073, 1e-3), (2, 0.058, 1e-3)):
        v = [1 - float(r[f"exp_q{q}_p0"]) for r in l0rows]
        rep.check(abs(st.mean(v) - target) < tol,
                  f"eps(0->1) on Q{q} = {target}",
                  f"{st.mean(v):.3f} +- {st.stdev(v):.3f}")
    for stage, target in (("L0", 0.500), ("L01", 0.211), ("FULL", 0.000)):
        d = [float(r["tv_tgt"]) - float(r["tv_sim"])
             for r in rows if r["stage"] == stage]
        rep.check(abs(st.mean(d) - target) < 1e-3,
                  f"readout gap at {stage} = {target}", f"{st.mean(d):.3f}")

    # ----------------------------------------------------------- marginals
    rep.section("Single-qubit marginals (the quantity that was wrong)")
    rep.info("The record carries per-run marginal columns as well as the full")
    rep.info("distribution. They must agree, and the manuscript must match both.")
    for ladder, expected in (("A", (0.524, 0.491, 0.450)),
                             ("B", (0.567, 0.515, 0.476))):
        sel = [r for r in rows if r["dist_id"] == "D1"
               and r["stage"] == "FULL" and r["ladder"] == ladder]
        from_cols = np.array([1 - st.mean([float(r[f"exp_q{j}_p0"]) for r in sel])
                              for j in range(3)])
        mean_dist = np.mean([[float(r[k]) for k in BASIS] for r in sel], axis=0)
        from_dist = marginals(mean_dist)
        rep.check(np.allclose(from_cols, from_dist, atol=1e-6),
                  f"ladder {ladder}: recorded marginals equal those of the mean law",
                  np.array2string(from_cols, precision=3))
        rep.check(np.allclose(from_cols, expected, atol=1e-3),
                  f"ladder {ladder}: matches the corrected manuscript values",
                  f"{np.round(from_cols,3)} vs {expected}")

    return rep.finish(exit_on_failure=False)


if __name__ == "__main__":
    raise SystemExit(main())
