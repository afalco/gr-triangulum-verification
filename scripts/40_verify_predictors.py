#!/usr/bin/env python3
"""
40_verify_predictors.py
=======================
Reproduces the predictor analysis of Section "Predictors of FULL-stage
fidelity", which claims:

    "the strongest rank correlations with pooled FULL-stage fidelity are
     obtained for the UCRy range and for the level-1 deviation Delta_1, both
     yielding rho = -0.64 with p = 0.12"

The published per-ladder FULL-stage results are pooled to a per-distribution
mean, then Spearman rank correlations are computed against each structural
descriptor.  The exploratory OLS fit is also reported, with its degrees of
freedom stated explicitly.

Usage
-----
    python scripts/40_verify_predictors.py
"""

from __future__ import annotations

import csv

import numpy as np
from scipy import stats

from _common import DATA, Report

from grtri import DIST_IDS, build_distributions, descriptors

PUBLISHED_RHO = -0.64
PUBLISHED_P = 0.12
TOL_RHO = 0.02
TOL_P = 0.02


def main() -> int:
    rep = Report("Predictors of FULL-stage fidelity")
    dists = build_distributions()

    # ------------------------------------------------------ pooled fidelity
    rows = []
    with open(DATA / "campaign_v2_published" / "full_stage_results.csv",
              newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))

    pooled = {}
    for dist_id in DIST_IDS:
        vals = [float(r["fidelity_mean"]) for r in rows
                if r["dist_id"] == dist_id]
        pooled[dist_id] = float(np.mean(vals))

    rep.section("Pooled FULL-stage fidelity per distribution")
    for dist_id in DIST_IDS:
        print(f"    {dist_id}: {pooled[dist_id]:.4f}")

    lo_id = min(pooled, key=pooled.get)
    hi_id = max(pooled, key=pooled.get)
    rep.check(lo_id == "D3" and abs(pooled["D3"] - 0.871) < 5e-4,
              "lowest pooled fidelity is D3 = 0.871",
              f"{lo_id} = {pooled[lo_id]:.4f}")
    rep.check(hi_id == "D6" and abs(pooled["D6"] - 0.964) < 1e-3,
              "highest pooled fidelity is D6 = 0.964",
              f"{hi_id} = {pooled[hi_id]:.4f}")
    rep.check(abs(pooled["D0"] - 0.962) < 1e-3,
              "uniform reference D0 = 0.962", f"{pooled['D0']:.4f}")
    best_run = max(float(r["fidelity_max"]) for r in rows)
    rep.check(abs(best_run - 0.989) < 5e-4,
              "best individual run of the campaign is 0.989",
              f"{best_run:.3f}")

    # ------------------------------------------------------ rank correlations
    rep.section("Spearman rank correlations with pooled FULL-stage fidelity")
    desc = {d: descriptors(dists[d]) for d in DIST_IDS}
    y = np.array([pooled[d] for d in DIST_IDS])

    covariates = {
        "shannon entropy": [desc[d].shannon_entropy_bits for d in DIST_IDS],
        "contrast": [desc[d].contrast for d in DIST_IDS],
        "log contrast": [np.log(desc[d].contrast) for d in DIST_IDS],
        "Delta_1 (level-1 dev.)": [desc[d].level1_deviation for d in DIST_IDS],
        "Delta_UCRy (range)": [desc[d].ucry_range for d in DIST_IDS],
        "max_ucry_dev (script)": [desc[d].max_ucry_dev for d in DIST_IDS],
    }

    print(f"\n  {'covariate':<26}{'rho':>8}{'p':>9}")
    print("  " + "-" * 43)
    results = {}
    for name, x in covariates.items():
        rho, pval = stats.spearmanr(x, y)
        results[name] = (float(rho), float(pval))
        print(f"  {name:<26}{rho:>8.3f}{pval:>9.3f}")
    print()

    for name in ("Delta_1 (level-1 dev.)", "Delta_UCRy (range)"):
        rho, pval = results[name]
        rep.check(abs(rho - PUBLISHED_RHO) < TOL_RHO,
                  f"{name}: rho reproduces the published -0.64",
                  f"rho = {rho:.3f}")
        rep.check(abs(pval - PUBLISHED_P) < TOL_P,
                  f"{name}: p reproduces the published 0.12",
                  f"p = {pval:.3f}")

    asym = max(abs(results["Delta_1 (level-1 dev.)"][0]),
               abs(results["Delta_UCRy (range)"][0]))
    simple = max(abs(results["shannon entropy"][0]),
                 abs(results["log contrast"][0]))
    rep.check(asym > simple,
              "asymmetry descriptors beat entropy and log-contrast",
              f"|rho| asym = {asym:.3f} vs simple = {simple:.3f}")

    # ------------------------------------------------------ OLS
    rep.section("Exploratory OLS fit (degrees of freedom)")
    X = np.column_stack([
        np.ones(7),
        [desc[d].shannon_entropy_bits for d in DIST_IDS],
        [desc[d].level1_deviation for d in DIST_IDS],
        [desc[d].ucry_range for d in DIST_IDS],
    ])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    ss_res = float(resid @ resid)
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot
    dof = X.shape[0] - X.shape[1]

    for name, b in zip(["intercept", "H_S", "Delta_1", "Delta_UCRy"], beta):
        print(f"    {name:<14}{b:+.5f}")
    print(f"    R^2 = {r2:.4f},  residual dof = {dof}")

    rep.check(dof == 3,
              "the 3-covariate model stated in the results section has 3 dof",
              f"dof = {dof}")
    rep.warn("The limitations section states 'five covariates ... only two "
             "residual degrees of freedom', the results section names three "
             "covariates, and 05_characterisation_analysis.py fits three "
             "(H_S, log contrast, max_ucry_dev). These three statements are "
             "mutually inconsistent and must be reconciled.")

    return rep.finish(exit_on_failure=False)


if __name__ == "__main__":
    raise SystemExit(main())
