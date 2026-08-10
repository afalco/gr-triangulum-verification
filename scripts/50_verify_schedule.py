#!/usr/bin/env python3
"""
50_verify_schedule.py
=====================
Reproduces the temporal-balance figures quoted in the "Temporal ordering and
session drift" paragraph of the manuscript.

Why this matters
----------------
Slow drift in field homogeneity, temperature and pulse calibration is the
standard objection to any comparison drawn across a long NMR campaign. It
threatens two claims at once: the stage-wise progression L0 -> L01 -> FULL, and
the cross-distribution structural analysis. The defence rests entirely on the
execution order, so the execution order should be checkable.

The schedule is rebuilt here from the same constants as
`experiments/campaign_v2/03_run_campaign.py`:

    N_RUNS_PER_GROUP = 25
    STAGES  = ["L0", "L01", "FULL"]
    LADDERS = {"L0": ["A"], "L01": ["A"], "FULL": ["A", "B"]}

giving 7 distributions x 4 groups = 28 groups, cycled five times in blocks of
five runs, for 700 runs total.

Caveat
------
This verifies the schedule the pipeline *builds*. It cannot verify that the
campaign was executed in one pass through that schedule rather than as seven
separately calibrated sessions -- that is a question for the experimental log,
and it is flagged as an open query in the manuscript.

Usage
-----
    python scripts/50_verify_schedule.py
"""

from __future__ import annotations

import statistics

from _common import Report

from grtri import DIST_IDS

N_RUNS_PER_GROUP = 25
STAGES = ("L0", "L01", "FULL")
LADDERS = {"L0": ("A",), "L01": ("A",), "FULL": ("A", "B")}
CYCLES = 5


def build_schedule() -> list[tuple[str, str, str]]:
    """Mirror of `build_schedule` in 03_run_campaign.py."""
    groups = [(d, s, l) for d in DIST_IDS for s in STAGES for l in LADDERS[s]]
    runs_per_cycle = N_RUNS_PER_GROUP // CYCLES
    schedule: list[tuple[str, str, str]] = []
    for _cycle in range(CYCLES):
        for group in groups:
            for _r in range(runs_per_cycle):
                schedule.append(group)
    return schedule


def main() -> int:
    rep = Report("Campaign schedule: temporal balance")
    sched = build_schedule()
    n = len(sched)

    rep.section("Schedule shape")
    n_groups = len(DIST_IDS) * sum(len(LADDERS[s]) for s in STAGES)
    rep.check(n_groups == 28, "28 groups (7 distributions x 4 groups)",
              f"{n_groups}")
    rep.check(n == 700, "700 runs in total", f"{n}")
    rep.check(len({sched[i] for i in range(20)}) > 1,
              "the schedule is interleaved, not distribution-by-distribution",
              f"first 20 runs span {len({sched[i] for i in range(20)})} groups")

    # ---------------------------------------------------------------- stages
    rep.section("Stage-wise temporal balance")
    stage_mean = {}
    for stage in STAGES:
        idx = [i for i, (_, s, _) in enumerate(sched) if s == stage]
        stage_mean[stage] = statistics.mean(idx)
        print(f"    {stage:<5} mean run index = {stage_mean[stage]:7.1f}"
              f"   (n = {len(idx)})")
    spread = max(stage_mean.values()) - min(stage_mean.values())
    print(f"    spread = {spread:.1f} runs out of {n}"
          f"  ({100 * spread / n:.1f}% of the campaign)\n")

    for stage, expected in (("L0", 342.0), ("L01", 347.0), ("FULL", 354.5)):
        rep.check(abs(stage_mean[stage] - expected) < 0.05,
                  f"{stage} mean run index reproduces the manuscript value",
                  f"{stage_mean[stage]:.1f} vs {expected}")
    rep.check(spread < 20,
              "the three stages are near-cotemporal (spread < 20 runs)",
              f"spread = {spread:.1f}")

    # ------------------------------------------------------- distributions
    rep.section("Cross-distribution temporal balance")
    dist_mean, firsts, lasts = {}, {}, {}
    for dist_id in DIST_IDS:
        idx = [i for i, (d, _, _) in enumerate(sched) if d == dist_id]
        dist_mean[dist_id] = statistics.mean(idx)
        firsts[dist_id], lasts[dist_id] = idx[0], idx[-1]
        print(f"    {dist_id}: first = {idx[0]:3d}   last = {idx[-1]:3d}"
              f"   mean = {dist_mean[dist_id]:7.1f}")
    d_spread = max(dist_mean.values()) - min(dist_mean.values())
    print(f"    mean positions span {min(dist_mean.values()):.1f} to "
          f"{max(dist_mean.values()):.1f}"
          f"  ({100 * d_spread / n:.1f}% of the campaign)\n")

    rep.check(abs(min(dist_mean.values()) - 289.5) < 0.05
              and abs(max(dist_mean.values()) - 409.5) < 0.05,
              "mean-position range reproduces the manuscript (289.5 to 409.5)",
              f"{min(dist_mean.values()):.1f} to {max(dist_mean.values()):.1f}")
    rep.check(abs(100 * d_spread / n - 17.1) < 0.2,
              "spread is 17% of the campaign", f"{100 * d_spread / n:.1f}%")
    rep.check(max(firsts.values()) <= 120 and min(lasts.values()) >= 579,
              "every distribution spans essentially the whole campaign",
              f"first runs <= {max(firsts.values())}, "
              f"last runs >= {min(lasts.values())}")
    rep.check(abs(dist_mean["D2"] - 329.5) < 0.05
              and abs(dist_mean["D3"] - 349.5) < 0.05,
              "D2 and D3 are near-cotemporal (the pair raised in review)",
              f"{dist_mean['D2']:.1f} and {dist_mean['D3']:.1f}")

    # ---------------------------------------------------------------- caveat
    rep.section("What this does not establish")
    rep.info("This verifies the schedule the pipeline builds, not the order in")
    rep.info("which the hardware actually ran. If the seven distributions were")
    rep.info("executed as seven separately calibrated sessions, none of the")
    rep.info("figures above apply. That is an open query in the manuscript.")
    rep.warn("Execution order must be confirmed against the experimental log")

    return rep.finish(exit_on_failure=False)


if __name__ == "__main__":
    raise SystemExit(main())
