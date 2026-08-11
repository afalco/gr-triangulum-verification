#!/usr/bin/env python3
"""
50_verify_schedule.py
=====================
Reconstructs the execution order of the 700-run campaign from the timestamps in
the deposited run record, and tests the drift confound that order creates.

Why this script was rewritten
-----------------------------
An earlier version rebuilt the schedule from the constants in
`03_run_campaign.py` and reported the temporal balance that `build_schedule`
*would* produce if the whole suite were interleaved. It warned that it could
not verify what actually ran. That warning turned out to be the important part:
the campaign was executed one distribution per session, using the `--dist`
option, not as one interleaved pass. The figures the old script produced were
correct about the code and wrong about the experiment, and the manuscript
briefly repeated them.

This version reads `campaign_v2_runs_clean.jsonl` and reports what happened.

What it establishes
-------------------
* the campaign ran as seven consecutive per-distribution sessions;
* within each session the four groups are cycled five times, so the stage
  comparison is balanced in time (mean positions 42.0 / 47.0 / 54.5 of 100 in
  every session, 342.0 / 347.0 / 354.5 of 700 pooled);
* the cross-distribution comparison is confounded with session, and the
  confound is therefore tested rather than assumed away.

Usage
-----
    python scripts/60_verify_from_runs.py     # metrics from the same record
    python scripts/50_verify_schedule.py      # execution order and drift
"""

from __future__ import annotations

import json
import statistics as st
from datetime import datetime

from _common import DATA, Report

from grtri import DIST_IDS

RUNS = DATA / "campaign_v2_runs" / "campaign_v2_runs_clean.jsonl"
STAGES = ("L0", "L01", "FULL")


def load():
    rec = []
    with open(RUNS, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                r = json.loads(line)
                r["_t"] = datetime.fromisoformat(
                    str(r["created"]).replace("Z", "+00:00")).replace(tzinfo=None)
                rec.append(r)
    rec.sort(key=lambda r: r["_t"])
    return rec


def main() -> int:
    rep = Report("Campaign execution order, from the run record")
    rec = load()

    rep.section("Record")
    rep.check(len(rec) == 700, "700 runs with timestamps", f"{len(rec)}")
    rep.info(f"acquired {rec[0]['_t']:%d %b %H:%M} to {rec[-1]['_t']:%d %b %H:%M}")

    # ------------------------------------------------- session structure
    order, seen = [], set()
    for r in rec:
        if r["dist_id"] not in seen:
            seen.add(r["dist_id"]); order.append(r["dist_id"])

    rep.section("Session structure")
    print(f"\n  {'#':>2}  {'dist':5}{'start':>16}{'end':>8}{'duration':>12}"
          f"{'FULL fid':>10}")
    print("  " + "-" * 56)
    sessions = {}
    for i, d in enumerate(order, 1):
        s = [r for r in rec if r["dist_id"] == d]
        f = st.mean(float(r["fidelity_vs_target"]) for r in s
                    if r["stage"] == "FULL")
        sessions[d] = (s[0]["_t"], s[-1]["_t"], f)
        start = f"{s[0]['_t']:%d %b %H:%M}"
        end = f"{s[-1]['_t']:%H:%M}"
        dur = str(s[-1]["_t"] - s[0]["_t"]).split(".")[0]
        print(f"  {i:>2}  {d:5}{start:>16}{end:>8}{dur:>12}{f:>10.3f}")
    print()

    rep.check(len(order) == 7, "seven sessions, one per distribution")
    rep.check(order == ["D3", "D2", "D5", "D6", "D4", "D1", "D0"],
              "execution order is D3, D2, D5, D6, D4, D1, D0", str(order))

    contiguous = True
    for d in order:
        idx = [i for i, r in enumerate(rec) if r["dist_id"] == d]
        if idx != list(range(idx[0], idx[0] + 100)):
            contiguous = False
    rep.check(contiguous,
              "each distribution occupies a contiguous block of 100 runs",
              "the campaign was NOT interleaved across distributions")

    durs = [(b - a).total_seconds() / 3600 for a, b, _ in sessions.values()]
    rep.check(3.3 < min(durs) and max(durs) < 3.5,
              "every session lasts about 3 h 25 min",
              f"{min(durs):.2f}-{max(durs):.2f} h")

    # ------------------------------------------------- stage balance
    rep.section("Stage balance within each session")
    for d in order:
        s = [r for r in rec if r["dist_id"] == d]
        m = {k: st.mean([i for i, r in enumerate(s) if r["stage"] == k])
             for k in STAGES}
        ok = all(abs(m[k] - t) < 0.05
                 for k, t in (("L0", 42.0), ("L01", 47.0), ("FULL", 54.5)))
        rep.check(ok, f"{d}: mean positions 42.0 / 47.0 / 54.5 of 100",
                  f"{m['L0']:.1f} / {m['L01']:.1f} / {m['FULL']:.1f}")

    pooled = {k: st.mean([i for i, r in enumerate(rec) if r["stage"] == k])
              for k in STAGES}
    for k, t in (("L0", 342.0), ("L01", 347.0), ("FULL", 354.5)):
        rep.check(abs(pooled[k] - t) < 0.05,
                  f"pooled mean run index at {k} is {t}", f"{pooled[k]:.1f}")

    # ------------------------------------------------- drift tests
    rep.section("Session effect on FULL-stage fidelity")
    try:
        from scipy import stats as sps
    except ImportError:
        rep.warn("scipy not installed; drift tests skipped")
        return rep.finish(exit_on_failure=False)

    f = [sessions[d][2] for d in order]
    rho, p = sps.spearmanr(range(1, 8), f)
    rep.info(f"execution order vs pooled FULL fidelity: rho={rho:+.3f} p={p:.3f}")
    rep.check(abs(rho - 0.321) < 0.01 and p > 0.4,
              "no systematic degradation across sessions")
    rep.check(order[0] == "D3" and order[-1] == "D0",
              "the worst target ran first and the second best last",
              "the opposite of monotone deterioration")

    within = []
    for d in order:
        s = [r for r in rec if r["dist_id"] == d and r["stage"] == "FULL"]
        within.append(sps.spearmanr(
            range(len(s)), [float(r["fidelity_vs_target"]) for r in s])[0])
    rep.info(f"within-session drift rho: {min(within):+.3f} to {max(within):+.3f}"
             f", median {st.median(within):+.3f}")
    rep.check(min(within) > -0.25 and max(within) < 0.35,
              "within-session drift is small and inconsistent in sign")

    rep.section("What this does and does not establish")
    rep.info("The stage comparison is balanced in time by construction.")
    rep.info("The cross-distribution comparison is confounded with session;")
    rep.info("the confound is tested above, not excluded by the design.")
    rep.warn("Distribution is perfectly confounded with acquisition time")

    return rep.finish(exit_on_failure=False)


if __name__ == "__main__":
    raise SystemExit(main())
