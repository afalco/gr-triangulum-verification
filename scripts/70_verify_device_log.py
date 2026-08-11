#!/usr/bin/env python3
"""
70_verify_device_log.py
=======================
Cross-checks the deposited extract of the SpinQ device log against the campaign
run record, and against the commanded angles computed from the target
distributions.

Why this matters
----------------
Every number in the paper is derived from `campaign_v2_runs_clean.jsonl`, which
was written by our own orchestration script. The device log is the instrument's
own record of the same jobs, exported from the SpinQuasar job database and
never touched by our pipeline. Where the two overlap they are an independent
check on each other: if the orchestrator had mislabelled a run, mixed up a
ladder, or silently retried a job, the two records would disagree.

They do not. The measured 8-outcome distributions agree to the twelve
significant figures the extract stores, which is the full precision available;
against the raw export the agreement is 2e-15, that is, they are the same
floating-point numbers. Job names, timestamps and compiled circuits line up
too.

What this script establishes
----------------------------
* the device log reproduces 498 of the 700 runs exactly;
* 200 of the 202 that are missing are missing because the export contains no
  entry at all for 31 March, the day the D3 and D2 sessions ran -- not because
  those sessions carry different job names, which is ruled out by angle
  fingerprinting. The remaining two are single runs, D4_FULL_A_004 and
  D5_FULL_B_017, absent from an otherwise complete 1 April export;
* the circuits the instrument executed command the angles that `grtri.gr_angles`
  derives from the target distributions, to the precision the log stores;
* the measured density matrices are consistent with the reported populations.

Usage
-----
    python scripts/70_verify_device_log.py
"""

from __future__ import annotations

import csv
import json
from collections import Counter
from datetime import datetime

import numpy as np

from _common import DATA, Report

from grtri.angles import gr_angles
from grtri.distributions import build_distributions

LOG = DATA / "device_log"
RUNS = DATA / "campaign_v2_runs" / "campaign_v2_runs_clean.jsonl"
KEYS = ("000", "001", "010", "011", "100", "101", "110", "111")

# The device clock runs two hours ahead of the orchestrator's UTC timestamps.
CLOCK_OFFSET_H = 2


def load_log() -> list[dict]:
    with (LOG / "device_log_paper.csv").open(encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def load_runs() -> dict[str, dict]:
    out = {}
    with RUNS.open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                r = json.loads(line)
                out[r["run_name"]] = r
    return out


def probs(row: dict, prefix: str) -> np.ndarray | None:
    if row[f"{prefix}_p000"] == "":
        return None
    return np.array([float(row[f"{prefix}_p{k}"]) for k in KEYS])


def main() -> int:
    rep = Report("Device log versus the campaign run record")
    log = load_log()
    runs = load_runs()

    # ------------------------------------------------------------ inventory
    rep.section("Inventory")
    rep.check(len(log) == 986, "986 experiments in the deposited extract",
              str(len(log)))
    rep.check(len({r["experiment_id"] for r in log}) == len(log),
              "experiment_id is unique; join on it, not on job_name")
    rep.check(len({r["job_name"] for r in log}) < len(log),
              "job_name is NOT unique across sessions",
              f"{len({r['job_name'] for r in log})} distinct names")

    groups = Counter(r["group"] for r in log)
    for g, n in sorted(groups.items()):
        rep.info(f"{g:28} {n:4}")
    rep.check(groups["campaign_v2"] == 498,
              "498 campaign runs in the device log", str(groups["campaign_v2"]))

    # --------------------------------------------------- agreement on probs
    rep.section("Agreement with the run record")
    c2 = {r["job_name"][3:]: r for r in log if r["group"] == "campaign_v2"}
    common = sorted(set(c2) & set(runs))
    rep.check(len(common) == 498,
              "every campaign run in the log is in the run record",
              f"{len(common)} matched")

    de = np.array([probs(c2[n], "exp") for n in common])
    re_ = np.array([[runs[n]["exp_probs"][k] for k in KEYS] for n in common])
    d = np.abs(de - re_).max()
    rep.check(d < 1e-12,
              "measured distributions agree to floating-point identity",
              f"max |device - record| = {d:.2e}")

    ds = np.array([probs(c2[n], "sim") for n in common])
    rs = np.array([[runs[n]["sim_probs"][k] for k in KEYS] for n in common])
    d = np.abs(ds - rs).max()
    rep.check(1e-9 < d < 1e-4,
              "ideal distributions agree to the log's angle precision",
              f"max |device - record| = {d:.2e}")
    rep.info("the residual comes from the log storing commanded angles to six")
    rep.info("significant figures; our simulator uses them at full precision")

    ok = all(c2[n]["dist_id"] == runs[n]["dist_id"]
             and c2[n]["stage"] == runs[n]["stage"]
             and c2[n]["ladder"] == runs[n]["ladder"] for n in common)
    rep.check(ok, "distribution, stage and ladder labels agree on every run")

    off = []
    for n in common:
        a = datetime.fromisoformat(c2[n]["created"])
        b = datetime.fromisoformat(runs[n]["created"]).replace(tzinfo=None)
        off.append((a - b).total_seconds() / 3600)
    rep.check(all(abs(x - CLOCK_OFFSET_H) < 0.01 for x in off),
              f"the device clock runs {CLOCK_OFFSET_H} h ahead, consistently",
              f"{min(off):.3f} to {max(off):.3f} h")

    # -------------------------------------------------------- what is absent
    rep.section("The 202 runs the export does not contain")
    missing = sorted(set(runs) - set(c2))
    rep.check(len(missing) == 202, "202 runs absent", str(len(missing)))
    by_dist = Counter(m.split("_")[0] for m in missing)
    rep.check(by_dist["D2"] == 100 and by_dist["D3"] == 100,
              "200 of them are the whole of the D2 and D3 sessions")
    rep.check(sorted(m for m in missing if not m.startswith(("D2", "D3")))
              == ["D4_FULL_A_004", "D5_FULL_B_017"],
              "the other two are single runs dropped from the 1 April export",
              "both completed normally in the run record")
    days = {r["created"][:10] for r in log}
    rep.check("2026-03-31" not in days,
              "the export contains no entry at all for 31 March",
              "the day both sessions ran, so the absence is an export gap")

    dists = build_distributions()

    def fingerprint(dist_id: str) -> set[float]:
        a = gr_angles(dists[dist_id])
        s = {round(abs(a.phi_root), 2)}
        s |= {round(abs(v), 2) for v in a.phi_level1.values()}
        for lad in ("A", "B"):
            s |= {round(abs(float(x)), 2) for x in a.ladder_angles[lad]}
        return s - {0.0}

    circuits = {}
    with (LOG / "device_log_circuits.jsonl").open(encoding="utf-8") as fh:
        for line in fh:
            c = json.loads(line)
            circuits[str(c["experiment_id"])] = \
                {round(abs(g.get("angle", 0)), 2) for g in c["gates"]} - {0.0}

    rep.info("angle fingerprinting over all 986 circuits:")
    for d_id in ("D1", "D4", "D5", "D6", "D2", "D3"):
        fp = fingerprint(d_id)
        n = sum(1 for a in circuits.values() if len(fp & a) >= 2)
        if d_id in ("D2", "D3"):
            rep.check(n == 0, f"{d_id}: not present under any job name",
                      f"{n} circuits carry its angles")
        else:
            rep.check(n >= 49, f"{d_id}: recovered by its angles",
                      f"{n} circuits")
    rep.info("D0 is exempt: its commanded angles are (90, 0, 0, 0) and the")
    rep.info("fingerprint collapses to a single value shared with other targets")

    # --------------------------------------------------- internal coherence
    rep.section("Internal coherence of the extract")
    by_id = {r["experiment_id"]: r for r in log}
    worst, checked = 0.0, 0
    with (LOG / "device_log_states.jsonl").open(encoding="utf-8") as fh:
        for line in fh:
            s = json.loads(line)
            R = np.array(s["real"], float)
            r = by_id.get(str(s["experiment_id"]))
            if r is None or R.size != 64:
                continue
            p = probs(r, "exp")
            if p is None:
                continue
            worst = max(worst, float(np.abs(np.diag(R.reshape(8, 8)) - p).max()))
            checked += 1
    rep.check(worst < 1e-9,
              "the diagonal of every deposited density matrix is its "
              "reported population vector",
              f"{checked} matrices, max deviation {worst:.2e}")

    bad = [r for r in log if r["status"] != "SUCCEED"]
    rep.check(len(bad) == 1, "one failed job in the extract, kept for honesty",
              ", ".join(f"{r['job_name']} ({r['status']})" for r in bad))

    rep.section("What this does and does not establish")
    rep.info("It establishes that the run record and the instrument's own")
    rep.info("database describe the same 498 executions, digit for digit.")
    rep.warn("Neither record stores the shot count, so Q1b stays open")
    rep.warn("The D2 and D3 sessions have no instrument-side corroboration")

    return rep.finish(exit_on_failure=False)


if __name__ == "__main__":
    raise SystemExit(main())
