#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build the deposited extract of the SpinQ device log.

Input   spinqit_experiments.csv, the full export of the Triangulum job database
        (1211 experiments, several unrelated projects, 6.8 MB of embedded JSON).
Output  device_log_paper.csv        one row per experiment
        device_log_circuits.jsonl   the compiled circuit of each
        device_log_states.jsonl     the measured 8x8 density matrix

The selection keeps every experiment the manuscript rests on and discards the
unrelated projects that share the instrument. Nothing is edited: every value is
copied or parsed from the export. An unrecognised job name is a hard error, so
neither an inclusion nor an exclusion can happen silently.

Usage
-----
    python data/device_log/build_device_log.py path/to/spinqit_experiments.csv
"""
from __future__ import annotations

import csv
import hashlib
import json
import re
import sys
from datetime import datetime
from pathlib import Path

OUT = Path(__file__).resolve().parent

KEYS = ("000", "001", "010", "011", "100", "101", "110", "111")

# --------------------------------------------------------------------------
# Group assignment. The first pattern that matches wins; anything unmatched is
# dropped and reported, so a new job name can never be silently included.
# --------------------------------------------------------------------------
GROUPS: list[tuple[str, str, str]] = [
    (r"^gr_D\d_(L0|L01|FULL)_[AB]_\d+$", "campaign_v2",
     "the 700-run benchmark campaign of Section 7"),

    (r"^GR_(L0|L01|FULL)_[AB]_r\d+$", "pilot_d1",
     "the earlier D1-only session at 2048 shots used in Section 5.3"),

    (r"^GR_(skewleft|skewright|palindromic|example|test_full_B)_n\d+", "pilot_gr",
     "Grover-Rudolph pilots on other targets, not cited in the paper"),

    (r"^(RO_FULL_prep_\d+|X_q\d+|bit_order_calib\w*|cal_Ry_q\d+)$",
     "characterisation_readout",
     "readout calibration, confusion matrix and bit-order convention"),

    (r"^(SPAM_\d+|Ry_Error_Gates_\d+)$", "characterisation_ry",
     "SPAM and single-qubit Ry error, Section 8.2"),

    (r"^(CX_STRESS\w*|CNOT_Stress_Gates_\d+|CX_DUMMY_r\d+)$",
     "characterisation_entangling",
     "CNOT depth and stress sequences, Section 8.3"),
]

# Job families that belong to other projects on the same instrument.
EXCLUDE = re.compile(
    r"^(qae_mlae|poda_lognorm|triangular_simetrica|dientes|exponencial|"
    r"gaussiana|bimodal|por_etapas|graycode_simulacion|naive_transpilacion|"
    r"Untitled Task)")


def classify(name: str) -> str | None:
    for pat, group, _ in GROUPS:
        if re.match(pat, name):
            return group
    return None


def parse_name(name: str) -> dict[str, str]:
    m = re.match(r"^gr_(D\d)_(L0|L01|FULL)_([AB])_(\d+)$", name)
    if m:
        return {"dist_id": m.group(1), "stage": m.group(2),
                "ladder": m.group(3), "repeat": str(int(m.group(4)))}
    m = re.match(r"^GR_(L0|L01|FULL)_([AB])_r(\d+)$", name)
    if m:
        return {"dist_id": "D1", "stage": m.group(1),
                "ladder": m.group(2), "repeat": str(int(m.group(3)))}
    return {"dist_id": "", "stage": "", "ladder": "", "repeat": ""}


def iso(s: str) -> datetime | None:
    try:
        return datetime.fromisoformat((s or "").split(".")[0])
    except ValueError:
        return None


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print(__doc__.strip())
        return 2
    src = Path(argv[1])
    if not src.is_file():
        print(f"no such file: {src}")
        return 2
    rows = list(csv.DictReader(src.open(encoding="utf-8", errors="replace")))
    print(f"read {len(rows)} experiments from the export")

    kept, dropped, unknown = [], 0, {}
    for r in rows:
        name = (r["Nmae"] or "").strip()
        g = classify(name)
        if g is None:
            if EXCLUDE.match(name):
                dropped += 1
            else:
                unknown[re.sub(r"\d+", "#", name)] = \
                    unknown.get(re.sub(r"\d+", "#", name), 0) + 1
            continue
        r["_group"] = g
        kept.append(r)

    if unknown:
        print("!! job names matched neither an included nor an excluded family:")
        for k, v in sorted(unknown.items()):
            print(f"     {v:4}  {k}")
        return 1
    print(f"keeping {len(kept)}, dropping {dropped} from unrelated projects")

    OUT.mkdir(parents=True, exist_ok=True)
    fields = (["experiment_id", "job_name", "group", "dist_id", "stage",
               "ladder", "repeat", "status", "origin", "submitted_epoch",
               "created", "finished", "duration_s", "n_qubits", "n_gates",
               "depth", "n_ry", "n_x", "n_cnot", "phi_root_deg",
               "circuit_sha1"]
              + [f"exp_p{k}" for k in KEYS] + [f"sim_p{k}" for k in KEYS])

    fcsv = (OUT / "device_log_paper.csv").open("w", newline="", encoding="utf-8")
    w = csv.DictWriter(fcsv, fieldnames=fields)
    w.writeheader()
    fcir = (OUT / "device_log_circuits.jsonl").open("w", encoding="utf-8")
    fsta = (OUT / "device_log_states.jsonl").open("w", encoding="utf-8")

    kept.sort(key=lambda r: (iso(r["CreatedTime"]) or datetime.min, r["Nmae"]))
    for r in kept:
        name = r["Nmae"].strip()
        gates = json.loads(r["Circuit"])["gates"]
        exp = json.loads(r["Experiment Result"])
        sim = json.loads(r["Simulation Result"])
        a, b = iso(r["CreatedTime"]), iso(r["FinishedTime"])
        types = [g["type"] for g in gates]
        blob = json.dumps(gates, sort_keys=True, separators=(",", ":"))
        root = [g for g in gates
                if g["type"] == "Ry" and g["timeslot"] == 0
                and g["qubitIndex"] == 0]

        rec = {
            "experiment_id": r["Number"], "job_name": name,
            "group": r["_group"], **parse_name(name),
            "status": r["Status"], "origin": r["Origin"],
            "submitted_epoch": r["Identify"],
            "created": r["CreatedTime"], "finished": r["FinishedTime"],
            "duration_s": f"{(b - a).total_seconds():.0f}" if a and b else "",
            "n_qubits": r["Bits"], "n_gates": len(gates),
            "depth": max((g["timeslot"] for g in gates), default=-1) + 1,
            "n_ry": types.count("Ry"), "n_x": types.count("X"),
            "n_cnot": types.count("CNOT"),
            "phi_root_deg": f"{root[0]['angle']:.4f}" if root else "",
            "circuit_sha1": hashlib.sha1(blob.encode()).hexdigest(),
        }
        ex, sm = exp.get("execution"), sim.get("simulation")
        for i, k in enumerate(KEYS):
            rec[f"exp_p{k}"] = f"{ex[i]:.12g}" if ex and len(ex) == 8 else ""
            rec[f"sim_p{k}"] = f"{sm[i]:.12g}" if sm and len(sm) == 8 else ""
        w.writerow(rec)

        fcir.write(json.dumps({"experiment_id": r["Number"], "job_name": name,
                               "gates": gates}, separators=(",", ":")) + "\n")
        m = exp.get("execution_matrix")
        if m:
            fsta.write(json.dumps({"experiment_id": r["Number"],
                                   "job_name": name,
                                   "real": m["real"], "imag": m["imag"]},
                                  separators=(",", ":")) + "\n")

    fcsv.close(); fcir.close(); fsta.close()
    for p in sorted(OUT.glob("device_log_*")):
        print(f"  {p.name:32} {p.stat().st_size/1e6:7.3f} MB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
