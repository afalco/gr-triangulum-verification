# Device log — the instrument's own record

This directory holds an extract of the SpinQuasar job database of the SpinQ
Triangulum used for the paper. It is a **second, independent record** of the
same executions described in `../campaign_v2_runs/`: that one was written by our
orchestration script, this one by the instrument. Neither derives from the
other, which is what makes the comparison in
`scripts/70_verify_device_log.py` worth running.

## Files

| file | rows | what it is |
|---|---|---|
| `device_log_paper.csv` | 986 | one row per experiment: identifiers, timing, circuit summary, measured and ideal 8-outcome distributions |
| `device_log_circuits.jsonl` | 986 | the compiled gate list the instrument actually executed, verbatim |
| `device_log_states.jsonl` | 985 | the measured 8×8 density matrix of each experiment, real and imaginary parts, row-major |

## Provenance

Exported from the Triangulum job database as `spinqit_experiments.csv`
(1211 experiments, 6.8 MB). The instrument is shared with other projects; 225
experiments belonging to those (amplitude estimation, unrelated density-function
studies, untitled local jobs) are excluded here. No value has been edited: every
field is copied or parsed from the export. `build_device_log.py` in this
directory regenerates all three files from it and **fails loudly** on any job
name it does not recognise, so nothing can be included or dropped silently.

## Groups

| `group` | rows | dates | role in the paper |
|---|---|---|---|
| `campaign_v2` | 498 | 1–2 Apr | the 700-run benchmark campaign, §7 |
| `pilot_gr` | 162 | 19 Mar – 26 May | Grover–Rudolph pilots on other targets, **not cited** |
| `characterisation_entangling` | 119 | 22 Apr – 9 Jun | CNOT depth and stress sequences, §8.3 |
| `pilot_d1` | 85 | 3 Mar – 22 Apr | the earlier D1-only session at 2048 shots, §5.3 |
| `characterisation_readout` | 64 | 30 Jan – 28 Mar | readout calibration, confusion matrix, bit-order convention |
| `characterisation_ry` | 58 | 8–9 Jun | SPAM and single-qubit `Ry` error, §8.2 |

`pilot_gr` is deposited although the paper does not cite it, because it is the
control group for the completeness argument below: it is where one would look
first for the missing sessions.

## Read this before using the data

**Join on `experiment_id`, never on `job_name`.** Only `campaign_v2` has unique
job names. Elsewhere the same name was reused across sessions — `GR_L0_B_r1`
appears on 3, 7 and 8 March — so a name-keyed join silently collides. Of the
986 rows only 760 have distinct names.

**The export has no entry for 31 March.** Its daily histogram jumps from 45
experiments on 29 March to 341 on 1 April, and 31 March is the day the D3
(18:01–21:25) and D2 (21:42–01:07) sessions ran. So 200 of the 700 campaign
runs have no instrument-side corroboration. This was checked rather than
assumed: each target commands a distinct set of Gray-code ladder sub-rotations,
recoverable from `device_log_circuits.jsonl`, and fingerprinting all 986
circuits recovers D1, D4, D5 and D6 while returning nothing for D2 or D3 under
any job name. Two further runs, `D4_FULL_A_004` and `D5_FULL_B_017`, are absent
from an otherwise complete 1 April export; both completed normally according to
the run record. Total: 498 of 700.

**The device clock runs two hours ahead** of the UTC timestamps in
`../campaign_v2_runs/`, uniformly, to the second.

**There is no shot count anywhere in the export**, and the database's own
`Fidelity` column is empty for all 1211 rows. The shot count in the paper comes
from the campaign orchestrator's configuration, not from the instrument.

**`submitted_epoch`** (the export's `Identify` field) is a Unix timestamp of
submission, not the platform run identifier recorded in
`../campaign_v2_runs/`; the two ID spaces are unrelated.

**One job failed**: `SPAM_000`, status `FAILED`, with no result vector. It is
kept rather than filtered so the extract matches the export row for row.

## Columns of `device_log_paper.csv`

| column | meaning |
|---|---|
| `experiment_id` | the export's `Number`; unique, use this as the key |
| `job_name` | name submitted to the instrument; not unique |
| `group` | see the table above |
| `dist_id`, `stage`, `ladder`, `repeat` | parsed from the job name, blank where it carries no such structure |
| `status`, `origin` | verbatim from the export |
| `submitted_epoch` | Unix timestamp of submission |
| `created`, `finished` | device clock, two hours ahead of UTC |
| `duration_s` | `finished − created`, in seconds |
| `n_qubits` | the export's `Bits` |
| `n_gates`, `depth`, `n_ry`, `n_x`, `n_cnot` | counted from the compiled gate list; `depth` is the number of distinct timeslots |
| `phi_root_deg` | the first `Ry` on q0, i.e. the commanded root angle, as stored by the instrument |
| `circuit_sha1` | SHA-1 of the canonicalised gate list, for deduplication |
| `exp_p000` … `exp_p111` | measured populations, ordering k = 4q0 + 2q1 + q2 |
| `sim_p000` … `sim_p111` | the instrument's own ideal simulation of the same circuit |

Populations are stored to twelve significant figures. Against the run record
they agree to 5×10⁻¹³ at that precision, and to 2×10⁻¹⁵ against the raw export
— that is, they are the same numbers. The ideal distributions agree only to
2×10⁻⁶, because the instrument stores commanded angles to six significant
figures while our simulator uses them at full precision; the discrepancy is a
property of the log, not of the experiment.

## Regenerating and checking

```bash
python data/device_log/build_device_log.py path/to/spinqit_experiments.csv
python scripts/70_verify_device_log.py
```
