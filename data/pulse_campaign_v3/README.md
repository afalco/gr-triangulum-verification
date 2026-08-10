# Pulse-level campaign (campaign 3) — data

Source: `main.tex` in the Overleaf project, the report by Yuefeng Lin
*"Experimental Analysis of Grover–Rudolph State Preparation on the SpinQ
Triangulum Platform: Investigating Pulse Calibration, Qubit Asymmetries,
Stability, and Compilation Effects"*.

These CSVs are a **transcription of published summary tables**, not raw run
data. Every value has been transcribed verbatim and then recomputed where
possible by `scripts/30_verify_pulse_campaign.py`.

## Provenance gap

Unlike campaign v2, the underlying run-level data for this campaign is **not in
any repository**. `main.tex` cites three text reports:

- `reports_03_ladder_full_A_vs_B_Control.txt`
- `reports_03_ladder_full_A_vs_B_exp1.txt`
- `reports_03_ladder_full_A_vs_B_exp2.txt`

None of these are present in `gr-triangulum-characterisation`, and there is no
script in that repository that executes the three configurations
(Control / Exp. 1 / Exp. 2) or the direct pulse-level preparation of D0–D6.

**Before submission** the campaign-3 material must either be added to the
repository with its execution scripts, or the paper's data availability
statement must be amended to say what is and is not available. As it stands,
Section 9 of the manuscript is not reproducible from the cited repositories.

## Files

| File | Manuscript table | Content |
|---|---|---|
| `distributions_avg.csv` | Table `tab:pulse-dist` | mean measured distribution, per config × ladder (5 runs each), target D1 |
| `metrics.csv` | Table `tab:pulse-metrics` | fidelity / TV / ℓ² of the **averaged** distribution vs target |
| `marginals.csv` | Table `tab:pulse-marginal` | reported single-qubit marginals `P(q_i=1)` |
| `stability.csv` | Table `tab:pulse-stability` | mean, std and range of the **per-run** fidelity over 10 runs |
| `ladder_diff.csv` | Table `tab:pulse-ladderdiff` | distance between the averaged A and B distributions |
| `pulse_level_d0d6.csv` | Table `tab:pulse_followup_summary` | direct pulse-level preparation, 25 runs per distribution |

## Known inconsistency

`marginals.csv` cannot be recomputed from `distributions_avg.csv`. Marginals are
linear functionals of the joint distribution, so this identity must hold exactly.
Run `scripts/30_verify_pulse_campaign.py` to see the discrepancy in full.
