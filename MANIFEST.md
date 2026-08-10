# Provenance manifest: manuscript → artefact

Every numbered element of `manuscript_v02.tex` and where it comes from.

Legend for **Status**:

- **exact** — regenerated here and matches to the printed precision
- **published** — transcribed from the paper; primary data lives in the campaign repo
- **gap** — primary data not available in any repository
- **conflict** — recomputation disagrees with the published value

---

## Sections 3–6 — gate-level campaign (v2)

| Manuscript element | Content | Produced by | Status |
|---|---|---|---|
| Fig. `gr-tree` | D1 dyadic tree, θ angles | `grtri.angles.gr_angles` | exact |
| Fig. `circuits` (a)–(d) | compiled L0 / L01 / FULL-A / FULL-B | `grtri.simulator._gate_list` | exact |
| Fig. `circuits` gate counts 3 / 13 / 21 | — | `gate_counts(..., identity_tail=True)` | **conflict** (shipped compiler gives 1 / 11 / 19) |
| §3 TV(p\*,u) = 0.30 for D1 | diagnostic sensitivity | `grtri.metrics.tv_distance` | exact |
| Table `readout` | ΔTV by stage, D1 | campaign repo run log | **conflict** (see §4 note, and 2× vs campaign-level) |
| §4.3 ε(0→1) for Q1, Q2 | assignment errors, D1 | campaign repo run log | **conflict** |
| §6.1 simulator exactness claim | — | `scripts/20_verify_simulator.py` | exact (≤ 5 × 10⁻¹⁶) |
| Table `stage_overall` | D1 stage-wise means | `runs_flat_v2.csv` | published |
| Fig. `histograms` (a)–(c) | staged D1 histograms, blue bars | `grtri.simulator.simulate` | exact |
| Table `ABcompare` | D1 A/B, Welch + Mann–Whitney | `05_characterisation_analysis.py` | published |
| §6.4 mean FULL distributions | D1 ladder A and B | `runs_flat_v2.csv` | published |
| §6.4 marginals (0.486, 0.496, 0.470) | D1 ladder A | — | **conflict** |
| Table `distributions_v2` (Table 5) | H_S, contrast, Δ₁, Δ_UCRy | `scripts/10_verify_distributions.py` | exact |
| Table `full_v2` | FULL-stage per distribution | `05_characterisation_analysis.py` | published |
| §6.7 ρ = −0.64, p = 0.12 | rank correlations | `scripts/40_verify_predictors.py` | exact (−0.643, 0.119) |
| §6.7 pooled range 0.871–0.964 | — | `scripts/40_verify_predictors.py` | exact |
| §6.7 OLS fit | descriptive regression | `scripts/40_verify_predictors.py` | **conflict** (covariate count and dof stated three ways) |
| §6.8 σ range 0.007–0.044 | run-to-run spread | `full_stage_results.csv` | published |

## Section 7 — error budget

| Manuscript element | Content | Produced by | Status |
|---|---|---|---|
| §7.1 ε(0→1) pooled over 175 L0 runs | campaign-level readout | campaign repo run log | published |
| §7.1 affine fit for Q0 | ε(0→1) ≈ 0.059, ε(1→0) ≈ 0.097 | campaign repo run log | published |
| §7.2 σ̂_φ = 2.04° (CI [1.78, 2.27]) | coherent control noise scale | campaign repo run log | published |
| §7.3 D0 FULL TV = 0.151, Fid = 0.962 | depolarising counterexample | `full_stage_results.csv` | exact |

## Section 9 — pulse-level campaign (v3)

| Manuscript element | Content | Produced by | Status |
|---|---|---|---|
| Table `pulse_followup_summary` | direct pulse-level D0–D6 | `data/pulse_campaign_v3/pulse_level_d0d6.csv` | **gap** |
| Table `pulse-dist` | averaged distributions | `data/pulse_campaign_v3/distributions_avg.csv` | **gap** (primary source unavailable) |
| Table `pulse-metrics` | fidelity / TV / ℓ² | `scripts/30_verify_pulse_campaign.py` | exact |
| Table `pulse-marginal` | single-qubit marginals | `scripts/30_verify_pulse_campaign.py` | **conflict** |
| Table `pulse-stability` | per-run fidelity spread | `data/pulse_campaign_v3/stability.csv` | **gap** |
| Table `pulse-ladderdiff` | TV(A,B), ℓ²(A,B), Fid(A,B) | `scripts/30_verify_pulse_campaign.py` | exact |
| §9.2 Control vs gate-level D1 FULL | 0.957 vs 0.909 | `scripts/30_verify_pulse_campaign.py` | **conflict** |

## Appendix A

| Manuscript element | Content | Produced by | Status |
|---|---|---|---|
| A.1 dyadic angle map | definition | `grtri.angles` | exact |
| A.2 trigonometric factorisation | definition | — | analytic |
| A.4 Gray-code ladder compilation | Walsh–Hadamard inversion | `grtri.angles.WALSH_HADAMARD` | exact |
| A.5 angle table for D1 | φ values | `GRAngles.as_flat_dict()` | exact (four angles corrected in v03) |

---

## Summary

| Status | Count |
|---|---|
| exact — regenerated here | 14 |
| published — in the campaign repo, not regenerated here | 8 |
| conflict — recomputation disagrees | 8 |
| gap — primary data unavailable | 4 |

The eight conflicts reduce to four independent root causes: the marginal
convention, the UCRy descriptor definition, the identity tail, and the
cross-campaign session difference. All four are resolvable by the authors
without new experiments, except the last, which needs a sentence of
explanation from Yuefeng Lin.
