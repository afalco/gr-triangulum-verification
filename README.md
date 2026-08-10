# gr-triangulum-verification

Verification package for the manuscript

> **Hardware characterisation of exact Grover–Rudolph state preparation on a
> 3-qubit NMR quantum computer** — Falcó, Falcó-Pomares, Latorre & Lin,
> submitted to *Quantum Information Processing*.

This package lets a reader verify every claim in the paper that does **not**
require the physical device: the exactness of the compiled circuits, the
structural descriptors of the benchmark suite, the statistical analysis, and
the internal consistency of the published tables.

It depends only on NumPy (and SciPy for the statistics). **No hardware, no
SpinQ SDK, no credentials.** Everything below runs in a few seconds on a
laptop.

```bash
git clone https://github.com/afalco/gr-triangulum-verification.git
cd gr-triangulum-verification
pip install -r requirements.txt
python scripts/10_verify_distributions.py
python scripts/20_verify_simulator.py
python scripts/30_verify_pulse_campaign.py     # expected to FAIL, see below
python scripts/40_verify_predictors.py
python -m pytest tests/ -v
```

---

## Relationship to the campaign repository

This is **not** a replacement for the campaign repository. The authoritative
execution pipeline lives at

> https://github.com/afalco/gr-triangulum-characterisation

and the core Grover–Rudolph implementation at

> https://github.com/afalco/grover-rudolph-practical-implementation

| | `gr-triangulum-characterisation` | `gr-triangulum-verification` |
|---|---|---|
| Purpose | run the experiments | verify the paper |
| Hardware | required (SpinQit + Triangulum) | never |
| Dependencies | spinqit, numpy, pandas, scipy | numpy, scipy |
| Distributions | generated | vendored + checksum-verified against the generator |
| Simulator | `spinqit.get_basic_simulator()`, sampled | exact state-vector, NumPy |
| Scope | campaign v2 (gate-level, 700 runs) | campaigns v2 **and** v3 (pulse-level) |

The exact state-vector simulator matters: the campaign simulates with 4096
shots, so its "ideal" reference carries sampling noise of order 10⁻². The
manuscript's claim of exactness *up to machine precision* is only checkable
with an analytic simulator, which is what `grtri.simulator` provides
(verified deviation ≤ 5 × 10⁻¹⁶).

---

## Layout

```
gr-triangulum-verification/
├── grtri/                        core package, NumPy only
│   ├── angles.py                 GR dyadic angle map, Gray-code ladders A/B
│   ├── simulator.py              exact 3-qubit state-vector simulator
│   ├── metrics.py                TV, l2, classical fidelity, marginals
│   ├── descriptors.py            Delta_1, Delta_UCRy, entropy, contrast
│   ├── distributions.py          canonical D0–D6 + SHA-256 provenance
│   └── backends.py               Backend ABC, ideal / surrogate / SpinQit
├── data/
│   ├── campaign_distributions.json          vendored from the campaign repo
│   ├── campaign_distributions_check.txt
│   ├── campaign_v2_published/               tables as printed in the paper
│   └── pulse_campaign_v3/                   campaign-3 tables from main.tex
├── scripts/                      four standalone verification scripts
└── tests/                        pytest suite (62 pass, 8 xfail)
```

---

## What each script establishes

### `10_verify_distributions.py`

Regenerates the benchmark suite from its definition and checks all seven
SHA-256 digests against the file that was used to drive the hardware, then
rebuilds **Table 5** of the manuscript from the distributions alone.

All seven rows reproduce to the printed precision, e.g. D3: Δ₁ = 53.13 (paper
53.1), Δ_UCRy = 102.12 (paper 102.1).

### `20_verify_simulator.py`

The central correctness claim. FULL-stage output equals the target to
5 × 10⁻¹⁶ for all seven distributions and both ladders; ladders A and B are
identical in the ideal model; the truncated stages reproduce the partial
supports plotted in the staged-histogram figure; and the level-2 sub-rotations
match the angles printed in the circuit figures (90°, 5.63°, 720°, 13.84° for
ladder A on D1, with the two middle rotations exchanged for ladder B).

It also reproduces the gate counts of the circuit figures — 3 / 13 / 21 for
L0 / L01 / FULL — but only when the *identity-safe tail* is included; see
"Known discrepancies" below.

### `30_verify_pulse_campaign.py`

Recomputes the campaign-3 tables from the one table carrying actual
measurements. The metric table and the ladder-difference table reproduce
exactly. **The marginal table does not** — this script exits non-zero by
design. See "Known discrepancies".

### `40_verify_predictors.py`

Reproduces the headline statistical result: Spearman ρ = −0.643, p = 0.119 for
both Δ₁ and Δ_UCRy against pooled FULL-stage fidelity, matching the published
"ρ = −0.64 with p = 0.12". Confirms the pooled fidelity range 0.871 (D3) to
0.964 (D6), D0 = 0.962, and the best single run 0.989.

---

## Known discrepancies surfaced by this package

These are reported here rather than hidden, because a referee can find all of
them. `REVIEW_NOTES_v02.md` in the parent folder tracks their resolution.

**1. Marginal tables are not recomputable.** Single-qubit marginals are linear
functionals of the joint distribution, so `marginal(mean) = mean(marginals)`
identically. The published marginals disagree with the published
distributions, in campaign 3 (up to 0.117) and, more mildly, in the gate-level
D1 tables. For Exp. 1 the two agree under a q0↔q2 relabelling; for Control and
Exp. 2 no relabelling reconciles them. This undermines the stated conclusion
that global pulse optimisation brings all marginals into [0.48, 0.52] — the
recomputed value is P(q2 = 1) ≈ 0.38.

**2. Four wrong angles in the dyadic-tree figure and Appendix A.5.**
*(fixed in manuscript v03)* The worked D1 example printed θ₀ = 54.74° and
θ₁ = 35.26°, whose correct values are 56.79° and 33.21°, and exchanged
θ₀₀ ↔ θ₁₁. Appendix A.5 carried the same four errors plus the corresponding
commanded angles. The compiled circuits were correct throughout — the L01 block
commands Ry(56.79°) and Ry(663.21°) = Ry(720° − 56.79°) — so no experimental
result was affected, but a referee checking the flagship worked example would
have caught it immediately. Verified by
`tests/test_reproducibility.py::TestCompilation`.

**3. Two different "UCRy" descriptors.** *(fixed in manuscript v03 and in the
campaign repo)* The paper defines Δ_UCRy = max φ₂ − min φ₂; the shipped
`05_characterisation_analysis.py` computed `max_ucry_dev = max|φ₂ − 90|`
instead. They differ (D1: 38.9 vs 19.5) and, critically, they rank the
distributions differently. The paper's headline correlation only holds for the
**range**: with the old covariate, ρ = +0.04, p = 0.94, i.e. no association at
all. The script now computes `delta1` and `ucry_range` and regresses on them;
`max_ucry_dev` is retained only so older artefacts stay regenerable.

Related, and also fixed in v03: the definitions of Δ₁ and Δ_UCRy were written
in terms of the theoretical angles θ but evaluated on the commanded angles
φ = 2θ. Table 5 has always been in φ.

**4. Degrees of freedom stated three ways.** *(fixed in v03)* The results
section named three covariates, the limitations section said "five covariates …
only two residual degrees of freedom", and the shipped script fitted three.
The three-covariate model has 3 residual dof; v03 states this consistently and
reports R² = 0.52 with the fitted coefficients.

**5. Gate counts.** *(explained in v03)* The circuit figures show 3 / 13 / 21
gates, which requires the two identity-equivalent `Ry` gates on q0 (0.06° and
719.94°). The shipped compiler `_spinqit_backend.py` does not emit them and
produces 1 / 11 / 19. Since Experiment 1 of campaign 3 is *defined* as removing
exactly these gates, the compiler that produced the campaign-3 Control group
cannot be the one in the repository. v03 documents the tail in the figure
caption; **the compiler variant that emits it still needs to be published.**

**6. Campaign size.** *(fixed in the campaign repo)* Its README stated 1050
runs (150 per distribution, both ladders at every stage) against the paper's
700 (4 groups × 25 × 7), and listed stale entropies (D1 as 2.77 against the
actual 2.846). The **code was always consistent with the paper** —
`LADDERS = {"L0": ["A"], "L01": ["A"], "FULL": ["A", "B"]}` — so only the
README and one docstring were wrong. Both are now corrected.

**7. Campaign 3 is not in any repository.** *(open)* No script executes the
Control / Exp. 1 / Exp. 2 configurations or the direct pulse-level preparation
of D0–D6, and the three `reports_03_*.txt` files cited in `main.tex` are not
published. Section 9 of the manuscript is currently not reproducible.

---

## Reproducing on hardware

`grtri.backends.SpinQitBackend` delegates to the campaign backend rather than
reimplementing it, so there is exactly one source of truth for the circuit
that reaches the device:

```python
from grtri import SpinQitBackend, build_distributions

backend = SpinQitBackend(
    campaign_dir="…/gr-triangulum-characterisation/experiments/campaign_v2",
    shots=4096,
)
assert backend.angles_agree_with_reference(build_distributions()["D3"])
probs = backend.run(build_distributions()["D3"], stage="FULL", ladder="B")
```

Credentials come from `SPINQ_IP`, `SPINQ_PORT`, `SPINQ_ACCOUNT`,
`SPINQ_PASSWORD` and must never be committed.

`angles_agree_with_reference` guards against silent drift between the vendor
angle map and the reference implementation — worth calling at the start of any
campaign.

---

## Licence

MIT, as the campaign repository.
