"""
Reproducibility test suite.

    python -m pytest tests/ -v

Tests are grouped by what they establish:

* `TestProvenance`   -- the benchmark suite regenerates bit-for-bit.
* `TestExactness`    -- the compiled circuits are exact in the ideal model.
* `TestCompilation`  -- angles and gate counts match the manuscript figures.
* `TestPaperTables`  -- published tables recompute from the published data.
* `TestKnownIssues`  -- documented inconsistencies, marked `xfail`.

`TestKnownIssues` is expected to XFAIL. Those tests turn into XPASS -- and the
suite starts reporting them as unexpected passes -- once the underlying data is
corrected, which is exactly the signal wanted.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from grtri import (  # noqa: E402
    DIST_IDS,
    build_distributions,
    descriptors,
    fidelity,
    gate_counts,
    gr_angles,
    l2_distance,
    marginals,
    simulate,
    tv_distance,
    verify_against_canonical,
)

DATA = ROOT / "data"
PULSE = DATA / "pulse_campaign_v3"
PUB = DATA / "campaign_v2_published"


def _read(path):
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _pulse_distributions():
    keys = ["p_000", "p_001", "p_010", "p_011",
            "p_100", "p_101", "p_110", "p_111"]
    return {(r["config"], r["ladder"]): np.array([float(r[k]) for k in keys])
            for r in _read(PULSE / "distributions_avg.csv")}


@pytest.fixture(scope="module")
def dists():
    return build_distributions()


# ---------------------------------------------------------------- provenance

class TestProvenance:

    def test_checksums(self):
        for dist_id, ok in verify_against_canonical().items():
            assert ok, f"{dist_id} does not regenerate to its stored checksum"

    def test_valid_probability_vectors(self, dists):
        for dist_id in DIST_IDS:
            p = dists[dist_id]
            assert p.shape == (8,)
            assert np.isclose(p.sum(), 1.0, atol=1e-12)
            assert np.all(p > 0), f"{dist_id} has a non-positive entry"

    def test_d5_d6_deterministic(self, dists):
        again = build_distributions()
        assert np.allclose(dists["D5"], again["D5"])
        assert np.allclose(dists["D6"], again["D6"])


# ---------------------------------------------------------------- exactness

class TestExactness:

    @pytest.mark.parametrize("dist_id", DIST_IDS)
    @pytest.mark.parametrize("ladder", ["A", "B"])
    def test_full_stage_is_exact(self, dists, dist_id, ladder):
        p = dists[dist_id]
        assert np.allclose(simulate(p, "FULL", ladder), p, atol=1e-12)

    @pytest.mark.parametrize("dist_id", DIST_IDS)
    def test_ladders_identical_in_ideal_model(self, dists, dist_id):
        p = dists[dist_id]
        assert np.allclose(simulate(p, "FULL", "A"),
                           simulate(p, "FULL", "B"), atol=1e-12)

    @pytest.mark.parametrize("dist_id", DIST_IDS)
    def test_l0_splits_mass(self, dists, dist_id):
        p = dists[dist_id]
        q = simulate(p, "L0", "A")
        assert np.isclose(q[0], p[:4].sum(), atol=1e-12)
        assert np.isclose(q[4], p[4:].sum(), atol=1e-12)
        assert np.isclose(q[[1, 2, 3, 5, 6, 7]].sum(), 0.0, atol=1e-12)

    def test_l01_partial_support(self, dists):
        q = simulate(dists["D1"], "L01", "A")
        assert np.allclose(q, [0.15, 0, 0.35, 0, 0.35, 0, 0.15, 0], atol=1e-12)

    @pytest.mark.parametrize("dist_id", DIST_IDS)
    def test_identity_tail_is_identity(self, dists, dist_id):
        p = dists[dist_id]
        assert np.allclose(simulate(p, "FULL", "A", identity_tail=False),
                           simulate(p, "FULL", "A", identity_tail=True),
                           atol=1e-12)


# -------------------------------------------------------------- compilation

class TestCompilation:

    def test_d1_angles_match_circuit_figure(self, dists):
        ang = gr_angles(dists["D1"])
        assert np.isclose(ang.phi_root, 90.0)
        assert np.isclose(ang.phi_level1["0"] / 2, 56.79, atol=5e-3)
        assert np.isclose(ang.phi_level1["1"] / 2, 33.21, atol=5e-3)
        assert np.allclose(ang.ladder_angles["A"],
                           [90.0, 5.63, 0.0, 13.84], atol=5e-3)
        assert np.allclose(ang.ladder_angles["B"],
                           [90.0, 13.84, 0.0, 5.63], atol=5e-3)

    def test_d1_tree_angles_match_figure(self, dists):
        """
        Theoretical angles printed on the dyadic-tree figure.

        The values asserted here are those of manuscript v03. Four of them were
        wrong in v01/v02: theta_0 (54.74 -> 56.79), theta_1 (35.26 -> 33.21),
        and theta_00 <-> theta_11, which had been exchanged. The compiled
        circuits were correct throughout, so no experimental result changed.
        """
        ang = gr_angles(dists["D1"])
        assert np.isclose(ang.theta_root, 45.00, atol=5e-3)

        theta1 = {k: v / 2 for k, v in ang.phi_level1.items()}
        assert np.isclose(theta1["0"], 56.79, atol=5e-3)
        assert np.isclose(theta1["1"], 33.21, atol=5e-3)

        theta2 = {k: v / 2 for k, v in ang.phi_level2.items()}
        assert np.isclose(theta2["00"], 54.74, atol=5e-3)
        assert np.isclose(theta2["01"], 49.11, atol=5e-3)
        assert np.isclose(theta2["10"], 40.89, atol=5e-3)
        assert np.isclose(theta2["11"], 35.26, atol=5e-3)

        # Complementary-pair relations quoted in the "Scope of D1" paragraph.
        assert np.isclose(theta1["0"] + theta1["1"], 90.0, atol=1e-9)
        assert np.isclose(theta2["00"] + theta2["11"], 90.0, atol=1e-9)
        assert np.isclose(theta2["01"] + theta2["10"], 90.0, atol=1e-9)

    def test_appendix_commanded_angles(self, dists):
        """Commanded angles listed at the end of Appendix A.5 (v03 values)."""
        ang = gr_angles(dists["D1"])
        assert np.isclose(ang.phi_root, 90.00, atol=5e-3)
        assert np.isclose(ang.phi_level1["0"], 113.58, atol=5e-3)
        assert np.isclose(ang.phi_level1["1"], 66.42, atol=5e-3)
        assert np.isclose(ang.phi_level2["00"], 109.47, atol=5e-3)
        assert np.isclose(ang.phi_level2["01"], 98.21, atol=5e-3)
        assert np.isclose(ang.phi_level2["10"], 81.79, atol=5e-3)
        assert np.isclose(ang.phi_level2["11"], 70.53, atol=5e-3)

    @pytest.mark.parametrize("stage,total", [("L0", 3), ("L01", 13),
                                             ("FULL", 21)])
    def test_gate_counts_with_identity_tail(self, stage, total):
        assert gate_counts(stage, "A", identity_tail=True)["total"] == total

    @pytest.mark.parametrize("dist_id", DIST_IDS)
    def test_table5_descriptors(self, dists, dist_id):
        pub = {r["dist_id"]: r
               for r in _read(PUB / "table5_descriptors_published.csv")}
        d = descriptors(dists[dist_id])
        assert abs(d.level1_deviation
                   - float(pub[dist_id]["delta1_deg"])) < 0.06
        assert abs(d.ucry_range
                   - float(pub[dist_id]["ucry_range_deg"])) < 0.06


# ------------------------------------------------------------- paper tables

class TestPaperTables:

    @pytest.mark.parametrize("row", _read(PULSE / "metrics.csv"),
                             ids=lambda r: f"{r['config']}-{r['ladder']}")
    def test_pulse_metrics_recompute(self, dists, row):
        vec = _pulse_distributions()[(row["config"], row["ladder"])]
        target = dists["D1"]
        assert abs(fidelity(target, vec) - float(row["fidelity"])) < 5e-4
        assert abs(tv_distance(target, vec) - float(row["tv"])) < 5e-4
        assert abs(l2_distance(target, vec) - float(row["l2"])) < 5e-4

    @pytest.mark.parametrize("row", _read(PULSE / "ladder_diff.csv"),
                             ids=lambda r: r["config"])
    def test_ladder_difference_recomputes(self, row):
        d = _pulse_distributions()
        a, b = d[(row["config"], "A")], d[(row["config"], "B")]
        assert abs(tv_distance(a, b) - float(row["tv_ab"])) < 5e-4
        assert abs(l2_distance(a, b) - float(row["l2_ab"])) < 5e-4
        assert abs(fidelity(a, b) - float(row["fidelity_ab"])) < 5e-4

    def test_pooled_full_stage_fidelity_range(self):
        rows = _read(PUB / "full_stage_results.csv")
        pooled = {d: np.mean([float(r["fidelity_mean"]) for r in rows
                              if r["dist_id"] == d]) for d in DIST_IDS}
        assert min(pooled.values()) == pytest.approx(0.871, abs=5e-4)
        assert max(pooled.values()) == pytest.approx(0.964, abs=1e-3)
        assert pooled["D0"] == pytest.approx(0.962, abs=1e-3)

    def test_spearman_headline_correlation(self, dists):
        stats = pytest.importorskip("scipy.stats")
        rows = _read(PUB / "full_stage_results.csv")
        pooled = [np.mean([float(r["fidelity_mean"]) for r in rows
                           if r["dist_id"] == d]) for d in DIST_IDS]
        for attr in ("level1_deviation", "ucry_range"):
            x = [getattr(descriptors(dists[d]), attr) for d in DIST_IDS]
            rho, pval = stats.spearmanr(x, pooled)
            assert rho == pytest.approx(-0.64, abs=0.02)
            assert pval == pytest.approx(0.12, abs=0.02)


class TestCompanionBounds:
    """
    Checks against the rigorous results of the companion paper,
    AIMS Mathematics 11(6), 16366-16394 (2026), doi:10.3934/math.2026672.
    """

    SIGMA_PHI_DEG = 2.04     # campaign-level RMS pulse-angle error
    N_QUBITS = 3

    def _pooled_tv(self):
        rows = _read(PUB / "full_stage_results.csv")
        return {d: float(np.mean([float(r["tv_mean"]) for r in rows
                                  if r["dist_id"] == d])) for d in DIST_IDS}

    def test_angle_perturbation_envelope(self):
        """
        TV <= min(1, n*eta) for a uniform per-angle perturbation of eta radians.
        Evaluated at eta = 3*sigma_phi, every pooled FULL-stage TV must fit.
        """
        eta = np.radians(3 * self.SIGMA_PHI_DEG)
        bound = min(1.0, self.N_QUBITS * eta)
        assert bound == pytest.approx(0.320, abs=5e-4)

        pooled = self._pooled_tv()
        for dist_id, tv in pooled.items():
            assert tv <= bound, (
                f"{dist_id} TV = {tv:.3f} exceeds the 3-sigma envelope {bound:.3f}"
            )
        assert max(pooled.values()) == pytest.approx(0.319, abs=1e-3)
        assert min(pooled.values()) == pytest.approx(0.130, abs=1e-3)

    def test_one_sigma_envelope_is_too_tight(self):
        """
        The manuscript states that evaluating the bound at one sigma puts it
        below *every* measured group, the closest being D6-B at 0.108 against a
        bound of 0.107. Checked here so the claim cannot silently rot.
        """
        bound = min(1.0, self.N_QUBITS * np.radians(self.SIGMA_PHI_DEG))
        assert bound == pytest.approx(0.107, abs=5e-4)

        rows = _read(PUB / "full_stage_results.csv")
        per_group = {f"{r['dist_id']}-{r['ladder']}": float(r["tv_mean"])
                     for r in rows}
        assert all(tv > bound for tv in per_group.values()), (
            "some group falls inside the one-sigma envelope"
        )
        closest = min(per_group, key=per_group.get)
        assert closest == "D6-B"
        assert per_group[closest] == pytest.approx(0.108, abs=1e-3)

    def test_shot_count_design_rule(self):
        """S >= 2^(n+1) log(2/delta)/eps^2, inverted at the campaign setting."""
        shots, delta = 4096, 0.05
        eps_ln = np.sqrt(2 ** (self.N_QUBITS + 1) * np.log(2 / delta) / shots)
        eps_log2 = np.sqrt(2 ** (self.N_QUBITS + 1) * np.log2(2 / delta) / shots)
        assert eps_ln == pytest.approx(0.120, abs=1e-3)
        assert eps_log2 == pytest.approx(0.144, abs=1e-3)

        pooled = self._pooled_tv()
        # D5 and D2 sit barely above the natural-log floor; D3 and D4 well above
        assert pooled["D5"] > eps_ln and pooled["D5"] < 1.2 * eps_ln
        assert pooled["D2"] > eps_ln and pooled["D2"] < 1.2 * eps_ln
        assert pooled["D3"] > 2 * eps_ln
        assert pooled["D4"] > 2 * eps_ln


# ------------------------------------------------------------- known issues

class TestKnownIssues:
    """
    Documented inconsistencies between the manuscript and its data.
    Each test XFAILs until the underlying issue is resolved.
    """

    @pytest.mark.xfail(reason="marginal table not reproducible from the "
                              "distribution table; see REVIEW_NOTES_v02.md",
                       strict=False)
    @pytest.mark.parametrize("row", _read(PULSE / "marginals.csv"),
                             ids=lambda r: f"{r['config']}-{r['ladder']}")
    def test_pulse_marginals_recompute(self, row):
        vec = _pulse_distributions()[(row["config"], row["ladder"])]
        pub = np.array([float(row[f"p_q{j}_1"]) for j in range(3)])
        assert np.allclose(marginals(vec), pub, atol=2e-3)

    @pytest.mark.xfail(reason="recomputed Exp.2 marginals give P(q2=1)~0.38",
                       strict=False)
    def test_exp2_marginals_are_balanced(self):
        d = _pulse_distributions()
        for ladder in ("A", "B"):
            m = marginals(d[("Exp2", ladder)])
            assert np.all((m >= 0.48) & (m <= 0.52))

    @pytest.mark.xfail(reason="shipped analysis uses max|phi-90|, which shows "
                              "no association with fidelity",
                       strict=False)
    def test_shipped_covariate_reproduces_headline(self, dists):
        stats = pytest.importorskip("scipy.stats")
        rows = _read(PUB / "full_stage_results.csv")
        pooled = [np.mean([float(r["fidelity_mean"]) for r in rows
                           if r["dist_id"] == d]) for d in DIST_IDS]
        x = [descriptors(dists[d]).max_ucry_dev for d in DIST_IDS]
        rho, _ = stats.spearmanr(x, pooled)
        assert rho == pytest.approx(-0.64, abs=0.02)
