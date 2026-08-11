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

    SHOTS = 4096          # campaign orchestrator default; not recorded per run

    def test_shot_count_design_rule(self):
        """
        S >= 2^(n+1) ln(2/delta)/eps^2 inverted at the campaign setting.

        The companion uses the natural logarithm (confirmed by the authors),
        giving eps = 0.120, and no target falls below it -- the closest is D5
        at 0.130. The base-two value is asserted too, only to record that the
        convention is not neutral: under base two D5 and D2 would fall below.
        """
        delta = 0.05
        eps_ln = np.sqrt(2 ** (self.N_QUBITS + 1) * np.log(2 / delta) / self.SHOTS)
        eps_log2 = np.sqrt(2 ** (self.N_QUBITS + 1) * np.log2(2 / delta) / self.SHOTS)
        assert eps_ln == pytest.approx(0.120, abs=1e-3)
        assert eps_log2 == pytest.approx(0.144, abs=1e-3)

        pooled = self._pooled_tv()
        assert [d for d in DIST_IDS if pooled[d] < eps_ln] == []
        assert min(pooled.values()) == pytest.approx(0.130, abs=1e-3)
        assert sorted(d for d in DIST_IDS if pooled[d] < eps_log2) == ["D2", "D5"]

    def test_realised_sampling_error_is_an_order_of_magnitude_smaller(self):
        """
        E[TV] ~ (1/2) sum_k sqrt(2 p_k(1-p_k)/(pi S)) for a multinomial sample.
        Every measured FULL-stage TV must exceed its own sampling scale by a
        wide margin, which is what licenses treating the discrepancies as real.
        """
        dists = build_distributions()
        pooled = self._pooled_tv()
        ratios = {}
        for d in DIST_IDS:
            p = dists[d]
            e = 0.5 * float(np.sum(np.sqrt(2 * p * (1 - p) / (np.pi * self.SHOTS))))
            assert 0.012 < e < 0.017, (d, e)
            ratios[d] = pooled[d] / e
        assert min(ratios.values()) == pytest.approx(9.2, abs=0.2)
        assert max(ratios.values()) == pytest.approx(25.1, abs=0.3)
        assert min(ratios, key=ratios.get) == "D0"
        assert max(ratios, key=ratios.get) == "D3"


class TestStructuralClaim:
    """
    The manuscript's structural claim and the controls it now reports.
    Added after the referee-style read; these numbers appear in the abstract,
    Section 6 and the Conclusion, so they must not drift.
    """

    UNIFORM = np.full(8, 0.125)

    def _pooled_fidelity(self):
        rows = _read(PUB / "full_stage_results.csv")
        return np.array([np.mean([float(r["fidelity_mean"]) for r in rows
                                  if r["dist_id"] == d]) for d in DIST_IDS])

    def _pooled_tv(self):
        rows = _read(PUB / "full_stage_results.csv")
        return np.array([np.mean([float(r["tv_mean"]) for r in rows
                                  if r["dist_id"] == d]) for d in DIST_IDS])

    def test_descriptors_are_collinear(self, dists):
        """rho(Delta_1, Delta_UCRy) = 0.93 -- they are one feature, not two."""
        stats = pytest.importorskip("scipy.stats")
        d1 = [descriptors(dists[d]).level1_deviation for d in DIST_IDS]
        uc = [descriptors(dists[d]).ucry_range for d in DIST_IDS]
        assert stats.spearmanr(d1, uc)[0] == pytest.approx(0.93, abs=0.01)

    def test_depolarising_prediction_has_no_purchase(self, dists):
        """
        Section 3 predicts TV_FULL ~ p_d * TV(p*,u). Rank correlation against
        the data is ~0, which is what lets the structural reading stand.
        """
        stats = pytest.importorskip("scipy.stats")
        tvu = [tv_distance(dists[d], self.UNIFORM) for d in DIST_IDS]
        rho, p = stats.spearmanr(tvu, self._pooled_tv())
        assert rho == pytest.approx(0.07, abs=0.02)
        assert p > 0.8

    def test_confounder_is_real(self, dists):
        """Distance from uniform does correlate with tree asymmetry."""
        stats = pytest.importorskip("scipy.stats")
        tvu = [tv_distance(dists[d], self.UNIFORM) for d in DIST_IDS]
        d1 = [descriptors(dists[d]).level1_deviation for d in DIST_IDS]
        assert stats.spearmanr(d1, tvu)[0] == pytest.approx(0.82, abs=0.02)

    def test_matched_cluster(self, dists):
        """
        D2, D4, D5, D6 sit within 0.014 of one another in TV(p*,u), yet span
        0.882-0.964 in fidelity, still ordered by Delta_1 at rho = -0.80.
        This is the paper's control for the confounder above.
        """
        stats = pytest.importorskip("scipy.stats")
        fid = dict(zip(DIST_IDS, self._pooled_fidelity()))
        tvu = {d: tv_distance(dists[d], self.UNIFORM) for d in DIST_IDS}

        cluster = [d for d in DIST_IDS if 0.36 <= tvu[d] <= 0.39]
        assert cluster == ["D2", "D4", "D5", "D6"]
        spread = max(tvu[d] for d in cluster) - min(tvu[d] for d in cluster)
        assert spread == pytest.approx(0.014, abs=1e-3)

        y = [fid[d] for d in cluster]
        assert min(y) == pytest.approx(0.882, abs=1e-3)
        assert max(y) == pytest.approx(0.964, abs=1e-3)

        d1 = [descriptors(dists[d]).level1_deviation for d in cluster]
        uc = [descriptors(dists[d]).ucry_range for d in cluster]
        assert stats.spearmanr(d1, y)[0] == pytest.approx(-0.80, abs=0.01)
        assert stats.spearmanr(uc, y)[0] == pytest.approx(-0.60, abs=0.01)

    def test_claim_does_not_survive_without_d2_and_d3(self, dists):
        """
        How much of the structural claim rests on the two sessions of 31 March.

        Those two are the ones absent from the device log, and there is an open
        question (Q8c) over whether they were acquired on the second Triangulum
        rather than the one that logged the other five. This test measures what
        would be left if they had to be dropped -- not because we expect to drop
        them, but so that the answer is on record and cannot drift.

        The answer is: not enough. D3 anchors the high-Delta_1 end and D2 the
        low-Delta_1 end of the matched cluster, so removing them costs both
        extremes at once.
        """
        stats = pytest.importorskip("scipy.stats")
        fid = dict(zip(DIST_IDS, self._pooled_fidelity()))
        d1 = {d: descriptors(dists[d]).level1_deviation for d in DIST_IDS}

        rho = lambda ks: stats.spearmanr([d1[k] for k in ks],
                                         [fid[k] for k in ks])
        r7, p7 = rho(list(DIST_IDS))
        assert r7 == pytest.approx(-0.643, abs=0.01) and p7 > 0.05

        five = [d for d in DIST_IDS if d not in ("D2", "D3")]
        r5, p5 = rho(five)
        assert r5 == pytest.approx(-0.500, abs=0.01)
        assert p5 > 0.35              # nothing survives on five points

        # the matched cluster loses its most informative member
        assert rho(["D2", "D4", "D5", "D6"])[0] == pytest.approx(-0.80, abs=0.01)
        assert rho(["D4", "D5", "D6"])[0] == pytest.approx(-1.0, abs=1e-9)

    def test_d0_ladders_command_identical_angles(self, dists):
        """
        The uniform target separates the two ladders at p = 0.0002 even though
        both command exactly the same rotation angles -- the paper's evidence
        that part of the ordering effect is pure gate scheduling.
        """
        ang = gr_angles(dists["D0"])
        a, b = ang.ladder_angles["A"], ang.ladder_angles["B"]
        assert np.allclose(a, b, atol=1e-12)
        assert np.allclose(a, [90.0, 0.0, 0.0, 0.0], atol=1e-9)

        rows = {r["dist_id"]: r for r in _read(PUB / "full_stage_results.csv")}
        assert float(rows["D0"]["p_mann_whitney"]) == pytest.approx(2e-4, abs=1e-5)

    def test_d3_ladders_command_different_angles(self, dists):
        """By contrast, an asymmetric target does differ in angle content."""
        ang = gr_angles(dists["D3"])
        a, b = ang.ladder_angles["A"], ang.ladder_angles["B"]
        assert not np.allclose(a, b, atol=1e-6)
        assert np.allclose(sorted(np.round(a, 6)), sorted(np.round(b, 6)))
        assert np.allclose(a, [90.0, -19.76, 0.0, 31.297], atol=5e-3)


class TestStagedProtocol:
    """
    The stage x distribution table and what the manuscript reads off it.
    Derived from `runs_flat_v2.csv`; the summary is vendored in
    `data/campaign_v2_published/stage_by_dist.csv`.
    """

    def _table(self):
        out = {}
        for r in _read(PUB / "stage_by_dist.csv"):
            out[(r["dist_id"], r["stage"])] = float(r["fidelity_mean"])
        return out

    def test_monotone_for_every_distribution(self):
        """L0 < L01 < FULL, without exception. Asserted in the limitations."""
        t = self._table()
        for d in DIST_IDS:
            l0, l01, full = t[(d, "L0")], t[(d, "L01")], t[(d, "FULL")]
            assert l0 < l01 < full, f"{d}: {l0} {l01} {full}"

    def test_l0_does_not_predict_full(self):
        """
        rho(L0, FULL) ~ 0 across the suite -- the paper's evidence that the
        staged protocol separates error regimes rather than tracking depth.
        """
        stats = pytest.importorskip("scipy.stats")
        t = self._table()
        l0 = [t[(d, "L0")] for d in DIST_IDS]
        l01 = [t[(d, "L01")] for d in DIST_IDS]
        full = [t[(d, "FULL")] for d in DIST_IDS]

        rho, p = stats.spearmanr(l0, full)
        assert rho == pytest.approx(-0.04, abs=0.02)
        assert p > 0.9
        assert stats.spearmanr(l01, full)[0] == pytest.approx(0.86, abs=0.02)

    def test_d6_reversal(self):
        """D6 is worst at L0 and best at FULL; D2 is best at L0."""
        t = self._table()
        l0 = {d: t[(d, "L0")] for d in DIST_IDS}
        full = {d: t[(d, "FULL")] for d in DIST_IDS}
        assert min(l0, key=l0.get) == "D6" and l0["D6"] == pytest.approx(0.400, abs=1e-3)
        assert max(full, key=full.get) == "D6" and full["D6"] == pytest.approx(0.964, abs=1e-3)
        assert max(l0, key=l0.get) == "D2"

    def test_d1_slice_matches_the_published_stage_table(self):
        """The D1 column reproduces Table `stage_overall` of the manuscript."""
        t = self._table()
        assert t[("D1", "L0")] == pytest.approx(0.486, abs=1e-3)
        assert t[("D1", "L01")] == pytest.approx(0.772, abs=1e-3)
        assert t[("D1", "FULL")] == pytest.approx(0.909, abs=1e-3)

    def test_d1_marginals_are_now_consistent(self):
        """
        The corrected D1 FULL marginals. These were wrong in v01-v03 and were
        replaced with the values computed from the run-level table; they must
        equal the marginals of the mean distributions quoted in the same
        subsection.
        """
        published = {r["ladder"]: np.array([float(r[f"p_q{j}_1"]) for j in range(3)])
                     for r in _read(PUB / "d1_marginals.csv")}
        mean_dist = {
            "A": np.array([0.104, 0.170, 0.120, 0.082, 0.154, 0.080, 0.172, 0.118]),
            "B": np.array([0.093, 0.150, 0.102, 0.089, 0.151, 0.091, 0.178, 0.146]),
        }
        for ladder, vec in mean_dist.items():
            assert np.allclose(marginals(vec), published[ladder], atol=2e-3), ladder


class TestExecutionOrder:
    """
    What the hardware actually did, from the timestamps in the run record.

    These tests exist because an earlier draft asserted the opposite. The
    scheduler in the campaign repository *can* interleave all seven
    distributions, and the manuscript briefly claimed it had; the record shows
    the campaign was run one distribution per session via `--dist`. Anything
    that depends on execution order must be checked here, not against the
    scheduler source.
    """

    RUNS = DATA / "campaign_v2_runs" / "campaign_v2_runs_clean.jsonl"

    def _records(self):
        import json
        from datetime import datetime
        rec = []
        with open(self.RUNS, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    r = json.loads(line)
                    r["_t"] = datetime.fromisoformat(
                        str(r["created"]).replace("Z", "+00:00")
                    ).replace(tzinfo=None)
                    rec.append(r)
        rec.sort(key=lambda r: r["_t"])
        return rec

    def test_seven_contiguous_sessions(self):
        rec = self._records()
        assert len(rec) == 700
        order, seen = [], set()
        for r in rec:
            if r["dist_id"] not in seen:
                seen.add(r["dist_id"]); order.append(r["dist_id"])
        assert order == ["D3", "D2", "D5", "D6", "D4", "D1", "D0"]
        for d in order:
            idx = [i for i, r in enumerate(rec) if r["dist_id"] == d]
            assert idx == list(range(idx[0], idx[0] + 100)), d

    def test_stages_are_balanced_within_every_session(self):
        """The one part of the drift argument the design does support."""
        import statistics as st
        rec = self._records()
        for d in DIST_IDS:
            s = [r for r in rec if r["dist_id"] == d]
            m = {k: st.mean([i for i, r in enumerate(s) if r["stage"] == k])
                 for k in ("L0", "L01", "FULL")}
            assert m["L0"] == pytest.approx(42.0, abs=0.05)
            assert m["L01"] == pytest.approx(47.0, abs=0.05)
            assert m["FULL"] == pytest.approx(54.5, abs=0.05)

        pooled = {k: st.mean([i for i, r in enumerate(rec) if r["stage"] == k])
                  for k in ("L0", "L01", "FULL")}
        assert pooled["L0"] == pytest.approx(342.0, abs=0.05)
        assert pooled["L01"] == pytest.approx(347.0, abs=0.05)
        assert pooled["FULL"] == pytest.approx(354.5, abs=0.05)

    def test_no_systematic_session_drift(self):
        """
        Distribution is confounded with acquisition time, so the confound is
        measured. rho = +0.32, p = 0.48: no degradation, and the worst target
        ran first.
        """
        import statistics as st
        stats = pytest.importorskip("scipy.stats")
        rec = self._records()
        order, seen = [], set()
        for r in rec:
            if r["dist_id"] not in seen:
                seen.add(r["dist_id"]); order.append(r["dist_id"])
        f = [st.mean(float(r["fidelity_vs_target"]) for r in rec
                     if r["dist_id"] == d and r["stage"] == "FULL")
             for d in order]
        rho, p = stats.spearmanr(range(1, 8), f)
        assert rho == pytest.approx(0.321, abs=0.01)
        assert p > 0.4
        assert order[0] == "D3" and f[0] == pytest.approx(0.871, abs=1e-3)
        assert order[-1] == "D0" and f[-1] == pytest.approx(0.962, abs=1e-3)


# ------------------------------------------------------- the device log

class TestDeviceLog:
    """
    The instrument's own record, cross-checked against ours.

    `data/device_log/` is an extract of the SpinQuasar job database. It was not
    produced by our pipeline, so where it overlaps the run record the two are an
    independent check on each other. These tests pin that agreement, and pin the
    two facts about the extract that are easy to get wrong: `job_name` is not
    unique, and the export is missing a day.
    """

    LOG = DATA / "device_log"

    @staticmethod
    @pytest.fixture(scope="class")
    def log():
        import csv
        with open(TestDeviceLog.LOG / "device_log_paper.csv",
                  encoding="utf-8") as fh:
            return list(csv.DictReader(fh))

    @staticmethod
    @pytest.fixture(scope="class")
    def runs():
        import json
        out = {}
        with open(DATA / "campaign_v2_runs" / "campaign_v2_runs_clean.jsonl",
                  encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    r = json.loads(line)
                    out[r["run_name"]] = r
        return out

    KEYS = ("000", "001", "010", "011", "100", "101", "110", "111")

    def test_experiment_id_is_the_only_safe_key(self, log):
        assert len(log) == 986
        assert len({r["experiment_id"] for r in log}) == 986
        # job names were reused across sessions; a name-keyed join collides
        assert len({r["job_name"] for r in log}) == 760

    def test_measured_distributions_are_identical_to_the_run_record(
            self, log, runs):
        c2 = {r["job_name"][3:]: r for r in log if r["group"] == "campaign_v2"}
        assert len(c2) == 498 and set(c2) <= set(runs)
        worst = 0.0
        for name, row in c2.items():
            a = np.array([float(row[f"exp_p{k}"]) for k in self.KEYS])
            b = np.array([runs[name]["exp_probs"][k] for k in self.KEYS])
            worst = max(worst, float(np.abs(a - b).max()))
        assert worst < 1e-12

    def test_labels_and_clock_offset_agree(self, log, runs):
        from datetime import datetime
        c2 = {r["job_name"][3:]: r for r in log if r["group"] == "campaign_v2"}
        for name, row in c2.items():
            rec = runs[name]
            assert (row["dist_id"], row["stage"], row["ladder"]) == \
                   (rec["dist_id"], rec["stage"], rec["ladder"])
            dt = (datetime.fromisoformat(row["created"])
                  - datetime.fromisoformat(rec["created"]).replace(tzinfo=None))
            assert dt.total_seconds() == pytest.approx(7200, abs=30)

    def test_the_export_is_missing_31_march(self, log, runs):
        c2 = {r["job_name"][3:] for r in log if r["group"] == "campaign_v2"}
        missing = sorted(set(runs) - c2)
        assert len(missing) == 202
        assert sum(m.startswith(("D2", "D3")) for m in missing) == 200
        assert [m for m in missing if not m.startswith(("D2", "D3"))] == \
               ["D4_FULL_A_004", "D5_FULL_B_017"]
        assert "2026-03-31" not in {r["created"][:10] for r in log}

    def test_d2_and_d3_are_absent_under_every_job_name(self, log, dists):
        """The angle fingerprint that settled the relabelling question."""
        import json
        from grtri.angles import gr_angles

        def fingerprint(d_id):
            a = gr_angles(dists[d_id])
            s = {round(abs(a.phi_root), 2)}
            s |= {round(abs(v), 2) for v in a.phi_level1.values()}
            for lad in ("A", "B"):
                s |= {round(abs(float(x)), 2) for x in a.ladder_angles[lad]}
            return s - {0.0}

        circuits = []
        with open(self.LOG / "device_log_circuits.jsonl", encoding="utf-8") as fh:
            for line in fh:
                c = json.loads(line)
                circuits.append({round(abs(g.get("angle", 0)), 2)
                                 for g in c["gates"]} - {0.0})
        hits = lambda d: sum(1 for a in circuits
                             if len(fingerprint(d) & a) >= 2)
        # the controls: these targets are recovered by their angles alone
        for d in ("D1", "D4", "D5", "D6"):
            assert hits(d) >= 49, d
        # the question: not present, at any threshold, under any name
        assert hits("D2") == 0
        assert hits("D3") == 0

    def test_density_matrix_diagonals_are_the_populations(self, log):
        import json
        by_id = {r["experiment_id"]: r for r in log}
        worst, seen = 0.0, 0
        with open(self.LOG / "device_log_states.jsonl", encoding="utf-8") as fh:
            for line in fh:
                s = json.loads(line)
                row = by_id.get(str(s["experiment_id"]))
                R = np.array(s["real"], float)
                if row is None or R.size != 64 or row["exp_p000"] == "":
                    continue
                p = np.array([float(row[f"exp_p{k}"]) for k in self.KEYS])
                worst = max(worst,
                            float(np.abs(np.diag(R.reshape(8, 8)) - p).max()))
                seen += 1
        assert seen == 985
        assert worst < 1e-9


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
