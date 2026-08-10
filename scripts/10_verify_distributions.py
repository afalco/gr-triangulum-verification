#!/usr/bin/env python3
"""
10_verify_distributions.py
==========================
Verifies the benchmark suite D0--D6 and regenerates Table 5 of the manuscript
("Extended campaign distribution suite") from the distributions themselves.

Checks
------
1. The suite regenerates bit-for-bit from its definition (SHA-256 checksums).
2. Every vector is a valid probability vector.
3. Entropy and contrast match the values stored in the canonical JSON.
4. The structural descriptors Delta_1 and Delta_UCRy reproduce the published
   Table 5 to the precision at which it is printed.

Usage
-----
    python scripts/10_verify_distributions.py
"""

from __future__ import annotations

import csv

import numpy as np

from _common import DATA, Report

from grtri import (
    DIST_IDS,
    DIST_TYPES,
    build_distributions,
    descriptors,
    load_canonical,
    verify_against_canonical,
)

TOL_PRINTED = 0.06   # Table 5 is printed to one decimal place


def main() -> int:
    rep = Report("Benchmark suite D0--D6: provenance and Table 5 regeneration")
    dists = build_distributions()
    canonical = load_canonical()

    # ---------------------------------------------------------------- 1
    rep.section("Checksum provenance (regeneration is bit-for-bit)")
    for dist_id, ok in verify_against_canonical().items():
        rep.check(ok, f"{dist_id} SHA-256 matches canonical JSON")

    # ---------------------------------------------------------------- 2
    rep.section("Probability-vector validity")
    for dist_id in DIST_IDS:
        p = dists[dist_id]
        rep.check(
            p.shape == (8,) and np.isclose(p.sum(), 1.0, atol=1e-12)
            and np.all(p > 0),
            f"{dist_id} is a strictly positive length-8 probability vector",
            f"sum={p.sum():.15f}",
        )

    # ---------------------------------------------------------------- 3
    rep.section("Entropy and contrast against the canonical JSON")
    for dist_id in DIST_IDS:
        d = descriptors(dists[dist_id])
        stored = canonical[dist_id]
        rep.check(
            np.isclose(d.shannon_entropy_bits, stored["shannon_entropy_bits"],
                       atol=1e-9)
            and np.isclose(d.contrast, stored["contrast"], atol=1e-9),
            f"{dist_id} entropy and contrast agree",
            f"H={d.shannon_entropy_bits:.4f} C={d.contrast:.2f}",
        )

    # ---------------------------------------------------------------- 4
    rep.section("Table 5 regeneration (Delta_1 and Delta_UCRy)")
    published = {}
    path = DATA / "campaign_v2_published" / "table5_descriptors_published.csv"
    with open(path, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            published[row["dist_id"]] = row

    print(f"\n  {'ID':<4}{'type':<18}{'H_S':>7}{'contrast':>10}"
          f"{'D1':>8}{'D1*':>8}{'UCRy':>9}{'UCRy*':>9}")
    print("  " + "-" * 74)
    for dist_id in DIST_IDS:
        d = descriptors(dists[dist_id])
        pub = published[dist_id]
        print(f"  {dist_id:<4}{DIST_TYPES[dist_id]:<18}"
              f"{d.shannon_entropy_bits:>7.2f}{d.contrast:>10.1f}"
              f"{d.level1_deviation:>8.1f}{float(pub['delta1_deg']):>8.1f}"
              f"{d.ucry_range:>9.1f}{float(pub['ucry_range_deg']):>9.1f}")
    print("  (columns marked * are the published Table 5 values)\n")

    for dist_id in DIST_IDS:
        d = descriptors(dists[dist_id])
        pub = published[dist_id]
        rep.check(
            abs(d.level1_deviation - float(pub["delta1_deg"])) < TOL_PRINTED,
            f"{dist_id} Delta_1 reproduces Table 5",
            f"computed={d.level1_deviation:.2f} published={pub['delta1_deg']}",
        )
        rep.check(
            abs(d.ucry_range - float(pub["ucry_range_deg"])) < TOL_PRINTED,
            f"{dist_id} Delta_UCRy reproduces Table 5",
            f"computed={d.ucry_range:.2f} published={pub['ucry_range_deg']}",
        )

    # ---------------------------------------------------------------- note
    rep.section("Descriptor definition mismatch with the shipped analysis")
    rep.info("Table 5 uses the UCRy *range* max(phi_2) - min(phi_2);")
    rep.info("05_characterisation_analysis.py uses max|phi_2 - 90| instead.")
    print()
    for dist_id in DIST_IDS:
        d = descriptors(dists[dist_id])
        print(f"    {dist_id}: ucry_range={d.ucry_range:7.2f}   "
              f"max_ucry_dev={d.max_ucry_dev:7.2f}")
    rep.warn("The regression covariate in the shipped script is NOT the "
             "descriptor reported in Table 5")

    return rep.finish(exit_on_failure=False)


if __name__ == "__main__":
    raise SystemExit(main())
