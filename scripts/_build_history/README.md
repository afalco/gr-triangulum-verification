# `_build_history/` — one-shot patch scripts

These three scripts were used, in this order, to extend
`scripts/90_make_tables.py` as the v06 manuscript was written. **They have
already been applied.** Running one again will abort on its own assertion
(`already patched`) or, for the first one, insert a duplicate — do not re-run
them. They are kept only as a record of how the generator grew.

| script | added to `90_make_tables.py` | used by |
|---|---|---|
| `patch_sec2_tables.py` | `tab_sessions`, `tab_spam`, `macros_metrics`, `macros_readout` | §2.3, §2.4, §2.5, §2.6 |
| `patch_targets.py` | `tab_targets` and the exactness macros | §1.3, §1.4 |
| `patch_angle.py` | `macros_angle` | §8.2 |

The generator is the artefact that matters, not these. It is deterministic:
regenerating into an empty directory and diffing against the manuscript's
`tables/` reproduces every file byte for byte.

    python scripts/90_make_tables.py --outdir /tmp/tbl_check
    diff -rq /tmp/tbl_check ../<overleaf folder>/tables

Earlier additions to the generator — the Benjamini–Yekutieli column, the suite
correlator table, the correlator intervention table, the density-matrix audit
and the three-compilation table of §5 — were edited in place rather than through
a patch script and have no entry here.
