#!/usr/bin/env python3
"""Append the Section 1 targets table + exactness macros to 90_make_tables.py."""
from pathlib import Path
import sys

p = Path(sys.argv[1] if len(sys.argv) > 1 else "scripts/90_make_tables.py")
s = p.read_text()
assert "def tab_targets" not in s, "already patched"

NEW = r'''
def tab_targets(out, macros):
    """The seven benchmark laws, and the exactness of their compiled circuits.

    Uses the deposited grtri package rather than the run records: the `sim_`
    vectors in those records are the vendor SDK's sampled output, whereas the
    claim here is about the compiled circuit evaluated analytically.
    """
    try:
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
        import grtri
    except Exception as exc:                                       # pragma: no cover
        print(f"  [skip] targets: grtri unavailable ({exc})")
        return
    laws = grtri.build_distributions()
    u = np.ones(8) / 8
    L = ["\\begin{tabular}{llccc}", "\\toprule",
         "Target & Law & $H$ (bits) & $p_{\\max}/p_{\\min}$ & $\\TV$ to uniform \\\\",
         "\\midrule"]
    worst_exact, worst_ab = 0.0, 0.0
    for d in DISTS:
        pv = np.asarray(laws[d], float)
        H = float(-sum(x * np.log2(x) for x in pv if x > 0))
        ratio = float(pv.max() / pv.min()) if pv.min() > 0 else float("inf")
        law = ",\\,".join(("%.3f" % x).rstrip("0").rstrip(".") for x in pv)
        L.append(f"{d} & $({law})$ & ${H:.2f}$ & ${ratio:.0f}$ & ${tv(pv, u):.3f}$ \\\\")
        qa = np.asarray(grtri.simulate(pv, "FULL", "A"), float)
        qb = np.asarray(grtri.simulate(pv, "FULL", "B"), float)
        worst_exact = max(worst_exact,
                          float(np.abs(qa - pv).max()), float(np.abs(qb - pv).max()))
        worst_ab = max(worst_ab, float(np.abs(qa - qb).max()))
    L += ["\\bottomrule", "\\end{tabular}"]
    (out / "targets.tex").write_text("\n".join(L) + "\n")
    macros["ExactMax"] = sci(worst_exact)
    macros["ExactAB"] = sci(worst_ab)
    macros["ExactNcirc"] = str(2 * len(DISTS))
    for stage, key in (("L0", "Lzero"), ("L01", "Lone"), ("FULL", "Full")):
        gc = grtri.gate_counts(stage, "A", True)
        macros[f"Gates{key}"] = str(gc["total"])
        macros[f"Gates{key}NoTail"] = str(grtri.gate_counts(stage, "A", False)["total"])


'''

s = s.replace("\ndef main():", NEW + "\ndef main():")
s = s.replace("    macros_readout(a.pilot, out, macros)",
              "    macros_readout(a.pilot, out, macros)\n    tab_targets(out, macros)")
p.write_text(s)
print("patched", p)
