#!/usr/bin/env python3
"""Append the Section 8 angle-noise diagnostics to 90_make_tables.py."""
from pathlib import Path
import sys

p = Path(sys.argv[1] if len(sys.argv) > 1 else "scripts/90_make_tables.py")
s = p.read_text()
assert "def macros_angle" not in s, "already patched"

NEW = r'''
def macros_angle(camp, out, macros):
    """The one commanded rotation whose realised angle can be read off directly.

    At L0 the circuit applies a single Ry to q0, so inverting
    P(q0=1) = sin^2(phi/2) run by run gives the realised angle against the
    commanded one. Reported raw, consistently with the rest of the paper.
    """
    l0 = [r for r in camp if r["stage"] == "L0"]
    by = {}
    for r in l0:
        ps = min(max(1 - float(r["sim_q0_p0"]), 0.0), 1.0)
        pe = min(max(1 - float(r["exp_q0_p0"]), 0.0), 1.0)
        d = (2 * np.degrees(np.arcsin(np.sqrt(pe)))
             - 2 * np.degrees(np.arcsin(np.sqrt(ps))))
        by.setdefault(r["dist_id"], []).append(d)
    groups = [np.array(by[d]) for d in DISTS]
    allv = np.concatenate(groups)
    rms = float(np.sqrt((allv ** 2).mean()))
    rng = np.random.default_rng(0)
    bs = [float(np.sqrt((rng.choice(allv, allv.size) ** 2).mean())) for _ in range(2000)]
    means = np.array([g.mean() for g in groups])
    within = float(np.sqrt(np.mean([g.var(ddof=1) for g in groups])))
    macros["AngleRms"] = f"{rms:.2f}"
    macros["AngleRmsLo"] = f"{np.percentile(bs, 2.5):.2f}"
    macros["AngleRmsHi"] = f"{np.percentile(bs, 97.5):.2f}"
    macros["AngleWithin"] = f"{within:.2f}"
    macros["AngleBetween"] = f"{means.std(ddof=1):.2f}"
    macros["AnglePredicted"] = f"{within / np.sqrt(len(groups[0])):.2f}"
    macros["AngleKW"] = sci(float(stats.kruskal(*groups).pvalue))
    macros["AngleMeanLo"] = f"{means.min():+.2f}"
    macros["AngleMeanHi"] = f"{means.max():+.2f}"


'''

s = s.replace("\ndef main():", NEW + "\ndef main():")
s = s.replace("    tab_targets(out, macros)",
              "    tab_targets(out, macros)\n    macros_angle(camp, out, macros)")
p.write_text(s)
print("patched", p)
