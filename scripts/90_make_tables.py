#!/usr/bin/env python3
"""
90_make_tables.py -- regenerate every numerical table of the manuscript from the
deposited run-level data. Single source of truth: no figure in the paper is
transcribed by hand.

    python scripts/90_make_tables.py --outdir ../tables

Inputs (paths relative to the repository root):
    data/campaign_v2_runs/runs_flat_v2.csv          700 runs, Mar-Apr 2026, CEU unit
    <gr-repo>/data/ladder_ab/D1_ladder_ab_runs.csv  100 runs, Oct 2026, CEU unit

Emits one .tex fragment per table, each a bare tabular to be \input into a
float, plus a macros file holding every number quoted in running text.
"""
from __future__ import annotations
import argparse, csv, json, os, sys
from pathlib import Path
import numpy as np
from scipy import stats

S = [format(i, "03b") for i in range(8)]
DISTS = ["D0", "D1", "D2", "D3", "D4", "D5", "D6"]


# --------------------------------------------------------------------- utils
def tv(p, q):  return 0.5 * float(np.sum(np.abs(np.asarray(p) - np.asarray(q))))
def l2(p, q):  return float(np.sqrt(np.sum((np.asarray(p) - np.asarray(q)) ** 2)))
def fid(p, q): return float(np.sum(np.sqrt(np.maximum(p, 0) * np.maximum(q, 0))) ** 2)

def vec(rows, pre):
    return np.array([[float(r[f"{pre}{s}"]) for s in S] for r in rows])

def mean_law(rows, pre="exp_"):
    return vec(rows, pre).mean(axis=0)

def ci95(v):
    v = np.asarray(v, float); m = v.mean()
    if len(v) < 2: return m, m, m
    se = v.std(ddof=1) / np.sqrt(len(v)); t = stats.t.ppf(0.975, len(v) - 1)
    return m, m - t * se, m + t * se

def ms(v, d=3):
    v = np.asarray(v, float)
    return f"${v.mean():.{d}f} \\pm {v.std(ddof=1):.{d}f}$"

def pfmt(p):
    if p < 1e-4: return "$<10^{-4}$"
    if p < 0.01: return f"${p:.4f}$"
    return f"${p:.3f}$"

def stars(p):
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "\\,ns"


def sci(x, d=1):
    """LaTeX-ready scientific notation, no dollar signs."""
    if x == 0: return "0"
    import math
    e = int(math.floor(math.log10(abs(x))))
    m = x / 10 ** e
    return f"{m:.{d}f}\\times10^{{{e}}}"


def mathp(p):
    """LaTeX-ready p-value, no dollar signs."""
    return f"{p:.3f}" if p >= 1e-3 else sci(p)


# ------------------------------------------------------------------ campaign
def load(path):
    rows = list(csv.DictReader(open(path)))
    if not rows: sys.exit(f"empty: {path}")
    return rows


def tab_stage_decomposition(camp, out):
    """Implementation error E_s against the stage's own ideal, separated from
    the distance A_s of that stage's ideal to the final target."""
    L = ["\\begin{tabular}{llccc}", "\\toprule",
         "Target & Stage & $E_s$ (implementation) & $A_s$ (incompleteness) "
         "& $\\TV$ vs final target \\\\", "\\midrule"]
    for d in DISTS:
        full = [r for r in camp if r["dist_id"] == d and r["stage"] == "FULL" and r["ladder"] == "A"]
        tgt = mean_law(full, "sim_")
        for k, st in enumerate(("L0", "L01", "FULL")):
            rs = [r for r in camp if r["dist_id"] == d and r["stage"] == st and r["ladder"] == "A"]
            E = np.array([float(r["tv_sim"]) for r in rs])
            A = tv(mean_law(rs, "sim_"), tgt)
            T = np.array([float(r["tv_tgt"]) for r in rs])
            L.append(f"{d if k == 0 else '':2} & {st} & {ms(E)} & ${A:.3f}$ & {ms(T)} \\\\")
        L.append("\\addlinespace")
    L += ["\\bottomrule", "\\end{tabular}"]
    (out / "stage_decomposition.tex").write_text("\n".join(L) + "\n")


def tab_full_ab(camp, out, macros):
    """FULL-stage A/B contrast per target, with an explicit multiplicity rule."""
    res = []
    for d in DISTS:
        a = [float(r["fid_tgt"]) for r in camp if r["dist_id"] == d and r["stage"] == "FULL" and r["ladder"] == "A"]
        b = [float(r["fid_tgt"]) for r in camp if r["dist_id"] == d and r["stage"] == "FULL" and r["ladder"] == "B"]
        res.append((d, np.array(a), np.array(b), stats.mannwhitneyu(a, b, alternative="two-sided")[1]))
    p = np.array([r[3] for r in res]); order = np.argsort(p)
    holm = np.empty(7); run = 0.0
    for i, idx in enumerate(order):
        run = max(run, (7 - i) * p[idx]); holm[idx] = min(run, 1.0)
    bh = np.empty(7); prev = 1.0
    for i, idx in enumerate(order[::-1]):
        prev = min(prev, p[idx] * 7 / (7 - i)); bh[idx] = min(prev, 1.0)

    c_m = sum(1.0 / i for i in range(1, 8))
    by = np.minimum(bh * c_m, 1.0)
    L = ["\\begin{tabular}{llcccccc}", "\\toprule",
         "Target & Ladder & Fidelity & $\\TV$ & $p_{\\mathrm{raw}}$ & $p_{\\mathrm{Holm}}$ "
         "& $p_{\\mathrm{BH}}$ & $p_{\\mathrm{BY}}$ \\\\", "\\midrule"]
    for i, (d, a, b, pv) in enumerate(res):
        ta = np.array([float(r["tv_tgt"]) for r in camp if r["dist_id"] == d and r["stage"] == "FULL" and r["ladder"] == "A"])
        tb = np.array([float(r["tv_tgt"]) for r in camp if r["dist_id"] == d and r["stage"] == "FULL" and r["ladder"] == "B"])
        L.append(f"{d} & A & {ms(a)} & {ms(ta)} & \\multirow{{2}}{{*}}{{{pfmt(pv)}{stars(pv)}}} "
                 f"& \\multirow{{2}}{{*}}{{{pfmt(holm[i])}}} & \\multirow{{2}}{{*}}{{{pfmt(bh[i])}}}"
                 f"& \\multirow{{2}}{{*}}{{{pfmt(by[i])}}} \\\\")
        L.append(f"   & B & {ms(b)} & {ms(tb)} & & & & \\\\")
        L.append("\\addlinespace")
    L += ["\\bottomrule", "\\end{tabular}"]
    (out / "full_ab.tex").write_text("\n".join(L) + "\n")
    macros["NsigRaw"] = int((p < 0.05).sum())
    macros["NsigHolm"] = int((holm < 0.05).sum())
    macros["NsigBH"] = int((bh < 0.05).sum())
    macros["SigHolmList"] = ", ".join(res[i][0] for i in range(7) if holm[i] < 0.05)
    macros["SigBHList"] = ", ".join(res[i][0] for i in range(7) if bh[i] < 0.05)
    macros["NsigBY"] = int((by < 0.05).sum())
    macros["SigBYList"] = ", ".join(res[i][0] for i in range(7) if by[i] < 0.05)
    macros["BYconst"] = f"{c_m:.4f}"


def tab_correlators(camp, oct_, out):
    """Two- and three-body classical correlators: structure the single-qubit
    marginals cannot see."""
    def corr(p):
        p = np.asarray(p); z = lambda k, i: 1 - 2 * int(S[k][i])
        o = {}
        for (i, j) in ((0, 1), (0, 2), (1, 2)):
            o[f"$\\langle Z_{i}Z_{j}\\rangle$"] = sum(p[k] * z(k, i) * z(k, j) for k in range(8))
        o["$\\langle Z_0Z_1Z_2\\rangle$"] = sum(p[k] * z(k, 0) * z(k, 1) * z(k, 2) for k in range(8))
        return o
    tgt = mean_law([r for r in camp if r["dist_id"] == "D1" and r["stage"] == "FULL"], "sim_")
    sets = [("Target", tgt)]
    for lad in ("A", "B"):
        sets.append((f"Mar--Apr, ladder~{lad}",
                     mean_law([r for r in camp if r["dist_id"] == "D1" and r["stage"] == "FULL" and r["ladder"] == lad])))
    for cfg, nm in (("control", "Control"), ("exp1", "Reduced")):
        for lad in ("A", "B"):
            sets.append((f"Oct, {nm}, ladder~{lad}",
                         mean_law([r for r in oct_ if r["config"] == cfg and r["ladder"] == lad])))
    keys = list(corr(tgt).keys())
    L = ["\\begin{tabular}{l" + "c" * len(keys) + "}", "\\toprule",
         "Mean law & " + " & ".join(keys) + " \\\\", "\\midrule"]
    for nm, p in sets:
        c = corr(p)
        L.append(nm + " & " + " & ".join(f"${c[k]:+.3f}$" for k in keys) + " \\\\")
        if nm == "Target": L.append("\\midrule")
    L += ["\\bottomrule", "\\end{tabular}"]
    (out / "correlators.tex").write_text("\n".join(L) + "\n")


def tab_october(oct_, out, macros):
    """The October factorial: four cells, then the paired contrasts."""
    L = ["\\begin{tabular}{llcccc}", "\\toprule",
         "Configuration & Ladder & Fidelity & $\\TV$ & $\\ell^2$ & Fidelity max \\\\", "\\midrule"]
    for cfg, nm in (("control", "Control"), ("exp1", "Reduced")):
        for k, lad in enumerate(("A", "B")):
            rs = [r for r in oct_ if r["config"] == cfg and r["ladder"] == lad]
            f = np.array([float(r["fid_tgt"]) for r in rs])
            L.append(f"{nm if k == 0 else '':8} & {lad} & {ms(f)} "
                     f"& {ms([float(r['tv_tgt']) for r in rs])} "
                     f"& {ms([float(r['l2_tgt']) for r in rs])} & ${f.max():.3f}$ \\\\")
        L.append("\\addlinespace")
    L += ["\\bottomrule", "\\end{tabular}"]
    (out / "october_cells.tex").write_text("\n".join(L) + "\n")

    cell = lambda c, l: {int(r["repeat"]): float(r["fid_tgt"])
                         for r in oct_ if r["config"] == c and r["ladder"] == l}
    cA, cB, eA, eB = cell("control", "A"), cell("control", "B"), cell("exp1", "A"), cell("exp1", "B")
    reps = sorted(set(cA) & set(cB) & set(eA) & set(eB))
    contrasts = [
        ("Ladder A $-$ B, Control",   np.array([cA[r] - cB[r] for r in reps])),
        ("Ladder A $-$ B, Reduced",   np.array([eA[r] - eB[r] for r in reps])),
        ("Reduction effect, ladder~A", np.array([eA[r] - cA[r] for r in reps])),
        ("Reduction effect, ladder~B", np.array([eB[r] - cB[r] for r in reps])),
        ("Ladder $\\times$ reduction interaction",
         np.array([(eB[r] - cB[r]) - (eA[r] - cA[r]) for r in reps])),
    ]
    L = ["\\begin{tabular}{lcccc}", "\\toprule",
         "Paired contrast & Estimate & 95\\% CI & $p$ (paired $t$) & $p$ (Wilcoxon) \\\\", "\\midrule"]
    for nm, d in contrasts:
        m, lo, hi = ci95(d)
        L.append(f"{nm} & ${m:+.4f}$ & $[{lo:+.4f},\\,{hi:+.4f}]$ "
                 f"& {pfmt(stats.ttest_1samp(d, 0)[1])} & {pfmt(stats.wilcoxon(d)[1])} \\\\")
    L += ["\\bottomrule", "\\end{tabular}"]
    (out / "october_contrasts.tex").write_text("\n".join(L) + "\n")
    macros["Nreps"] = len(reps)
    m, lo, hi = ci95(contrasts[-1][1])
    macros["InterEst"] = f"{m:+.4f}"; macros["InterLo"] = f"{lo:+.4f}"; macros["InterHi"] = f"{hi:+.4f}"
    macros["InterP"] = mathp(float(stats.ttest_1samp(contrasts[-1][1], 0)[1]))


def tab_longitudinal(camp, oct_, out, macros):
    """The ladder contrast across the three compilations of one logical circuit.

    The instrument record shows that the April campaign and the October campaign
    did not execute the same compiled circuit: April emitted 19 gates with the
    two level-1 undo rotations commanded as -theta + 720 deg, October Control 21
    gates with -theta and an identity-safe tail, October Reduced 18 gates. All
    three prepare the same target from the same logical circuit.
    """
    f = lambda rs: np.array([float(r["fid_tgt"]) for r in rs])
    ma = f([r for r in camp if r["dist_id"] == "D1" and r["stage"] == "FULL" and r["ladder"] == "A"])
    mb = f([r for r in camp if r["dist_id"] == "D1" and r["stage"] == "FULL" and r["ladder"] == "B"])
    dm = ma.mean() - mb.mean()
    cells = [("Oct, Control", "21", "$-\\theta$", "present",
              f([r for r in oct_ if r["config"] == "control" and r["ladder"] == "A"]),
              f([r for r in oct_ if r["config"] == "control" and r["ladder"] == "B"])),
             ("Apr, FULL", "19", "$-\\theta+720\\dg$", "absent", ma, mb),
             ("Oct, Reduced", "18", "$-\\theta$", "absent",
              f([r for r in oct_ if r["config"] == "exp1" and r["ladder"] == "A"]),
              f([r for r in oct_ if r["config"] == "exp1" and r["ladder"] == "B"]))]
    L = ["\\begin{tabular}{llllccc}", "\\toprule",
         "Acquisition & Gates & Undo rotations & Tail & Ladder A & Ladder B "
         "& A $-$ B \\\\", "\\midrule"]
    for nm, ng, undo, tail, a, b in cells:
        pv = float(stats.mannwhitneyu(a, b).pvalue)
        L.append(f"{nm} & {ng} & {undo} & {tail} & {ms(a)} & {ms(b)} & "
                 f"${a.mean() - b.mean():+.4f}$ ({pfmt(pv).strip('$')}) \\\\")
        key = {"Oct, Control": "Ctl", "Apr, FULL": "Apr", "Oct, Reduced": "Red"}[nm]
        macros[f"Cmp{key}AB"] = f"{a.mean() - b.mean():+.4f}"
        macros[f"Cmp{key}P"] = mathp(pv)
        macros[f"Cmp{key}Gates"] = ng
    L += ["\\bottomrule", "\\end{tabular}"]
    (out / "longitudinal.tex").write_text("\n".join(L) + "\n")

    # difference of effects, April against each October cell
    for cfg, key in (("control", "Ctl"), ("exp1", "Red")):
        oa = f([r for r in oct_ if r["config"] == cfg and r["ladder"] == "A"])
        ob = f([r for r in oct_ if r["config"] == cfg and r["ladder"] == "B"])
        do = oa.mean() - ob.mean()
        se = np.sqrt(ma.var(ddof=1) / len(ma) + mb.var(ddof=1) / len(mb)
                     + oa.var(ddof=1) / len(oa) + ob.var(ddof=1) / len(ob))
        z = (do - dm) / se
        pv = 2 * (1 - stats.norm.cdf(abs(z)))
        macros[f"Diff{key}Est"] = f"{do - dm:+.4f}"
        macros[f"Diff{key}Lo"] = f"{do - dm - 1.96 * se:+.4f}"
        macros[f"Diff{key}Hi"] = f"{do - dm + 1.96 * se:+.4f}"
        macros[f"Diff{key}Z"] = f"{z:.2f}"
        macros[f"Diff{key}P"] = mathp(pv)
    # kept for backward compatibility with text written against the old table
    macros["SessionLadderEst"] = macros["DiffCtlEst"]
    macros["SessionLadderZ"] = macros["DiffCtlZ"]
    macros["SessionLadderP"] = macros["DiffCtlP"]


def tab_marginals(camp, oct_, out):
    L = ["\\begin{tabular}{llccc}", "\\toprule",
         "Configuration & Ladder & $P(q_0{=}1)$ & $P(q_1{=}1)$ & $P(q_2{=}1)$ \\\\", "\\midrule"]
    for cfg, nm in (("control", "Control"), ("exp1", "Reduced")):
        for k, lad in enumerate(("A", "B")):
            rs = [r for r in oct_ if r["config"] == cfg and r["ladder"] == lad]
            m = [1 - np.mean([float(r[f"exp_q{i}_p0"]) for r in rs]) for i in range(3)]
            L.append(f"{nm if k == 0 else '':8} & {lad} & " + " & ".join(f"${v:.3f}$" for v in m) + " \\\\")
        L.append("\\addlinespace")
    L.append("\\midrule")
    rs = [r for r in camp if r["dist_id"] == "D1" and r["stage"] == "FULL" and r["ladder"] == "A"]
    m = [1 - np.mean([float(r[f"exp_q{i}_p0"]) for r in rs]) for i in range(3)]
    L.append("Mar--Apr 2026 & A & " + " & ".join(f"${v:.3f}$" for v in m) + " \\\\")
    L += ["\\bottomrule", "\\end{tabular}"]
    (out / "marginals.tex").write_text("\n".join(L) + "\n")



# ------------------------------------------------- correlators across suite
ZOPS = {}
for _i, _j in ((0, 1), (0, 2), (1, 2)):
    ZOPS[f"$\\langle Z_{_i}Z_{_j}\\rangle$"] = np.array(
        [(1 - 2 * int(S[k][_i])) * (1 - 2 * int(S[k][_j])) for k in range(8)], float)
ZOPS["$\\langle Z_0Z_1Z_2\\rangle$"] = np.array(
    [np.prod([1 - 2 * int(S[k][i]) for i in range(3)]) for k in range(8)], float)
MOPS = {f"$\\langle Z_{i}\\rangle$": np.array([1 - 2 * int(S[k][i]) for k in range(8)], float)
        for i in range(3)}


def tab_correlators_suite(camp, out, macros):
    """Every target of the suite, target vs measured two- and three-body
    correlators, and the test of a single-contraction description."""
    L = ["\\begin{tabular}{llcccc}", "\\toprule",
         "Target & & " + " & ".join(ZOPS.keys()) + " \\\\", "\\midrule"]
    allT, allE, lams = [], [], []
    for d in DISTS:
        rs = [r for r in camp if r["dist_id"] == d and r["stage"] == "FULL"]
        t, e = mean_law(rs, "sim_"), mean_law(rs)
        ct = {k: float(t @ v) for k, v in ZOPS.items()}
        ce = {k: float(e @ v) for k, v in ZOPS.items()}
        L.append(f"{d} & target & " + " & ".join("$%+.3f$" % (ct[k] if abs(ct[k]) > 5e-4 else 0.0) for k in ZOPS) + " \\\\")
        L.append("   & measured & " + " & ".join(("$%+.3f$" % (ce[k] if abs(ce[k]) > 5e-4 else 0.0)) for k in ZOPS) + " \\\\")
        L.append("\\addlinespace")
        T = np.array(list(ct.values())); E = np.array(list(ce.values()))
        lams.append(float(T @ E / (T @ T)) if T @ T > 1e-12 else np.nan)
        for k in list(ZOPS) + list(MOPS):
            v = ZOPS.get(k, MOPS.get(k))
            allT.append(float(t @ v)); allE.append(float(e @ v))
    L += ["\\bottomrule", "\\end{tabular}"]
    (out / "correlators_suite.tex").write_text("\n".join(L) + "\n")

    allT, allE = np.array(allT), np.array(allE)
    lam = float(allT @ allE / (allT @ allT))
    r2 = 1 - float(np.sum((allE - lam * allT) ** 2) / np.sum((allE - allE.mean()) ** 2))
    zero = np.abs(allE[np.abs(allT) < 1e-6])
    fin = [x for x in lams if np.isfinite(x)]
    macros["CorrLambda"] = f"{lam:.3f}"
    macros["CorrRtwo"] = f"{r2:.2f}"
    macros["CorrResidMax"] = f"{np.abs(allE - lam * allT).max():.3f}"
    macros["CorrLamMin"] = f"{min(fin):+.2f}"
    macros["CorrLamMax"] = f"{max(fin):+.2f}"
    macros["CorrZeroN"] = str(int(zero.size))
    macros["CorrZeroMax"] = f"{zero.max():.3f}"
    macros["CorrZeroRms"] = f"{np.sqrt((zero ** 2).mean()):.3f}"
    # D0: uniform target, invariant under any depolarising channel
    rs = [r for r in camp if r["dist_id"] == "D0" and r["stage"] == "FULL"]
    e0 = mean_law(rs)
    _zz = ZOPS["$\\langle Z_0Z_1\\rangle$"]
    macros["CorrDzeroZZ"] = "%+.3f" % float(e0 @ _zz)
    macros["CorrDzeroF"] = f"{np.mean([float(r['fid_tgt']) for r in rs]):.3f}"

    # targets whose single-qubit marginals are exactly balanced, and the
    # marginal accuracy attained on them
    nbal, margdev = 0, {}
    for d in DISTS:
        rs = [r for r in camp if r["dist_id"] == d and r["stage"] == "FULL"]
        t, e = mean_law(rs, "sim_"), mean_law(rs)
        mt = np.array([float(t @ v) for v in MOPS.values()])
        me = np.array([float(e @ v) for v in MOPS.values()])
        if np.abs(mt).max() < 1e-6: nbal += 1
        margdev[d] = float(np.abs(mt - me).max()) / 2.0
    macros["NbalancedTargets"] = str(nbal)
    for d in DISTS:
        macros[f"MargDev{d}"] = f"{margdev[d]:.3f}"
    # contraction of the dominant correlator on the two most structured targets
    for d in ("D3", "D4"):
        rs = [r for r in camp if r["dist_id"] == d and r["stage"] == "FULL"]
        t, e = mean_law(rs, "sim_"), mean_law(rs)
        zz = ZOPS["$\\langle Z_0Z_1\\rangle$"]
        macros[f"CorrRatio{d}"] = f"{100 * float(e @ zz) / float(t @ zz):.0f}"
        macros[f"CorrTgt{d}"] = "%+.3f" % float(t @ zz)
        macros[f"CorrObs{d}"] = "%+.3f" % float(e @ zz)

    # the leniency illustration quoted in Section 7: D1, ladder A
    rs = [r for r in camp if r["dist_id"] == "D1" and r["stage"] == "FULL" and r["ladder"] == "A"]
    macros["LenientFid"] = "%.3f" % np.mean([float(r["fid_tgt"]) for r in rs])
    macros["LenientTV"] = "%.3f" % np.mean([float(r["tv_tgt"]) for r in rs])


def tab_correlator_intervention(oct_, out, macros):
    """The October intervention seen in the dominant correlator, paired."""
    zz = ZOPS["$\\langle Z_0Z_1\\rangle$"]
    L = ["\\begin{tabular}{lccc}", "\\toprule",
         "Ladder & Control & Reduced & Reduced $-$ Control \\\\", "\\midrule"]
    for lad in ("A", "B"):
        c = vec(sorted([r for r in oct_ if r["config"] == "control" and r["ladder"] == lad],
                       key=lambda r: int(r["repeat"])), "exp_") @ zz
        e = vec(sorted([r for r in oct_ if r["config"] == "exp1" and r["ladder"] == lad],
                       key=lambda r: int(r["repeat"])), "exp_") @ zz
        m, lo, hi = ci95(e - c)
        pv = float(stats.ttest_rel(e, c).pvalue)
        L.append(f"{lad} & ${c.mean():+.3f} \\pm {c.std(ddof=1):.3f}$ & "
                 f"${e.mean():+.3f} \\pm {e.std(ddof=1):.3f}$ & "
                 f"${m:+.3f}$ $[{lo:+.3f},{hi:+.3f}]$, ${mathp(pv)}$ \\\\")
        macros[f"CorrRed{lad}Est"] = f"{m:+.3f}"
        macros[f"CorrRed{lad}Lo"] = f"{lo:+.3f}"
        macros[f"CorrRed{lad}Hi"] = f"{hi:+.3f}"
        macros[f"CorrRed{lad}P"] = mathp(pv)
        macros[f"CorrRed{lad}Res"] = f"{e.mean():+.3f}"
    L += ["\\bottomrule", "\\end{tabular}"]
    (out / "correlator_intervention.tex").write_text("\n".join(L) + "\n")
    tgt = float(mean_law([r for r in oct_], "sim_") @ zz)
    macros["CorrTargetZZ"] = f"{tgt:+.3f}"


def tab_density(export, oct_, out, macros):
    """Documentary audit of the execution_matrix objects in the instrument
    export. Nothing here is used as a physical result."""
    import json
    if not Path(export).exists():
        print(f"  [skip] density audit: {export} not found")
        return
    csv.field_size_limit(10 ** 9)
    runs = {r["run_name"]: r for r in oct_}
    rows = [r for r in csv.DictReader(open(export))
            if r.get("Nmae") in runs and r.get("Status") == "SUCCEED"]
    seen = {}
    for r in rows:
        seen[r["Nmae"]] = seen.get(r["Nmae"], 0) + 1
    uniq = [r for r in rows if seen[r["Nmae"]] == 1]
    H = T = EV = PD = 0.0
    rk, srk, recs = [], [], []
    for r in uniq:
        er = json.loads(r["Experiment Result"]); em = er["execution_matrix"]
        rho = (np.array(em["real"], float) + 1j * np.array(em["imag"], float)).reshape(8, 8)
        H = max(H, float(np.abs(rho - rho.conj().T).max()))
        T = max(T, abs(float(rho.trace().real) - 1))
        ev, _ = np.linalg.eigh((rho + rho.conj().T) / 2)
        EV = min(EV, float(ev.min())); rk.append(int((ev > 1e-9).sum()))
        pops = np.array(er["execution"], float)
        dep = np.array([float(runs[r["Nmae"]][f"exp_{s}"]) for s in S])
        PD = max(PD, float(np.abs(pops - dep).max()))
        sm = json.loads(r["Simulation Result"])["simulation_matrix"]
        SR = (np.array(sm["real"], float) + 1j * np.array(sm["imag"], float)).reshape(8, 8)
        es, vs = np.linalg.eigh((SR + SR.conj().T) / 2); srk.append(int((es > 1e-9).sum()))
        recs.append((r["Nmae"], rho, pops, vs[:, -1], float(runs[r["Nmae"]]["fid_tgt"])))
    P = np.array([x[2] for x in recs])
    C = np.array([np.abs(x[1] - np.diag(np.diag(x[1]))) for x in recs])
    d, i, j = min((0.5 * float(np.abs(P[a] - P[b]).sum()), a, b)
                  for a in range(len(P)) for b in range(a + 1, len(P)))
    F = np.array([float(np.real(x[3].conj() @ x[1] @ x[3])) for x in recs])
    Fcl = np.array([x[4] for x in recs])
    pur = np.array([float(np.real(np.trace(x[1] @ x[1]))) for x in recs])
    lab = [x[0] for x in recs]
    macros["DMn"] = str(len(recs))
    macros["DMdup"] = str(len(seen) - len(uniq))
    macros["DMpopDev"] = sci(PD)
    macros["DMherm"] = sci(H)
    macros["DMtrace"] = sci(T)
    macros["DMminev"] = sci(EV)
    macros["DMrankThree"] = str(int(sum(1 for x in rk if x == 3)))
    macros["DMrankFour"] = str(int(sum(1 for x in rk if x == 4)))
    macros["DMrankOther"] = str(int(sum(1 for x in rk if x not in (3, 4))))
    macros["DMsimRank"] = str(int(max(srk)))
    macros["DMcohTV"] = f"{d:.3f}"
    macros["DMcohDiff"] = f"{float(np.abs(C[i] - C[j]).max()):.3f}"
    macros["DMstateF"] = f"{float(np.median(F)):.3f}"
    macros["DMstateFlo"] = f"{F.min():.3f}"
    macros["DMstateFhi"] = f"{F.max():.3f}"
    macros["DMclF"] = f"{float(np.median(Fcl)):.3f}"
    macros["DMpurity"] = f"{float(np.median(pur)):.3f}"
    rho_s, p_s = stats.spearmanr(F, Fcl)
    macros["DMspearman"] = f"{rho_s:.2f}"
    for cfg, nm in (("control", "Ctl"), ("exp1", "Red")):
        a = np.array([F[k] for k, n in enumerate(lab) if f"_{cfg}_A_" in n])
        b = np.array([F[k] for k, n in enumerate(lab) if f"_{cfg}_B_" in n])
        macros[f"DMp{nm}"] = f"{float(stats.mannwhitneyu(a, b).pvalue):.3f}"
        macros[f"DM{nm}A"] = f"{a.mean():.3f}"
        macros[f"DM{nm}B"] = f"{b.mean():.3f}"



# --------------------------------------------------------------- section 2
MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
          "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


def tab_sessions(runs_jsonl, out, macros):
    """Acquisition structure of the first campaign: one target per contiguous
    session, reconstructed from the run-level timestamps."""
    import json
    from datetime import datetime
    if not Path(runs_jsonl).exists():
        print(f"  [skip] sessions: {runs_jsonl} not found")
        return
    rows = [json.loads(l) for l in open(runs_jsonl)]
    by = {}
    for r in rows:
        by.setdefault(r["dist_id"], []).append(r)
    recs = []
    for d, v in by.items():
        t0 = min(datetime.fromisoformat(x["created"]) for x in v)
        t1 = max(datetime.fromisoformat(x["end"]) for x in v)
        recs.append((t0, t1, d, len(v), sum(float(x["duration_s"]) for x in v) / 3600))
    recs.sort()
    L = ["\\begin{tabular}{llccc}", "\\toprule",
         "Order & Target & Session start (UTC) & Span & Runs \\\\", "\\midrule"]
    for i, (t0, t1, d, n, _) in enumerate(recs, 1):
        start = f"{t0.day}~{MONTHS[t0.month - 1]}, {t0:%H:%M}"
        span = (t1 - t0).total_seconds() / 3600
        L.append(f"{i} & {d} & {start} & ${span:.1f}$\\,h & {n} \\\\")
    L += ["\\bottomrule", "\\end{tabular}"]
    (out / "sessions.tex").write_text("\n".join(L) + "\n")
    macros["SessOrder"] = ", ".join(r[2] for r in recs)
    macros["SessNtotal"] = str(sum(r[3] for r in recs))
    macros["SessHours"] = f"{sum(r[4] for r in recs):.1f}"
    macros["SessSpanLo"] = f"{min((r[1] - r[0]).total_seconds() / 3600 for r in recs):.1f}"
    macros["SessSpanHi"] = f"{max((r[1] - r[0]).total_seconds() / 3600 for r in recs):.1f}"
    macros["SessFirst"] = f"{recs[0][0].day}~{MONTHS[recs[0][0].month - 1]}"
    macros["SessLast"] = f"{recs[-1][1].day}~{MONTHS[recs[-1][1].month - 1]}"


def tab_spam(camp, out, macros):
    """The one target-independent device observable the truncated stages give:
    population on qubits that receive no gate."""
    DI = {"L0": (1, 2), "L01": (2,)}
    L = ["\\begin{tabular}{lccc}", "\\toprule",
         "Session & \\multicolumn{2}{c}{$L0$: $P(q_i{=}1)$, no gate applied} "
         "& $L01$: $P(q_2{=}1)$ \\\\",
         "\\cmidrule(lr){2-3}\\cmidrule(lr){4-4}",
         " & $q_1$ & $q_2$ & $q_2$ \\\\", "\\midrule"]
    cols = {"L0": {1: [], 2: []}, "L01": {2: []}}
    means = {}
    for d in DISTS:
        row = [d]
        for st, idle in DI.items():
            rs = [r for r in camp if r["dist_id"] == d and r["stage"] == st]
            for i in idle:
                v = np.array([1 - float(r[f"exp_q{i}_p0"]) for r in rs])
                cols[st][i].append(v)
                row.append(f"${v.mean():.3f} \\pm {v.std(ddof=1):.3f}$")
        means[d] = row
        L.append(" & ".join(row) + " \\\\")
    L += ["\\midrule"]
    pooled = []
    for st, idle in DI.items():
        for i in idle:
            pooled.append(np.concatenate(cols[st][i]).mean())
    L.append("Pooled & " + " & ".join(f"${v:.3f}$" for v in pooled) + " \\\\")
    L += ["\\bottomrule", "\\end{tabular}"]
    (out / "spam.tex").write_text("\n".join(L) + "\n")

    macros["SpamQone"] = f"{pooled[0]:.3f}"
    macros["SpamQtwo"] = f"{pooled[1]:.3f}"
    macros["SpamQtwoLone"] = f"{pooled[2]:.3f}"
    macros["SpamKWqone"] = sci(float(stats.kruskal(*cols["L0"][1]).pvalue))
    macros["SpamKWqtwo"] = sci(float(stats.kruskal(*cols["L0"][2]).pvalue))
    macros["SpamKWLone"] = sci(float(stats.kruskal(*cols["L01"][2]).pvalue))
    pair = np.array([0.5 * (cols["L0"][1][k].mean() + cols["L0"][2][k].mean())
                     for k in range(len(DISTS))])
    macros["SpamRangeLo"] = f"{pair.min():.3f}"
    macros["SpamRangeHi"] = f"{pair.max():.3f}"
    macros["SpamRangeRatio"] = f"{pair.max() / pair.min():.1f}"
    f = np.array([np.mean([float(r["fid_tgt"]) for r in camp
                           if r["dist_id"] == d and r["stage"] == "FULL"]) for d in DISTS])
    rho, pv = stats.spearmanr(pair, f)
    macros["SpamFidRho"] = f"{rho:+.2f}"
    macros["SpamFidP"] = f"{pv:.2f}"
    order = {d: i for i, d in enumerate(macros["SessOrder"].split(", "))} \
        if "SessOrder" in macros else {d: i for i, d in enumerate(DISTS)}
    o = np.array([order[d] for d in DISTS])
    rho, pv = stats.spearmanr(pair, o)
    macros["SpamOrderRho"] = f"{rho:+.2f}"
    macros["SpamOrderP"] = f"{pv:.2f}"
    # population falling outside the ideal support, per stage
    for st, nm in (("L0", "Lzero"), ("L01", "Lone")):
        leaks = []
        for d in DISTS:
            rs = [r for r in camp if r["dist_id"] == d and r["stage"] == st]
            t, e = mean_law(rs, "sim_"), mean_law(rs)
            leaks.append(float(sum(e[k] for k in range(8) if t[k] <= 1e-9)))
        macros[f"Leak{nm}Lo"] = f"{min(leaks):.3f}"
        macros[f"Leak{nm}Hi"] = f"{max(leaks):.3f}"


def macros_metrics(camp, out, macros):
    """Mean of per-run fidelities against fidelity of the mean law."""
    g = []
    for d in DISTS:
        for lad in ("A", "B"):
            rs = [r for r in camp if r["dist_id"] == d and r["stage"] == "FULL"
                  and r["ladder"] == lad]
            t, e = mean_law(rs, "sim_"), mean_law(rs)
            g.append(fid(e, t) - np.mean([float(r["fid_tgt"]) for r in rs]))
    g = np.array(g)
    macros["FidGapLo"] = f"{g.min():+.4f}"
    macros["FidGapHi"] = f"{g.max():+.4f}"
    macros["FidGapMed"] = f"{np.median(g):+.4f}"
    macros["FidGapNcells"] = str(len(g))

    # the TV identity at the FULL stage: the ideal simulator IS the target there
    full = [r for r in camp if r["stage"] == "FULL"]
    d = np.array([abs(float(r["tv_tgt"]) - float(r["tv_sim"])) for r in full])
    macros["TVidentityMax"] = sci(float(d.max()))
    macros["TVidentityN"] = str(len(full))


def macros_readout(pilot, out, macros):
    """The readout-mitigation session: confusion matrix and its effect."""
    pilot = Path(pilot)
    if not pilot.exists():
        print(f"  [skip] readout: {pilot} not found")
        return
    M = np.load(pilot / "Mfull_20260308-112738.npy")
    macros["RoCond"] = f"{np.linalg.cond(M):.2f}"
    macros["RoDiag"] = f"{np.mean(np.diag(M)):.3f}"
    rows = list(csv.DictReader(open(pilot / "gr_summary_20260308-131538.csv")))
    want = [("Raw (no mitigation)", "raw"),
            ("Mitigated (confusion-matrix inverse)", "mitigated"),
            ("Partially mitigated ($\\lambda=0.3$)", "mixed")]
    nm, labs = [], []
    for lab, key in want:
        hit = [r for r in rows if key in r["comparison"].lower()]
        if hit:
            nm.append(hit[0]); labs.append(lab)
    L = ["\\begin{tabular}{lccc}", "\\toprule",
         "Post-processing & $\\TV$ & $\\ell^2$ & Classical fidelity \\\\", "\\midrule"]
    for lab, r in zip(labs, nm):
        L.append(f"{lab} & ${float(r['tv']):.3f}$ & ${float(r['l2']):.3f}$ "
                 f"& ${float(r['fidelity']):.3f}$ \\\\")
    L += ["\\bottomrule", "\\end{tabular}"]
    (out / "readout.tex").write_text("\n".join(L) + "\n")
    for lab, r in zip(["Raw", "Mit", "Mix"], nm):
        macros[f"Ro{lab}Fid"] = f"{float(r['fidelity']):.3f}"
        macros[f"Ro{lab}TV"] = f"{float(r['tv']):.3f}"



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



def main():
    ap = argparse.ArgumentParser()
    here = Path(__file__).resolve().parents[1]
    ap.add_argument("--campaign", default=str(here / "data/campaign_v2_runs/runs_flat_v2.csv"))
    ap.add_argument("--october", default=str(
        here.parent / "grover-rudolph-practical-implementation/data/ladder_ab/D1_ladder_ab_runs.csv"))
    ap.add_argument("--export", default=str(
        here.parent / "grover-rudolph-practical-implementation/data/spinqit_export_ladderab_20261005.csv"))
    ap.add_argument("--runs-jsonl", default=str(
        here / "data/campaign_v2_runs/campaign_v2_runs_clean.jsonl"))
    ap.add_argument("--pilot", default=str(
        here.parent / "grover-rudolph-practical-implementation/data/pilot_d1_20260308"))
    ap.add_argument("--outdir", default=str(here / "tables"))
    a = ap.parse_args()
    out = Path(a.outdir); out.mkdir(parents=True, exist_ok=True)
    camp, oct_ = load(a.campaign), load(a.october)
    print(f"campaign: {len(camp)} runs   october: {len(oct_)} runs")

    macros = {}
    tab_stage_decomposition(camp, out)
    tab_full_ab(camp, out, macros)
    tab_correlators(camp, oct_, out)
    tab_october(oct_, out, macros)
    tab_longitudinal(camp, oct_, out, macros)
    tab_marginals(camp, oct_, out)
    tab_correlators_suite(camp, out, macros)
    tab_correlator_intervention(oct_, out, macros)
    tab_density(a.export, oct_, out, macros)
    tab_sessions(a.runs_jsonl, out, macros)
    tab_spam(camp, out, macros)
    macros_metrics(camp, out, macros)
    macros_readout(a.pilot, out, macros)
    tab_targets(out, macros)
    macros_angle(camp, out, macros)

    with (out / "generated_macros.tex").open("w") as fh:
        fh.write("% generated by scripts/90_make_tables.py -- do not edit\n")
        words = {"0": "zero", "1": "one", "2": "two", "3": "three", "4": "four",
                 "5": "five", "6": "six", "7": "seven", "8": "eight", "9": "nine"}
        for k, v in macros.items():
            for d, w in words.items():        # TeX names admit letters only
                k = k.replace(d, w)
            fh.write(f"\\newcommand{{\\gen{k}}}{{{v}}}\n")
    for f in sorted(out.glob("*.tex")):
        print(f"  wrote {f.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
