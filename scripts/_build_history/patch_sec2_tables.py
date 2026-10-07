#!/usr/bin/env python3
"""Append the Section 2 generators to 90_make_tables.py."""
from pathlib import Path
import sys

p = Path(sys.argv[1] if len(sys.argv) > 1 else "scripts/90_make_tables.py")
s = p.read_text()

NEW = r'''
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
    nm = [r for r in rows if r["comparison"] == "Target vs NM"]
    labs = ["Raw", "Mitigated", "Mixed"]
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


'''

s = s.replace("\ndef main():", NEW + "\ndef main():")
s = s.replace(
    '    ap.add_argument("--outdir", default=str(here / "tables"))',
    '    ap.add_argument("--runs-jsonl", default=str(\n'
    '        here / "data/campaign_v2_runs/campaign_v2_runs_clean.jsonl"))\n'
    '    ap.add_argument("--pilot", default=str(\n'
    '        here.parent / "grover-rudolph-practical-implementation/data/pilot_d1_20260308"))\n'
    '    ap.add_argument("--outdir", default=str(here / "tables"))')
s = s.replace(
    "    tab_density(a.export, oct_, out, macros)",
    "    tab_density(a.export, oct_, out, macros)\n"
    "    tab_sessions(a.runs_jsonl, out, macros)\n"
    "    tab_spam(camp, out, macros)\n"
    "    macros_metrics(camp, out, macros)\n"
    "    macros_readout(a.pilot, out, macros)")
p.write_text(s)
print("patched", p)
