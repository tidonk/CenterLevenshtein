"""Generate a per-instance solver-comparison table in Markdown and LaTeX.

Rows    : instances, sorted by (length, number, iteration)
Columns : GRB | mono | benders_full | benders_partial

Cell values:
  X.XXs        solved to optimality  (solving time)
  gap X.X%     ran, not optimal
  --           not run / no CSV data

Bottom summary rows (one value per method column):
  # optimal        count of instances solved to optimality
  # ran            count of instances where method ran
  sum (s)          sum of solving times  [over solved instances]
  mean (s)         mean  --
  sh. geomean (s)  shifted geometric mean (shift=10)  --
  Wilcoxon p       two-sided signed-rank vs GRB  ("ref" in GRB col)
  Wilcoxon sig.    *** p<0.001  ** p<0.01  * p<0.05  ns otherwise

Outputs:
  plots/instance_table.md
  plots/instance_table.tex

Usage:
  python visu/instance_table.py [--results DIR] [--tl SECONDS]
"""
import argparse
import csv
import glob
import math
import os
import re

from scipy.stats import wilcoxon as scipy_wilcoxon

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RESULTS = os.path.join(ROOT, "results")
PLOTS   = os.path.join(ROOT, "plots")

METHOD_ORDER = ["GRB", "mono", "benders_full", "benders_partial", "benders_2rand", "benders_adap_sp"]
MD_LABELS = {
    "GRB":              "Gurobi",
    "mono":             "SCIP (mono)",
    "benders_full":     "Benders (full)",
    "benders_partial":  "Benders (m/4)",
    "benders_2rand":    "Benders (2 rand)",
    "benders_adap_sp":  "Benders (adap. SP)",
}
TEX_LABELS = {
    "GRB":              r"Gurobi",
    "mono":             r"SCIP (mono)",
    "benders_full":     r"Benders (full)",
    "benders_partial":  r"Benders ($\nicefrac{m}{4}$)",
    "benders_2rand":    r"Benders (2 rand.)",
    "benders_adap_sp":  r"Benders (adap.\ SP)",
}
TOL = 1e-6


# ---------------------------------------------------------------------------
# Data loading  (mirrors compare_analyze.load_results)
# ---------------------------------------------------------------------------
class Rec:
    __slots__ = ("ran", "has_primal", "solved", "time", "gap")

    def __init__(self, ran=False, has_primal=False, solved=False,
                 time=None, gap=None):
        self.ran = ran
        self.has_primal = has_primal
        self.solved = solved
        self.time = time
        self.gap = gap


def _f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def load_results(results_dir):
    """Return {method: {instance_stem: Rec}}."""
    out = {}
    for path in sorted(glob.glob(os.path.join(results_dir, "*.csv"))):
        if os.path.basename(path) == "result_path_binary_2n.csv":
            continue
        with open(path) as f:
            for row in csv.DictReader(f):
                if "method" not in row:
                    break
                method = row["method"]
                inst   = os.path.basename(row["instance"])
                bound  = _f(row.get("bound"))
                prim_solver = _f(row.get("primal"))
                tcost  = _f(row.get("true_cost"))
                primal = tcost if tcost is not None else prim_solver
                solved = row.get("status") == "optimal"
                if method == "GRB":
                    gap = _f(row.get("gap"))
                elif primal is not None and bound is not None and abs(primal) > TOL:
                    gap = abs(primal - bound) / abs(primal)
                else:
                    gap = None
                if solved:
                    gap = 0.0
                out.setdefault(method, {})[inst] = Rec(
                    ran=True,
                    has_primal=primal is not None,
                    solved=solved,
                    time=_f(row.get("time")),
                    gap=gap,
                )
    return out


def score(rec, tl):
    """Dominance score for Wilcoxon (lower = better)."""
    if rec is None or not rec.ran:
        return tl + 1 + 2e6
    if not rec.has_primal:
        return tl + 1 + 1e6
    if rec.solved:
        return rec.time if rec.time is not None else tl
    return tl + 1 + (rec.gap if rec.gap is not None else 1.0)


def is_better(rec, ref_rec, tl):
    """True if rec has a strictly lower dominance score than ref_rec."""
    return score(rec, tl) < score(ref_rec, tl)


def shifted_geomean(values, shift=10.0):
    if not values:
        return float("nan")
    return math.exp(sum(math.log(v + shift) for v in values) / len(values)) - shift


# ---------------------------------------------------------------------------
# Instance sorting key
# ---------------------------------------------------------------------------
def _sort_key(name):
    base = name.replace(".txt", "")
    if base.startswith("rm_"):
        parts = base.split("_")
        return (0, int(parts[1]), 0, int(parts[2]))
    if base.startswith("hamming_"):
        parts = base.split("_")
        return (1, int(parts[1]), 0, int(parts[2]))
    m = re.match(r"I_(\d+)_(\d+)_(\d+)", base)
    if m:
        return (2, int(m.group(1)), int(m.group(2)), int(m.group(3)))
    return (3, 0, 0, 0)


# ---------------------------------------------------------------------------
# Cell formatting
# ---------------------------------------------------------------------------
def fmt_cell(rec):
    if rec is None or not rec.ran:
        return "--"
    if rec.solved:
        t = rec.time if rec.time is not None else 0.0
        return f"{t:.2f}"
    if rec.has_primal and rec.gap is not None:
        return f"{rec.gap * 100:.1f}%"
    return "no sol."


def fmt_cell_tex(rec):
    if rec is None or not rec.ran:
        return r"\text{--}"
    if rec.solved:
        t = rec.time if rec.time is not None else 0.0
        return f"{t:.2f}"
    if rec.has_primal and rec.gap is not None:
        return f"{rec.gap * 100:.1f}\\%"
    return r"\text{no sol.}"


# ---------------------------------------------------------------------------
# Wilcoxon vs reference method
# ---------------------------------------------------------------------------
def wilcoxon_vs_ref(data, instances, tl, ref_method):
    """Return {method: (p_value, sig_str)} for all methods vs ref_method.
    The ref_method itself maps to (None, "ref")."""
    results = {}
    ref_scores = [score(data.get(ref_method, {}).get(i), tl) for i in instances]
    for m in METHOD_ORDER:
        if m == ref_method:
            results[m] = (None, "ref")
            continue
        if m not in data:
            results[m] = (float("nan"), "n/a")
            continue
        m_scores = [score(data[m].get(i), tl) for i in instances]
        diffs = [a - b for a, b in zip(ref_scores, m_scores) if abs(a - b) > 1e-12]
        if not diffs:
            results[m] = (float("nan"), "tie")
            continue
        try:
            # H1: ref_scores > m_scores (method is better than reference)
            _, p = scipy_wilcoxon(ref_scores, m_scores,
                                  zero_method="wilcox", alternative="greater")
        except ValueError:
            p = float("nan")
        if math.isnan(p):
            sig = "n/a"
        elif p < 0.001:
            sig = "***"
        elif p < 0.01:
            sig = "**"
        elif p < 0.05:
            sig = "*"
        else:
            sig = "ns"
        results[m] = (p, sig)
    return results


# ---------------------------------------------------------------------------
# Summary stats per method
# ---------------------------------------------------------------------------
def method_stats(data, instances, tl):
    """Return {method: dict-of-stats}."""
    n = len(instances)
    out = {}
    for m in METHOD_ORDER:
        md = data.get(m, {})
        n_ran    = sum(1 for i in instances if md.get(i) and md[i].ran)
        n_opt    = sum(1 for i in instances if md.get(i) and md[i].solved)
        times    = [md[i].time for i in instances
                    if md.get(i) and md[i].solved and md[i].time is not None]
        s        = sum(times) if times else float("nan")
        mean_t   = (s / len(times)) if times else float("nan")
        sgm      = shifted_geomean(times)
        out[m]   = dict(n=n, n_ran=n_ran, n_opt=n_opt,
                        sum=s, mean=mean_t, sgm=sgm)
    return out


# ---------------------------------------------------------------------------
# Markdown table
# ---------------------------------------------------------------------------
def build_md(data, instances, stats, wilco, labels, methods, ref_label,
             ref_method=None, tl=600.0):
    cols = ["**Instance**"] + [f"**{labels[m]}**" for m in methods]
    sep  = [":---"] + ["---:" for _ in methods]

    rows = []
    rows.append(" | ".join(cols))
    rows.append(" | ".join(sep))

    for inst in instances:
        cells = [inst]
        ref_rec = data.get(ref_method, {}).get(inst) if ref_method else None
        for m in methods:
            rec = data.get(m, {}).get(inst)
            cell = fmt_cell(rec)
            if m != ref_method and ref_rec is not None and is_better(rec, ref_rec, tl):
                cell = f"**{cell}**"
            cells.append(cell)
        rows.append(" | ".join(cells))

    def sr(label, values):
        return " | ".join([f"**{label}**"] + list(values))

    def _fmt(v):
        return f"{v:.2f}" if not math.isnan(v) else "--"

    n = stats[methods[0]]["n"]
    rows.append(sr("\\# optimal",  [f"{stats[m]['n_opt']}/{n}" for m in methods]))
    rows.append(sr("\\# ran",      [f"{stats[m]['n_ran']}/{n}" for m in methods]))
    rows.append(sr("sum",          [_fmt(stats[m]["sum"])       for m in methods]))
    rows.append(sr("mean",         [_fmt(stats[m]["mean"])      for m in methods]))
    rows.append(sr("sh. geomean",  [_fmt(stats[m]["sgm"])       for m in methods]))

    p_vals, sigs = [], []
    for m in methods:
        p, sig = wilco[m]
        p_vals.append(f"{p:.4f}" if p is not None and not math.isnan(p) else sig)
        sigs.append(sig)
    rows.append(sr(f"Wilcoxon *p* (better than {ref_label})", p_vals))
    rows.append(sr("Wilcoxon sig.",                   sigs))

    return "\n".join(rows)


# ---------------------------------------------------------------------------
# LaTeX table
# ---------------------------------------------------------------------------
def _tex_esc(s):
    return s.replace("_", r"\_").replace("%", r"\%").replace("&", r"\&")


def build_tex(data, instances, stats, wilco, labels, methods, ref_label,
              label_suffix="", ref_method=None, tl=600.0):
    colspec = "l" + "r" * len(methods)

    lines = []
    lines.append(r"\begin{table}[htbp]")
    lines.append(r"\centering")
    lines.append(r"\small")
    caption = ("Per-instance solver comparison" + (f" ({label_suffix})" if label_suffix else "")
               + r" (time in seconds if optimal; gap if feasible; \text{--} if not run)")
    lines.append(r"\caption{" + caption + "}")
    tag = "noGRB" if "GRB" not in methods else "GRB"
    lines.append(r"\label{tab:instance-comparison-" + tag + "}")
    lines.append(r"\begin{tabular}{" + colspec + r"}")
    lines.append(r"\toprule")
    lines.append(" & ".join(["Instance"] + [labels[m] for m in methods]) + r" \\")
    lines.append(r"\midrule")

    for inst in instances:
        ref_rec = data.get(ref_method, {}).get(inst) if ref_method else None
        cells = [_tex_esc(inst)]
        for m in methods:
            rec = data.get(m, {}).get(inst)
            cell = fmt_cell_tex(rec)
            if m != ref_method and ref_rec is not None and is_better(rec, ref_rec, tl):
                cell = r"\textbf{" + cell + "}"
            cells.append(cell)
        lines.append(" & ".join(cells) + r" \\")

    lines.append(r"\midrule")

    def sr(label, values):
        return r"\textbf{" + label + "} & " + " & ".join(values) + r" \\"

    def _fmt(v):
        return f"{v:.2f}" if not math.isnan(v) else r"\text{--}"

    n = stats[methods[0]]["n"]
    lines.append(sr(r"\# optimal",     [f"{stats[m]['n_opt']}/{n}" for m in methods]))
    lines.append(sr(r"\# ran",         [f"{stats[m]['n_ran']}/{n}" for m in methods]))
    lines.append(sr("sum",             [_fmt(stats[m]["sum"])       for m in methods]))
    lines.append(sr("mean",            [_fmt(stats[m]["mean"])      for m in methods]))
    lines.append(sr(r"sh.\ geomean",   [_fmt(stats[m]["sgm"])       for m in methods]))

    p_vals, sigs = [], []
    for m in methods:
        p, sig = wilco[m]
        p_vals.append(f"{p:.4f}" if p is not None and not math.isnan(p) else r"\text{" + sig + "}")
        sigs.append(r"\text{" + sig + "}")
    lines.append(sr(f"Wilcoxon $p$ (better than {_tex_esc(ref_label)})", p_vals))
    lines.append(sr("Wilcoxon sig.", sigs))

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Helpers to build one variant
# ---------------------------------------------------------------------------
SCIP_METHODS = ["mono", "benders_full", "benders_partial", "benders_2rand", "benders_adap_sp"]


def make_variant(data, all_insts, stats, tl, methods, ref, label_suffix=""):
    """Return (md_text, tex_text) for the given method subset and reference."""
    wilco = wilcoxon_vs_ref(data, all_insts, tl, ref)
    ref_label_md  = MD_LABELS.get(ref, ref)
    ref_label_tex = TEX_LABELS.get(ref, ref)
    md  = build_md( data, all_insts, stats, wilco, MD_LABELS,  methods, ref_label_md,
                    ref_method=ref, tl=tl)
    tex = build_tex(data, all_insts, stats, wilco, TEX_LABELS, methods, ref_label_tex,
                    label_suffix=label_suffix, ref_method=ref, tl=tl)
    return md, tex


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=RESULTS)
    ap.add_argument("--tl",      type=float, default=600.0)
    ap.add_argument("--out",     default=PLOTS, help="output directory")
    args = ap.parse_args()

    data = load_results(args.results)
    if not data:
        print(f"No CSV files found under {args.results}")
        return

    all_insts = sorted(
        set().union(*[set(d.keys()) for d in data.values()]),
        key=_sort_key,
    )
    stats = method_stats(data, all_insts, args.tl)
    os.makedirs(args.out, exist_ok=True)

    variants = []

    # --- GRB variant (all four methods, Wilcoxon vs GRB) ---
    grb_methods = [m for m in METHOD_ORDER if m in data]
    if "GRB" in data:
        md, tex = make_variant(data, all_insts, stats, args.tl,
                               grb_methods, ref="GRB", label_suffix="with Gurobi")
        variants.append(("GRB", md, tex,
                         os.path.join(args.out, "instance_table_GRB.md"),
                         os.path.join(args.out, "instance_table_GRB.tex")))

    # --- SCIP-only variant (mono / benders_full / benders_partial, vs mono) ---
    scip_methods = [m for m in SCIP_METHODS if m in data]
    if scip_methods:
        ref_scip = "mono" if "mono" in data else scip_methods[0]
        md, tex = make_variant(data, all_insts, stats, args.tl,
                               scip_methods, ref=ref_scip, label_suffix="SCIP only")
        variants.append(("noGRB", md, tex,
                         os.path.join(args.out, "instance_table_noGRB.md"),
                         os.path.join(args.out, "instance_table_noGRB.tex")))

    for name, md, tex, out_md, out_tex in variants:
        with open(out_md,  "w") as f:
            f.write(md + "\n")
        with open(out_tex, "w") as f:
            f.write(tex + "\n")
        print(f"\n{'='*60}")
        print(f"  variant: {name}")
        print('='*60)
        print(md)
        print(f"\nwrote {out_md}")
        print(f"wrote {out_tex}")


if __name__ == "__main__":
    main()
