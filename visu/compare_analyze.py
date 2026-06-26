"""Analyze the 4-way comparison: GRB, SCIP-mono, SCIP-Benders (full), SCIP-Benders
(partial, retain ceil(m/4)).

Inputs (in results/):
  - bench_compare.csv            long format from compare_run.py (SCIP methods)
  - result_path_binary_2n.csv    GRB (Gurobi) reference

Outputs (in results/):
  - perf_profile.png / .pdf      Dolan-More performance profile on solving time
  - comparison_report.md         table (sum/mean/shifted-geomean on solved) +
                                 Wilcoxon signed-rank tests (all instances)

Conventions:
  * "solved" = proved optimal (gap ~ 0) within the time limit.
  * For the Benders methods the *validated* true cost (true_cost column) is used
    as the primal; the gap is recomputed from it and the SCIP dual bound.
  * Signed-rank dominance score per (method, instance), lower = better:
        solved             -> solving time           (in [0, TL])
        feasible, not opt   -> TL + 1 + gap           (larger gap = worse)
        ran, no primal      -> TL + 1 + 1e6
        not run             -> TL + 1 + 2e6
    so: solved < feasible < no-primal < not-run, exactly the requested ordering.
"""
import argparse
import csv
import math
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import ConnectionPatch
from scipy.stats import wilcoxon

import glob

TOL = 1e-6
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RESULTS = os.path.join(ROOT, "results")
PLOTS = os.path.join(ROOT, "plots")

METHOD_ORDER = ["GRB", "mono", "benders_full", "benders_partial", "benders_2rand", "benders_adap_sp"]
LABELS = {"GRB": "Gurobi (mono)", "mono": "SCIP (mono)",
          "benders_full": "SCIP-Benders (full)",
          "benders_partial": "SCIP-Benders (partial m/4)",
          "benders_2rand": "SCIP-Benders (2 rand)",
          "benders_adap_sp": "SCIP-Benders (adap. SP)"}


class Rec:
    __slots__ = ("ran", "has_primal", "solved", "time", "gap", "primal", "bound")

    def __init__(self, ran=False, has_primal=False, solved=False,
                 time=None, gap=None, primal=None, bound=None):
        self.ran, self.has_primal, self.solved = ran, has_primal, solved
        self.time, self.gap, self.primal, self.bound = time, gap, primal, bound


def _f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def load_results(results_dir):
    """Read every per-instance results/<INSTANCE>.csv (each holds one row per
    method) into {method: {instance: Rec}}. The GRB reference CSV is skipped."""
    out = {}
    pattern = os.path.join(results_dir, "*.csv")
    for path in glob.glob(pattern):
        if os.path.basename(path) == "result_path_binary_2n.csv":
            continue
        with open(path) as f:
            for row in csv.DictReader(f):
                if "method" not in row:
                    break  # not a per-instance results file
                method = row["method"]
                inst = os.path.basename(row["instance"])
                bound = _f(row.get("bound"))
                solver_primal = _f(row.get("primal"))
                tcost = _f(row.get("true_cost"))
                # prefer the validated true cost as the primal when available
                primal = tcost if tcost is not None else solver_primal
                solved = (row.get("status") == "optimal")
                if method == "GRB":
                    gap = _f(row.get("gap"))
                elif primal is not None and bound is not None and abs(primal) > TOL:
                    gap = abs(primal - bound) / abs(primal)
                else:
                    gap = None
                if solved:
                    gap = 0.0
                out.setdefault(method, {})[inst] = Rec(
                    ran=True, has_primal=primal is not None, solved=solved,
                    time=_f(row.get("time")), gap=gap, primal=primal, bound=bound)
    return out


def score(rec, tl):
    if rec is None or not rec.ran:
        return tl + 1 + 2e6
    if not rec.has_primal:
        return tl + 1 + 1e6
    if rec.solved:
        return rec.time if rec.time is not None else tl
    return tl + 1 + (rec.gap if rec.gap is not None else 1.0)


def shifted_geomean(values, shift=10.0):
    if not values:
        return float("nan")
    return math.exp(sum(math.log(v + shift) for v in values) / len(values)) - shift


def build_table(data, instances, tl):
    """Aggregates (sum/mean/sgm of time) on the COMMON solved subset."""
    common = [i for i in instances
              if all(data[m].get(i) and data[m][i].solved for m in METHOD_ORDER if m in data)]
    lines = []
    lines.append(f"Common solved-to-optimality subset: {len(common)} / {len(instances)} instances")
    lines.append("")
    lines.append("| method | # solved (all) | sum t (s) | mean t (s) | shifted-geomean t (s) |")
    lines.append("|---|---|---|---|---|")
    for m in METHOD_ORDER:
        if m not in data:
            continue
        nsolved = sum(1 for i in instances if data[m].get(i) and data[m][i].solved)
        times = [data[m][i].time for i in common if data[m][i].time is not None]
        ssum = sum(times) if times else float("nan")
        smean = (ssum / len(times)) if times else float("nan")
        sgm = shifted_geomean(times)
        lines.append(f"| {LABELS[m]} | {nsolved}/{len(instances)} | "
                     f"{ssum:.2f} | {smean:.2f} | {sgm:.2f} |")
    return "\n".join(lines), common


def wilcoxon_block(data, instances, tl):
    lines = ["", "## Wilcoxon signed-rank (all instances; dominance score)", "",
             "One-sided: p-value tests whether the winner column is significantly",
             "better than the other. Lower score is better. n = non-tied instances.", "",
             "| A | B | n | W | p-value | better |", "|---|---|---|---|---|---|"]
    methods = [m for m in METHOD_ORDER if m in data]
    for a_i in range(len(methods)):
        for b_i in range(a_i + 1, len(methods)):
            A, B = methods[a_i], methods[b_i]
            sa = [score(data[A].get(i), tl) for i in instances]
            sb = [score(data[B].get(i), tl) for i in instances]
            diffs = [x - y for x, y in zip(sa, sb)]
            nz = [d for d in diffs if abs(d) > 1e-12]
            if not nz:
                lines.append(f"| {LABELS[A]} | {LABELS[B]} | 0 | - | - | tie |")
                continue
            med = sorted(nz)[len(nz) // 2]
            better = LABELS[A] if med < 0 else LABELS[B]
            # one-sided: test H1 in the direction of the observed winner
            alt = "less" if med < 0 else "greater"
            try:
                W, p = wilcoxon(sa, sb, zero_method="wilcox", alternative=alt)
            except ValueError:
                W, p = float("nan"), float("nan")
            lines.append(f"| {LABELS[A]} | {LABELS[B]} | {len(nz)} | "
                         f"{W:.1f} | {p:.4g} | {better} |")
    return "\n".join(lines)


def perf_profile(data, instances, tl, out_png):
    """Combined Dolan-Moré time profile (left) + gap CDF (right), publication-ready.

    A dashed vertical line marks the scale break. The gap axis starts at 0 %
    there (instances solved within TL have gap = 0) and increases rightward.
    Lines are uninterrupted across the break via ConnectionPatch connectors.
    """
    methods = [m for m in METHOD_ORDER if m in data]
    n = len(instances)

    # --- LEFT: Dolan-Moré time-ratio profile ---
    ratios = {m: [] for m in methods}
    max_ratio = 1.0
    for i in instances:
        best = min(
            (data[m][i].time for m in methods
             if data[m].get(i) and data[m][i].solved and data[m][i].time is not None),
            default=None,
        )
        if best is None or best <= 0:
            best = None
        for m in methods:
            rec = data[m].get(i)
            if rec and rec.solved and rec.time is not None and best:
                r = rec.time / best
                ratios[m].append(r)
                max_ratio = max(max_ratio, r)
            else:
                ratios[m].append(float("inf"))

    taus = sorted({1.0 + k * (max_ratio - 1.0) / 500 for k in range(501)} | {max_ratio})

    # --- RIGHT: gap CDF at time limit ---
    final_gaps = {m: [] for m in methods}
    max_gap_pct = 0.0
    for i in instances:
        for m in methods:
            rec = data[m].get(i)
            if rec and rec.solved:
                final_gaps[m].append(0.0)
            elif rec and rec.has_primal and rec.gap is not None and math.isfinite(rec.gap):
                g = rec.gap * 100.0
                final_gaps[m].append(g)
                max_gap_pct = max(max_gap_pct, g)
            else:
                final_gaps[m].append(float("inf"))

    max_gap_pct = min(max(max_gap_pct * 1.05, 1.0), 100.0)
    gap_xs = sorted({g * max_gap_pct / 500 for g in range(501)} | {max_gap_pct})

    # Per-method style: (color, linestyle, linewidth)
    # GRB = Wong blue; mono = black; four Benders = grey shades + distinct dashes
    STYLES = {
        "GRB":             ("#0072B2", "solid",   1.8),
        "mono":            ("#000000", "solid",   1.8),
        "benders_full":    ("#444444", "solid",   1.5),
        "benders_partial": ("#666666", "dashed",  1.5),
        "benders_2rand":   ("#888888", "dashdot", 1.5),
        "benders_adap_sp": ("#AAAAAA", "dotted",  1.5),
    }
    PUB_LABELS = {
        "GRB":             r"Gurobi",
        "mono":            r"SCIP compact",
        "benders_full":    r"SCIP-Benders (full)",
        "benders_partial": r"SCIP-Benders (partial, $\lceil m/4\rceil$)",
        "benders_2rand":   r"SCIP-Benders (2\,rand)",
        "benders_adap_sp": r"SCIP-Benders (adap.\ SP)",
    }

    rc = {
        "text.usetex":     True,
        "font.family":     "serif",
        "axes.labelsize":  11,
        "font.size":       10,
        "legend.fontsize": 9,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
    }
    with plt.rc_context(rc):
        fig, (ax1, ax2) = plt.subplots(
            1, 2, sharey=True, figsize=(8, 4.2),
            gridspec_kw={"width_ratios": [1, 1], "wspace": 0},
        )
        handles = []

        for m in methods:
            c, ls, lw = STYLES.get(m, ("#999999", "solid", 1.5))
            label = PUB_LABELS.get(m, m)

            # Left panel — time profile
            ys_t = [sum(1 for r in ratios[m] if r <= t) / n for t in taus]
            (ln,) = ax1.step(taus, ys_t, where="post",
                             color=c, ls=ls, lw=lw, label=label, zorder=2)
            handles.append(ln)

            # Right panel — gap CDF
            ys_g = [sum(1 for g in final_gaps[m] if g <= gx) / n for gx in gap_xs]
            ax2.step(gap_xs, ys_g, where="post",
                     color=c, ls=ls, lw=lw, zorder=2)

            # Connector so lines are visually uninterrupted at the panel boundary
            y_bound = sum(1 for r in ratios[m] if math.isfinite(r)) / n
            fig.add_artist(ConnectionPatch(
                xyA=(max_ratio, y_bound), coordsA="data", axesA=ax1,
                xyB=(0.0,       y_bound), coordsB="data", axesB=ax2,
                color=c, ls=ls, lw=lw, zorder=2, clip_on=False,
            ))

        # Dashed separator drawn behind data lines (zorder 1 < data zorder 2)
        ax2.axvline(0.0, color="black", ls="--", lw=1.2, zorder=1, clip_on=False)

        ax1.set_xlabel(r"$\tau$ (time\,/\,best time)")
        ax1.set_ylabel(r"fraction of instances")
        ax2.set_xlabel(r"optimality gap at time limit (\%)")

        ax1.set_ylim(0, 1.02)
        ax1.set_xlim(1.0, max_ratio)    # step extends naturally to exact xlim right
        ax2.set_xlim(0.0, max_gap_pct)  # step starts at exact xlim left

        ax1.grid(True, alpha=0.25, zorder=0)
        ax2.grid(True, alpha=0.25, zorder=0)

        ax1.spines["right"].set_visible(False)
        ax2.spines["left"].set_visible(False)
        ax2.tick_params(left=False)

        ax1.legend(handles=handles,
                   labels=[PUB_LABELS.get(m, m) for m in methods],
                   loc="upper left", fontsize=9, framealpha=0.95, edgecolor="0.75")

        ax2.text(1.0, 0.0, rf"$n = {n}$",
                 transform=ax2.transAxes,
                 ha="right", va="bottom", fontsize=9,
                 color="0.4")

        plt.tight_layout()
        for ext in (".png", ".pdf"):
            plt.savefig(out_png.replace(".png", ext), dpi=300, bbox_inches="tight")
        plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=RESULTS, help="dir with per-instance CSVs")
    ap.add_argument("--tl", type=float, default=600.0, help="time limit (s)")
    ap.add_argument("--out", default=os.path.join(PLOTS, "comparison_report.md"))
    ap.add_argument("--profile", default=os.path.join(PLOTS, "perf_profile.png"))
    args = ap.parse_args()

    os.makedirs(PLOTS, exist_ok=True)
    data = load_results(args.results)
    data = {m: data[m] for m in METHOD_ORDER if m in data and data[m]}

    instances = sorted(set().union(*[set(d.keys()) for d in data.values()]))
    if not instances:
        print("No data found. Run compare_run.py first.")
        return

    table, common = build_table(data, instances, args.tl)
    wilco = wilcoxon_block(data, instances, args.tl)

    # Full profile (all methods), and always a SCIP-only profile (no GRB) since
    # GRB is a different solver/machine and would otherwise set the baseline.
    perf_profile(data, instances, args.tl, args.profile)
    profiles = [os.path.relpath(args.profile, ROOT)]
    data_nogrb = {m: d for m, d in data.items() if m != "GRB"}
    if data_nogrb:
        nomgrb_png = args.profile.replace(".png", "_noGRB.png")
        perf_profile(data_nogrb, instances, args.tl, nomgrb_png)
        profiles.append(os.path.relpath(nomgrb_png, ROOT))

    report = (f"# SCIP/Gurobi comparison ({len(instances)} instances, TL={args.tl:.0f}s)\n\n"
              f"Methods: {', '.join(LABELS[m] for m in data)}\n\n"
              f"## Aggregates on solved instances\n\n{table}\n\n{wilco}\n\n"
              f"Performance profiles:\n" + "".join(f"- {p}\n" for p in profiles))
    with open(args.out, "w") as f:
        f.write(report)
    print(report)
    print(f"\nwrote {args.out}")
    for p in profiles:
        print(f"wrote {p} (+ .pdf)")


if __name__ == "__main__":
    main()
