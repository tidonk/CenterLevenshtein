"""3-D scatter: string length (n) x number of strings (m) x speedup vs SCIP mono.

One plot per Benders variant. Only instances where both mono and the method
solved to optimality are included. The reference plane at speedup=1 is drawn
for orientation.

Outputs:
  plots/corr_3d.pdf / .png
"""
import argparse
import csv
import glob
import os
import sys

import numpy as np
import plotly.graph_objects as go

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))

from scip_common import read_instance, instance_dims  # noqa: E402

RESULTS  = os.path.join(ROOT, "results")
PLOTS    = os.path.join(ROOT, "plots")
INST_DIRS = [os.path.join(ROOT, "random"), os.path.join(ROOT, "ecc")]

# mono is the baseline; only the two Benders variants are plotted
METHODS = ["benders_full", "benders_2rand"]
LABELS = {
    "benders_full":  "SCIP-Benders",
    "benders_2rand": "SCIP-Benders (partial)",
}
COLORS = {
    "benders_full":  "#2ca02c",
    "benders_2rand": "#9467bd",
}


# ---------------------------------------------------------------------------
# Load actual instance dimensions
# ---------------------------------------------------------------------------
def load_dims():
    """Return {basename: (n, m)} reading every instance file once."""
    dims = {}
    for d in INST_DIRS:
        for path in glob.glob(os.path.join(d, "*.txt")):
            base = os.path.basename(path)
            try:
                m, sigma = read_instance(path)
                n, _, _ = instance_dims(sigma)
                dims[base] = (n, m)
            except Exception:
                pass
    return dims


# ---------------------------------------------------------------------------
# Load results
# ---------------------------------------------------------------------------
def load_results(results_dir):
    """Return list of dicts: {method, instance, time, solved}."""
    rows = []
    for path in sorted(glob.glob(os.path.join(results_dir, "*.csv"))):
        if os.path.basename(path) == "result_path_binary_2n.csv":
            continue
        with open(path) as f:
            for row in csv.DictReader(f):
                if "method" not in row:
                    break
                try:
                    t = float(row["time"]) if row.get("time") else None
                except ValueError:
                    t = None
                rows.append({
                    "method":  row["method"],
                    "instance": os.path.basename(row["instance"]),
                    "time":    t,
                    "solved":  row.get("status") == "optimal",
                })
    return rows


# ---------------------------------------------------------------------------
# Build interactive 3-D scatter: n x m x speedup  (plotly → HTML)
# ---------------------------------------------------------------------------
def make_3d_plot(rows, dims, out_base):
    mono_time = {r["instance"]: r["time"]
                 for r in rows
                 if r["method"] == "mono" and r["solved"] and r["time"] is not None}

    traces = []
    all_n, all_m, all_s = [], [], []

    for method in METHODS:
        ns, ms, speedups, hover = [], [], [], []
        for r in rows:
            if r["method"] != method or not r["solved"] or r["time"] is None:
                continue
            inst = r["instance"]
            if inst not in mono_time or inst not in dims:
                continue
            n, m = dims[inst]
            sp = mono_time[inst] / r["time"]
            ns.append(n); ms.append(m); speedups.append(sp)
            hover.append(f"{inst}<br>n={n}, m={m}<br>speedup={sp:.2f}x"
                         f"<br>t={r['time']:.2f}s  mono={mono_time[inst]:.2f}s")
            all_n.append(n); all_m.append(m); all_s.append(sp)

        traces.append(go.Scatter3d(
            x=ns, y=ms, z=speedups,
            mode="markers",
            marker=dict(size=5, color=COLORS[method], opacity=0.8),
            name=LABELS[method],
            text=hover,
            hovertemplate="%{text}<extra></extra>",
        ))

    # Reference plane at speedup = 1
    if all_n and all_m:
        n0, n1 = min(all_n), max(all_n)
        m0, m1 = min(all_m), max(all_m)
        traces.append(go.Surface(
            x=[[n0, n1], [n0, n1]],
            y=[[m0, m0], [m1, m1]],
            z=[[1, 1], [1, 1]],
            opacity=0.15,
            colorscale=[[0, "gray"], [1, "gray"]],
            showscale=False,
            hoverinfo="skip",
            name="Baseline (mono)",
        ))

    fig = go.Figure(data=traces)
    fig.update_layout(
        scene=dict(
            xaxis_title="String length n",
            yaxis_title="Number of strings m",
            zaxis_title="Speedup vs. SCIP mono",
        ),
        legend=dict(x=0.01, y=0.99),
        margin=dict(l=0, r=0, t=30, b=0),
        title="Speedup of Benders decomposition over SCIP mono",
    )

    out_html = f"{out_base}.html"
    fig.write_html(out_html, include_plotlyjs="cdn")
    print(f"wrote {out_html}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=RESULTS)
    ap.add_argument("--tl", type=float, default=600.0)
    ap.add_argument("--out", default=PLOTS)
    args = ap.parse_args()

    rows = load_results(args.results)
    if not rows:
        print(f"No results found under {args.results}")
        return

    dims = load_dims()
    os.makedirs(args.out, exist_ok=True)

    make_3d_plot(rows, dims, os.path.join(args.out, "corr_3d"))
    print("Open plots/corr_3d.html in a browser to interact.")


if __name__ == "__main__":
    main()
