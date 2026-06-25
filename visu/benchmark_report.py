"""Generate per-instance comparison tables and grouped bar charts.

This script compares three approaches across the instances present in both
benchmark CSVs:

- GRB: result_hayashida_binary_sum.csv (uses SolutionTime / GAP)
- SCIP: bench_scip.csv, mono_time / mono_obj vs bend_true_cost
- SCIP-Benders: bench_scip.csv, bend_time / bend_obj vs bend_true_cost

The output is intentionally dependency-free (standard library only) so it can
run in the current repository environment without matplotlib/pandas/scipy.

Outputs:
  - an SVG grouped bar chart
  - a markdown table with per-family values and summary rows
  - a text summary with Wilcoxon signed-rank tests (normal approximation)

The benchmark files in this repository contain repeated seeds for the GRB run
(`I_5_10_0.txt`, `I_5_10_1.txt`, ...), while the SCIP file stores a single row
per instance family (`random/I_5_10_0.txt`, ...). To make the comparison
meaningful, the script groups rows by instance family (e.g. `I_5_10`) and
aggregates repeated GRB seeds within each family.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple


TOL = 1e-6


@dataclass(frozen=True)
class Cell:
    """A rendered cell value.

    `display` is what appears in the table, while `numeric` is used for the
    bar chart and the Wilcoxon-style score calculations.
    """

    display: str
    numeric: float
    is_time: bool
    optimal: bool


MISSING_CELL = Cell(display="--", numeric=0.0, is_time=False, optimal=False)


def normalize_instance(name: str) -> str:
    return os.path.basename(name.strip())


def family_key(name: str) -> str:
    """Collapse seed-specific filenames to a common family label.

    Examples:
      - `I_5_10_0.txt` -> `I_5_10`
      - `random/I_5_10_0.txt` -> `I_5_10`
    """

    base = normalize_instance(name)
    if base.endswith(".txt"):
        base = base[:-4]
    parts = base.split("_")
    if len(parts) >= 2 and parts[-1].isdigit():
        return "_".join(parts[:-1])
    return base


def parse_float(value: str) -> Optional[float]:
    if value is None:
        return None
    value = value.strip()
    if not value:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def is_close(a: Optional[float], b: Optional[float], tol: float = TOL) -> bool:
    return a is not None and b is not None and abs(a - b) <= tol


def load_grb(path: str) -> Dict[str, List[dict]]:
    rows: Dict[str, List[dict]] = {}
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if not row:
                continue
            inst_raw = row.get("Instance", "")
            if not inst_raw or inst_raw == "Instance":
                continue
            inst = family_key(inst_raw)
            rows.setdefault(inst, []).append(row)
    return rows


def load_scip(path: str) -> Dict[str, List[dict]]:
    rows: Dict[str, List[dict]] = {}
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if not row:
                continue
            inst_raw = row.get("instance", "")
            if not inst_raw:
                continue
            inst = family_key(inst_raw)
            rows.setdefault(inst, []).append(row)
    return rows


def grb_cell(row: dict) -> Cell:
    time_v = parse_float(row.get("SolutionTime", ""))
    gap_v = parse_float(row.get("GAP", ""))
    optimal = gap_v is not None and abs(gap_v) <= TOL
    if optimal:
        assert time_v is not None
        return Cell(display=f"{time_v:.3f}s", numeric=time_v, is_time=True, optimal=True)
    # Report the terminal gap when the instance is not solved to optimality.
    gap_v = 0.0 if gap_v is None else max(0.0, gap_v)
    return Cell(display=f"g={gap_v:.3g}", numeric=gap_v, is_time=False, optimal=False)


def scip_cell(row: dict, time_key: str, obj_key: str) -> Cell:
    time_v = parse_float(row.get(time_key, ""))
    obj_v = parse_float(row.get(obj_key, ""))
    true_cost_v = parse_float(row.get("bend_true_cost", ""))
    if time_v is None or obj_v is None or true_cost_v is None:
        raise ValueError(f"Missing SCIP values for {row}")

    optimal = is_close(obj_v, true_cost_v)
    if optimal:
        return Cell(display=f"{time_v:.3f}s", numeric=time_v, is_time=True, optimal=True)

    # Approximate a terminal gap from objective error. This keeps the cell in a
    # common scale for the chart and makes the comparison monotone.
    if abs(true_cost_v) <= TOL:
        gap = abs(obj_v - true_cost_v)
    else:
        gap = abs(obj_v - true_cost_v) / abs(true_cost_v)
    gap = max(0.0, gap)
    return Cell(display=f"g={gap:.3g}", numeric=gap, is_time=False, optimal=False)


def aggregate_cells(cells: Sequence[Cell], label: str) -> Cell:
    """Aggregate repeated rows for the same family.

    If every row is optimal, the family is optimal and we report the mean time.
    Otherwise we report the mean terminal value, which is a gap for the
    time-limited rows and can be interpreted as a family-level penalty.
    """

    if not cells:
        raise ValueError(f"No rows to aggregate for {label}")
    optimal = all(cell.optimal for cell in cells)
    numeric = sum(cell.numeric for cell in cells) / len(cells)
    if optimal:
        return Cell(display=f"{numeric:.3f}s", numeric=numeric, is_time=True, optimal=True)
    return Cell(display=f"g={numeric:.3g}", numeric=numeric, is_time=False, optimal=False)


def performance_score(cell: Cell, raw_time: float) -> float:
    """Score used for pairwise ranking tests.

    Optimal runs are scored by solve time; time-limited runs are penalized by
    adding the terminal gap to the solve time.
    """

    return raw_time if cell.optimal else raw_time + cell.numeric


def geometric_mean(values: Sequence[float]) -> Optional[float]:
    vals = [v for v in values if v > 0]
    if not vals:
        return None
    return math.exp(sum(math.log(v) for v in vals) / len(vals))


def rank_abs_diffs(diffs: Sequence[float]) -> List[float]:
    """Average ranks for absolute differences, ascending.

    Returns ranks aligned with `diffs`.
    """

    indexed = sorted(enumerate(abs(d) for d in diffs), key=lambda t: t[1])
    ranks = [0.0] * len(diffs)
    i = 0
    rank = 1
    while i < len(indexed):
        j = i
        v = indexed[i][1]
        while j < len(indexed) and abs(indexed[j][1] - v) <= TOL:
            j += 1
        avg_rank = (rank + (rank + (j - i) - 1)) / 2.0
        for k in range(i, j):
            ranks[indexed[k][0]] = avg_rank
        rank += j - i
        i = j
    return ranks


def wilcoxon_signed_rank(x: Sequence[float], y: Sequence[float]) -> Tuple[float, float, int]:
    """Wilcoxon signed-rank test with a normal approximation.

    Returns (W, p_value, n_effective). Smaller W means stronger evidence of a
    difference. p_value is two-sided.
    """

    diffs = [a - b for a, b in zip(x, y) if abs(a - b) > TOL]
    n = len(diffs)
    if n == 0:
        return 0.0, 1.0, 0

    ranks = rank_abs_diffs(diffs)
    w_pos = sum(r for r, d in zip(ranks, diffs) if d > 0)
    w_neg = sum(r for r, d in zip(ranks, diffs) if d < 0)
    w = min(w_pos, w_neg)

    mean = n * (n + 1) / 4.0
    # Tie-corrected variance (normal approximation).
    abs_vals = [abs(d) for d in diffs]
    counts: Dict[float, int] = {}
    for v in abs_vals:
        key = round(v / TOL) * TOL
        counts[key] = counts.get(key, 0) + 1
    tie_correction = sum(c**3 - c for c in counts.values() if c > 1)
    var = (n * (n + 1) * (2 * n + 1) - tie_correction / 2.0) / 24.0
    if var <= 0:
        return w, 1.0, n
    z = (w - mean + 0.5) / math.sqrt(var)
    # Two-sided p from the standard normal distribution using erf.
    p = math.erfc(abs(z) / math.sqrt(2.0))
    return w, p, n


def format_score_cell(cell: Cell) -> str:
    return cell.display


def write_markdown_table(
    rows: Sequence[str],
    cols: Sequence[str],
    cells: Dict[Tuple[str, str], Cell],
    summary: Dict[str, Dict[str, Optional[float]]],
    path: str,
    optimal_instances: Sequence[str],
    only_in_grb: Sequence[str],
    only_in_scip: Sequence[str],
) -> str:
    lines: List[str] = []
    lines.append("# Benchmark comparison report")
    lines.append("")
    lines.append("## Coverage")
    lines.append(f"- Families in GRB CSV: {len(rows) + len(only_in_grb)}")
    lines.append(f"- Families in SCIP CSV: {len(rows) + len(only_in_scip)}")
    lines.append(f"- Families in both CSVs: {len(rows)}")
    lines.append("")
    if only_in_grb:
        lines.append("### Only in GRB")
        lines.append(", ".join(only_in_grb))
        lines.append("")
    if only_in_scip:
        lines.append("### Only in SCIP")
        lines.append(", ".join(only_in_scip))
        lines.append("")

    lines.append("| instance family | " + " | ".join(cols) + " |")
    lines.append("|---|" + "---|" * len(cols))
    for inst in rows:
        values = [cells.get((inst, col), MISSING_CELL).display for col in cols]
        lines.append("| " + inst + " | " + " | ".join(values) + " |")

    lines.append("| **sum (optimal-only)** | " + " | ".join(
        f"{summary[c]['sum']:.3f}s" if summary[c]['sum'] is not None else "—"
        for c in cols
    ) + " |")
    lines.append("| **mean (optimal-only)** | " + " | ".join(
        f"{summary[c]['mean']:.3f}s" if summary[c]['mean'] is not None else "—"
        for c in cols
    ) + " |")
    lines.append("| **geo mean (optimal-only)** | " + " | ".join(
        f"{summary[c]['geo_mean']:.3f}s" if summary[c]['geo_mean'] is not None else "—"
        for c in cols
    ) + " |")
    lines.append("| **optimal families used** | " + " | ".join(
        f"{summary[c]['n_opt']:.0f}" if summary[c]['n_opt'] is not None else "0"
        for c in cols
    ) + " |")

    lines.append("")
    lines.append(f"Optimal-only footer computed on: {', '.join(optimal_instances) if optimal_instances else 'none'}")
    text = "\n".join(lines) + "\n"
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    return text


def escape_svg(text: str) -> str:
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def svg_rect(x: float, y: float, w: float, h: float, fill: str, opacity: float = 1.0) -> str:
    return f'<rect x="{x:.2f}" y="{y:.2f}" width="{w:.2f}" height="{h:.2f}" fill="{fill}" fill-opacity="{opacity:.3f}" />'


def write_svg_chart(
    instances: Sequence[str],
    cols: Sequence[str],
    cells: Dict[Tuple[str, str], Cell],
    path: str,
    title: str,
) -> None:
    width = max(980, 160 * len(instances) + 120)
    height = 560
    margin_left = 70
    margin_right = 30
    margin_top = 70
    margin_bottom = 120
    plot_w = width - margin_left - margin_right
    plot_h = height - margin_top - margin_bottom

    vals = [cells.get((inst, col), MISSING_CELL).numeric for inst in instances for col in cols]
    max_val = max(vals) if vals else 1.0
    max_val = max(max_val, 1e-6)

    colors = {
        "GRB": "#4c78a8",
        "SCIP": "#f58518",
        "SCIP-Benders": "#54a24b",
    }
    series_width = 18
    gap_within = 8
    group_gap = 38
    bars_per_group = len(cols)
    group_w = bars_per_group * series_width + (bars_per_group - 1) * gap_within
    step = group_w + group_gap

    def y(v: float) -> float:
        return margin_top + plot_h - (v / max_val) * plot_h

    def fmt_tick(v: float) -> str:
        if v >= 10:
            return f"{v:.0f}"
        if v >= 1:
            return f"{v:.1f}"
        return f"{v:.2f}"

    svg: List[str] = []
    svg.append('<?xml version="1.0" encoding="UTF-8" standalone="no"?>')
    svg.append(f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">')
    svg.append('<rect x="0" y="0" width="100%" height="100%" fill="white"/>')
    svg.append(f'<text x="{width/2:.1f}" y="32" text-anchor="middle" font-family="sans-serif" font-size="20" font-weight="bold">{escape_svg(title)}</text>')
    svg.append(f'<text x="{width/2:.1f}" y="52" text-anchor="middle" font-family="sans-serif" font-size="12" fill="#555">Solid bars = optimal (solve time); translucent bars = timelimited (gap value)</text>')

    # Axes and grid.
    for i in range(6):
        v = max_val * i / 5.0
        yy = y(v)
        svg.append(f'<line x1="{margin_left}" y1="{yy:.2f}" x2="{width - margin_right}" y2="{yy:.2f}" stroke="#e6e6e6"/>')
        svg.append(f'<text x="{margin_left - 8}" y="{yy + 4:.2f}" text-anchor="end" font-family="sans-serif" font-size="11" fill="#555">{fmt_tick(v)}</text>')

    svg.append(f'<line x1="{margin_left}" y1="{margin_top}" x2="{margin_left}" y2="{margin_top + plot_h}" stroke="#444"/>')
    svg.append(f'<line x1="{margin_left}" y1="{margin_top + plot_h}" x2="{width - margin_right}" y2="{margin_top + plot_h}" stroke="#444"/>')

    # Bars.
    for gi, inst in enumerate(instances):
        group_x = margin_left + gi * step + 12
        label_x = group_x + group_w / 2.0
        for ci, col in enumerate(cols):
            cell = cells.get((inst, col), MISSING_CELL)
            x = group_x + ci * (series_width + gap_within)
            bar_h = 0 if max_val <= 0 else (cell.numeric / max_val) * plot_h
            yy = margin_top + plot_h - bar_h
            fill = colors.get(col, "#888")
            opacity = 1.0 if cell.optimal else 0.55
            if cell is not MISSING_CELL:
                svg.append(svg_rect(x, yy, series_width, bar_h, fill, opacity))
                svg.append(
                    f'<text x="{x + series_width/2:.2f}" y="{yy - 4:.2f}" text-anchor="middle" '
                    f'font-family="sans-serif" font-size="10" fill="#222">{escape_svg(cell.display)}</text>'
                )
            else:
                svg.append(
                    f'<text x="{x + series_width/2:.2f}" y="{margin_top + plot_h - 6:.2f}" text-anchor="middle" '
                    f'font-family="sans-serif" font-size="10" fill="#777">--</text>'
                )
        # rotated instance label
        svg.append(
            f'<text x="{label_x:.2f}" y="{margin_top + plot_h + 20}" text-anchor="end" '
            f'transform="rotate(-35 {label_x:.2f},{margin_top + plot_h + 20})" '
            f'font-family="sans-serif" font-size="11" fill="#333">{escape_svg(inst)}</text>'
        )

    # Legend.
    legend_y = height - 78
    legend_items = [("GRB", colors["GRB"]), ("SCIP", colors["SCIP"]), ("SCIP-Benders", colors["SCIP-Benders"])]
    lx = margin_left
    for label, color in legend_items:
        svg.append(svg_rect(lx, legend_y - 12, 14, 14, color, 1.0))
        svg.append(f'<text x="{lx + 20}" y="{legend_y}" font-family="sans-serif" font-size="12">{escape_svg(label)}</text>')
        lx += 130
    svg.append(svg_rect(lx, legend_y - 12, 14, 14, "#999", 0.55))
    svg.append(f'<text x="{lx + 20}" y="{legend_y}" font-family="sans-serif" font-size="12">timelimited cell</text>')

    svg.append('</svg>')
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(svg))


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate benchmark comparison report.")
    parser.add_argument("--grb", default="result_path_binary_2n.csv", help="GRB results CSV")
    parser.add_argument("--scip", default="bench_scip_heur_600.csv", help="SCIP benchmark CSV")
    parser.add_argument("--out-prefix", default="benchmark_report", help="Output file prefix")
    args = parser.parse_args()

    grb_rows = load_grb(args.grb)
    scip_rows = load_scip(args.scip)

    cols = ["GRB", "SCIP", "SCIP-Benders"]
    cells: Dict[Tuple[str, str], Cell] = {}

    # Build the common family set.
    all_families = sorted(set(grb_rows) | set(scip_rows))
    if not all_families:
        raise SystemExit("No instances were found in either CSV.")

    common_families = sorted(set(grb_rows) & set(scip_rows))

    only_in_grb = sorted(set(grb_rows) - set(scip_rows))
    only_in_scip = sorted(set(scip_rows) - set(grb_rows))

    for fam in all_families:
        if fam in grb_rows:
            grb_group = [grb_cell(r) for r in grb_rows[fam]]
            cells[(fam, "GRB")] = aggregate_cells(grb_group, fam)
        if fam in scip_rows:
            scip_group = [scip_cell(r, "mono_time", "mono_obj") for r in scip_rows[fam]]
            bend_group = [scip_cell(r, "bend_time", "bend_obj") for r in scip_rows[fam]]
            cells[(fam, "SCIP")] = aggregate_cells(scip_group, fam)
            cells[(fam, "SCIP-Benders")] = aggregate_cells(bend_group, fam)

    # Footer summaries over the subset solved to optimality by all approaches.
    optimal_instances = [
        fam for fam in all_families
        if (fam, "GRB") in cells and (fam, "SCIP") in cells and (fam, "SCIP-Benders") in cells
        and cells[(fam, "GRB")].optimal and cells[(fam, "SCIP")].optimal and cells[(fam, "SCIP-Benders")].optimal
    ]

    summary: Dict[str, Dict[str, Optional[float]]] = {}
    for col in cols:
        times = [cells[(fam, col)].numeric for fam in optimal_instances]
        summary[col] = {
            "sum": sum(times) if times else None,
            "mean": (sum(times) / len(times)) if times else None,
            "geo_mean": geometric_mean(times),
            "n_opt": float(len(times)) if times else 0.0,
        }

    # Write outputs.
    table_path = f"{args.out_prefix}.md"
    svg_path = f"{args.out_prefix}.svg"
    txt_path = f"{args.out_prefix}.txt"

    table_text = write_markdown_table(
        all_families,
        cols,
        cells,
        summary,
        table_path,
        optimal_instances,
        only_in_grb,
        only_in_scip,
    )
    write_svg_chart(all_families, cols, cells, svg_path, "Per-family comparison: GRB vs SCIP vs SCIP-Benders")

    # Pairwise Wilcoxon summaries.
    # Use the score = time if optimal else time + gap.
    pair_lines: List[str] = []
    pair_lines.append("Wilcoxon signed-rank tests (paired, two-sided; normal approximation).")
    pair_lines.append("Score used: solve time if optimal, otherwise solve time + terminal gap.")
    pair_lines.append("")
    for a, b in [("GRB", "SCIP"), ("GRB", "SCIP-Benders"), ("SCIP", "SCIP-Benders")]:
        xs: List[float] = []
        ys: List[float] = []
        for fam in common_families:
            ca = cells[(fam, a)]
            cb = cells[(fam, b)]
            # raw time is the displayed time if optimal, otherwise the terminal time
            # limit (approximated by the same numeric field for the CSVs we have).
            # For SCIP the underlying time is stored in the CSV columns; for GRB it
            # is SolutionTime.
            if a == "GRB":
                raw_a = sum(parse_float(r.get("SolutionTime", "")) or 0.0 for r in grb_rows[fam]) / len(grb_rows[fam])
            elif a == "SCIP":
                raw_a = sum(parse_float(r.get("mono_time", "")) or 0.0 for r in scip_rows[fam]) / len(scip_rows[fam])
            else:
                raw_a = sum(parse_float(r.get("bend_time", "")) or 0.0 for r in scip_rows[fam]) / len(scip_rows[fam])
            if b == "GRB":
                raw_b = sum(parse_float(r.get("SolutionTime", "")) or 0.0 for r in grb_rows[fam]) / len(grb_rows[fam])
            elif b == "SCIP":
                raw_b = sum(parse_float(r.get("mono_time", "")) or 0.0 for r in scip_rows[fam]) / len(scip_rows[fam])
            else:
                raw_b = sum(parse_float(r.get("bend_time", "")) or 0.0 for r in scip_rows[fam]) / len(scip_rows[fam])
            xs.append(performance_score(ca, raw_a))
            ys.append(performance_score(cb, raw_b))
        w, p, n = wilcoxon_signed_rank(xs, ys)
        pair_lines.append(f"{a} vs {b}: W={w:.3f}, p={p:.4g}, n={n}")

    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(table_text)
        f.write("\n\n")
        f.write("\n".join(pair_lines))
        f.write("\n")

    print(f"Wrote {svg_path}")
    print(f"Wrote {table_path}")
    print(f"Wrote {txt_path}")
    print(f"Common families: {len(common_families)}")
    print(f"Optimal-only families: {len(optimal_instances)}")
    for line in pair_lines:
        print(line)


if __name__ == "__main__":
    main()