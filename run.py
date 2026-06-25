"""Run a SINGLE instance through all methods (GRB reference + the three SCIP
configs). Writes the full per-method solver logs to logs/<INSTANCE>/<METHOD>.log
and one solution CSV results/<INSTANCE>.csv (header + one row per method).

This is the unit of work for the cluster: submit one job per instance.

Usage:
    python run.py <instance> [time_limit] [method1 method2 ...]

    <instance>    path or bare name, e.g. random/I_10_30_0.txt or I_10_30_0.txt
    time_limit    seconds (default 600)
    methods       subset of: GRB mono benders_full benders_partial (default all)

Example:
    python run.py I_15_20_0.txt 600
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))
from run_methods import run_instance, ALL_METHODS  # noqa: E402


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    inst = sys.argv[1]
    time_limit = float(sys.argv[2]) if len(sys.argv) > 2 else 600
    methods = sys.argv[3:] if len(sys.argv) > 3 else ALL_METHODS

    rows, out = run_instance(inst, time_limit=time_limit, methods=methods)
    for r in rows:
        t = f"{r['time']:.2f}s" if r["time"] is not None else "-"
        print(f"  {r['method']:16s} status={str(r['status']):10s} "
              f"primal={r['primal']} true_cost={r['true_cost']} "
              f"bound={r['bound']} gap={r['gap']} time={t}")
    print(f"solution csv -> {out}")


if __name__ == "__main__":
    main()
