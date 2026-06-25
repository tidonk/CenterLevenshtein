"""Run the full test set (all methods) by looping run_instance over every
instance. For each instance it writes the per-method logs (logs/<INSTANCE>/) and
one solution CSV results/<INSTANCE>.csv.

Idempotent: an instance whose results/<INSTANCE>.csv already exists is skipped,
so the job can be resumed / run incrementally on the cluster.

Usage:
    python compare_run.py [time_limit] [--lengths 5 10 ...] [--force]
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))
from run_methods import run_instance, resolve_instance, RESULTS  # noqa: E402

NUMBERS = [10, 20, 30, 40, 50]
ITS = [0, 1, 2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("time_limit", nargs="?", type=float, default=600)
    ap.add_argument("--lengths", type=int, nargs="+", default=[5, 10, 15, 20])
    ap.add_argument("--force", action="store_true", help="re-run even if csv exists")
    args = ap.parse_args()

    os.makedirs(RESULTS, exist_ok=True)
    for length in args.lengths:
        for number in NUMBERS:
            for it in ITS:
                base = f"I_{length}_{number}_{it}.txt"
                if not os.path.exists(resolve_instance(base)):
                    continue
                out = os.path.join(RESULTS, base.replace(".txt", ".csv"))
                if os.path.exists(out) and not args.force:
                    print(f"skip {base} (exists)", flush=True)
                    continue
                rows, out = run_instance(base, time_limit=args.time_limit)
                summ = "  ".join(
                    f"{r['method']}={r['time']:.1f}s/{r['status']}"
                    if r["time"] is not None else f"{r['method']}=NA"
                    for r in rows)
                print(f"{base:18s} {summ}", flush=True)
    print("compare_run complete.")


if __name__ == "__main__":
    main()
