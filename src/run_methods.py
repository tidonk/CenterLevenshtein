"""Shared experiment driver.

Layout (anchored at the repo root, so it works from any cwd / on the cluster):
  logs/<INSTANCE>/<METHOD>.log   full solver log per (instance, method)
  results/<INSTANCE>.csv         single instance's solution rows (header + one
                                 row per method)
  plots/                         tables and performance profiles (visu/)

Methods:
  GRB             Gurobi monolith -- NOT solved here (no gurobipy); the row and
                  log are emitted from the recorded reference
                  results/result_path_binary_2n.csv
  mono            SCIP monolith
  benders_full    custom Benders, full decomposition (keep_in_master=0)
  benders_partial custom Benders, partial decomposition (retain ceil(m/4))
"""
import csv
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

LOGS = os.path.join(ROOT, "logs")
RESULTS = os.path.join(ROOT, "results")
PLOTS = os.path.join(ROOT, "plots")
RANDOM_DIR = os.path.join(ROOT, "random")
GRB_REF = os.path.join(RESULTS, "result_path_binary_2n.csv")

import scip_mono       # noqa: E402
import scip_benders    # noqa: E402
from scip_common import read_instance, instance_dims  # noqa: E402
from validate_benders import build_mono_fixed          # noqa: E402

PARTIAL_FRACTION = 0.25   # benders_partial retains ceil(m/4) blocks in master
ADAP_SP_FRAC     = 0.5    # benders_adap_sp solves 50% of subproblems per round
SOLVE_METHODS = ["mono", "benders_full", "benders_partial", "benders_2rand", "benders_adap_sp"]
ALL_METHODS = ["GRB"] + SOLVE_METHODS

FIELDS = ["instance", "length", "number", "it", "method", "status",
          "primal", "bound", "gap", "time", "nodes", "true_cost"]


def resolve_instance(inst):
    """Accept a path or a bare name; resolve against <root>/random if needed."""
    if os.path.exists(inst):
        return inst
    cand = os.path.join(RANDOM_DIR, os.path.basename(inst))
    return cand if os.path.exists(cand) else inst


def _dims_from_name(base):
    parts = base.replace(".txt", "").split("_")  # I_<len>_<num>_<it>
    g = lambda i: int(parts[i]) if len(parts) > i else None
    return g(1), g(2), g(3)


def instance_log_dir(base):
    return os.path.join(LOGS, base.replace(".txt", ""))


def _true_cost(inst, res, time_limit):
    if res.get("xvals") is None:
        return None
    _m, sigma = read_instance(inst)
    n, p, q = instance_dims(sigma)
    mod = build_mono_fixed(sigma, n, p, q, res["xvals"], res["uvals"])
    mod.setParam("limits/time", time_limit)
    mod.hideOutput()
    mod.optimize()
    return mod.getObjVal() if mod.getNSols() > 0 else None


# --------------------------------------------------------------------------
# GRB: emitted from the recorded reference (no local Gurobi)
# --------------------------------------------------------------------------
def _load_grb_ref():
    ref = {}
    if os.path.exists(GRB_REF):
        with open(GRB_REF) as f:
            for row in csv.DictReader(f):
                ref[os.path.basename(row["Instance"])] = row
    return ref


_GRB_REF = None


def grb_one(inst, log_dir=LOGS):
    """Return the GRB reference row (FIELDS) for inst and write its log."""
    global _GRB_REF
    if _GRB_REF is None:
        _GRB_REF = _load_grb_ref()
    base = os.path.basename(resolve_instance(inst))
    length, number, it = _dims_from_name(base)
    ref = _GRB_REF.get(base)
    ldir = instance_log_dir(base)
    os.makedirs(ldir, exist_ok=True)
    logp = os.path.join(ldir, "GRB.log")
    if ref is None:
        with open(logp, "w") as f:
            f.write(f"# GRB reference: instance {base} not found in {GRB_REF}\n")
        return {"instance": base, "length": length, "number": number, "it": it,
                "method": "GRB", "status": "missing", "primal": None,
                "bound": None, "gap": None, "time": None, "nodes": None,
                "true_cost": None}
    gap = float(ref["GAP"])
    primal = float(ref["Best Incumbent"])
    bound = float(ref["BestBound"])
    t = float(ref["SolutionTime"])
    status = "optimal" if gap <= 1e-6 else "timelimit"
    with open(logp, "w") as f:
        f.write("# Gurobi (gurobipy) reference result -- recorded from "
                f"{os.path.basename(GRB_REF)}\n# (not re-solved locally; no gurobipy)\n\n")
        for k, v in ref.items():
            f.write(f"{k:18s}: {v}\n")
    return {"instance": base, "length": length, "number": number, "it": it,
            "method": "GRB", "status": status, "primal": primal, "bound": bound,
            "gap": gap, "time": t, "nodes": float(ref.get("Nodes", "nan") or "nan"),
            "true_cost": primal}


# --------------------------------------------------------------------------
# SCIP solves
# --------------------------------------------------------------------------
def solve_one(inst, method, time_limit=600, seed=2025):
    """Solve one instance with one SCIP method; returns a row dict (FIELDS).
    The full solver log is written to logs/<INSTANCE>/<METHOD>.log."""
    inst = resolve_instance(inst)
    base = os.path.basename(inst)
    log_file = os.path.join(instance_log_dir(base), f"{method}.log")

    if method == "mono":
        r = scip_mono.solve_instance(inst, time_limit=time_limit, seed=seed,
                                     quiet=True, log_file=log_file)
        tcost = r["obj"]
    elif method == "benders_full":
        r = scip_benders.solve_instance(inst, time_limit=time_limit, seed=seed,
                                        quiet=True, custom=True, keep_in_master=0,
                                        log_file=log_file)
        tcost = _true_cost(inst, r, time_limit)
    elif method == "benders_partial":
        r = scip_benders.solve_instance(inst, time_limit=time_limit, seed=seed,
                                        quiet=True, custom=True,
                                        keep_in_master=PARTIAL_FRACTION,
                                        log_file=log_file)
        tcost = _true_cost(inst, r, time_limit)
    elif method == "benders_2rand":
        import random as _random
        m_count, _ = read_instance(resolve_instance(inst))
        rng = _random.Random(seed)
        keep = set(rng.sample(range(m_count), min(2, m_count)))
        r = scip_benders.solve_instance(inst, time_limit=time_limit, seed=seed,
                                        quiet=True, custom=True,
                                        keep_in_master=keep,
                                        log_file=log_file)
        tcost = _true_cost(inst, r, time_limit)
    elif method == "benders_adap_sp":
        r = scip_benders.solve_instance(inst, time_limit=time_limit, seed=seed,
                                        quiet=True, custom=True, keep_in_master=0,
                                        subprobfrac=ADAP_SP_FRAC,
                                        log_file=log_file)
        tcost = _true_cost(inst, r, time_limit)
    else:
        raise ValueError(f"unknown method: {method}")

    length, number, it = _dims_from_name(base)
    return {"instance": base, "length": length, "number": number, "it": it,
            "method": method, "status": r["status"], "primal": r["obj"],
            "bound": r["bound"], "gap": r["gap"], "time": r["solving_time"],
            "nodes": r["nodes"], "true_cost": tcost}


def run_instance(inst, time_limit=600, methods=ALL_METHODS, seed=2025):
    """Run all requested methods on one instance. Writes the per-method logs and
    a single solution CSV results/<INSTANCE>.csv (header + one row per method).
    Returns the list of rows."""
    base = os.path.basename(resolve_instance(inst))
    rows = []
    for method in methods:
        if method == "GRB":
            rows.append(grb_one(inst))
        else:
            rows.append(solve_one(inst, method, time_limit=time_limit, seed=seed))
    os.makedirs(RESULTS, exist_ok=True)
    out = os.path.join(RESULTS, base.replace(".txt", ".csv"))

    # Merge: keep any existing rows for methods we are NOT running now.
    existing = {}
    if os.path.exists(out):
        with open(out) as f:
            for row in csv.DictReader(f):
                existing[row["method"]] = row
    for r in rows:
        existing[r["method"]] = r
    merged = [existing[m] for m in ALL_METHODS if m in existing]

    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(merged)
    return rows, out
