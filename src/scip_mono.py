"""SCIP monolithic port of path_binary_digitalized.py.

Builds the whole median model in one SCIP Model and solves it. This is the
control / correctness gate for the Benders run.
"""
import os
import sys
import time

from pyscipopt import Model

from scip_common import (read_instance, instance_dims, add_x_vars, add_u_vars,
                         add_length_constraints, add_block, log_path)


def build_mono(sigma, n, p, q):
    model = Model("path_binary_mono")
    x = add_x_vars(model, n)
    u = add_u_vars(model, p, q)
    add_length_constraints(model, u, p, q)

    obj = 0
    for idx, word in enumerate(sigma):
        obj = obj + add_block(model, x, u, idx, word, n, p, q)
    model.setObjective(obj, "minimize")
    return model, x, u


def solve_instance(path, time_limit=600, seed=2025, quiet=True, log_dir="logs",
                   log_label="mono", log_file=None):
    m, sigma = read_instance(path)
    n, p, q = instance_dims(sigma)
    model, x, u = build_mono(sigma, n, p, q)

    model.setParam("limits/time", time_limit)
    model.setParam("randomization/randomseedshift", seed)
    lf = log_file or (log_path(path, log_label, time_limit, log_dir) if log_dir else None)
    if lf:
        os.makedirs(os.path.dirname(lf), exist_ok=True)
        model.setLogfile(lf)
    if quiet:
        model.hideOutput()

    t0 = time.time()
    model.optimize()
    wall = time.time() - t0

    if lf:
        # Append SCIP's full statistics (incl. the additional per-plugin tables)
        # to the per-run log.
        model.printStatistics()

    res = {
        "instance": path,
        "status": model.getStatus(),
        "obj": model.getObjVal() if model.getNSols() > 0 else None,
        "bound": model.getDualbound(),
        "gap": model.getGap(),
        "nodes": model.getNNodes(),
        "lp_iters": model.getNLPIterations(),
        "solving_time": model.getSolvingTime(),
        "wall_time": wall,
    }

    median = ""
    if model.getNSols() > 0:
        for idx in range(n):
            if idx >= p and model.getVal(u[idx]) < 0.5:
                break
            median += str(int(round(model.getVal(x[idx]))))
    res["median"] = median
    return res


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "random/I_5_10_0.txt"
    tl = float(sys.argv[2]) if len(sys.argv) > 2 else 600
    r = solve_instance(path, time_limit=tl, quiet=False)
    print("\n=== SCIP monolith ===")
    for k, v in r.items():
        print(f"  {k}: {v}")
