"""Validate a Benders solution by plugging its (x, u) assignment into the
monolithic model: fix the linking variables to the Benders values, re-solve the
(now pure-LP-over-flows) monolith, and confirm it is feasible with the same
objective. If fixing the Benders x,u makes the monolith infeasible or changes
the objective, the Benders solution is not a valid solution of the full model.
"""
import sys

from pyscipopt import Model

import scip_benders
from scip_common import (read_instance, instance_dims, add_x_vars, add_u_vars,
                         add_length_constraints, add_block)


def build_mono_fixed(sigma, n, p, q, xvals, uvals):
    model = Model("mono_fixed")
    x = add_x_vars(model, n)
    u = add_u_vars(model, p, q)
    add_length_constraints(model, u, p, q)
    # Fix the linking variables to the Benders assignment.
    for i in range(n):
        model.addCons(x[i] == xvals[i], name=f"fix_x_{i}")
    for k in range(p, q + 1):
        model.addCons(u[k] == uvals[k], name=f"fix_u_{k}")
    obj = 0
    for idx, word in enumerate(sigma):
        obj = obj + add_block(model, x, u, idx, word, n, p, q)
    model.setObjective(obj, "minimize")
    return model


def validate(path, time_limit=120):
    print(f">>> {path}")
    bend = scip_benders.solve_instance(path, time_limit=time_limit, quiet=True)
    if bend.get("xvals") is None:
        print("    Benders produced no solution; cannot validate.")
        return False

    m, sigma = read_instance(path)
    n, p, q = instance_dims(sigma)

    model = build_mono_fixed(sigma, n, p, q, bend["xvals"], bend["uvals"])
    model.setParam("limits/time", time_limit)
    model.hideOutput()
    model.optimize()

    status = model.getStatus()
    feasible = model.getNSols() > 0 and status in ("optimal", "feasible")
    mono_obj = model.getObjVal() if model.getNSols() > 0 else None
    bend_obj = bend["obj"]
    obj_match = (mono_obj is not None and bend_obj is not None
                 and abs(mono_obj - bend_obj) < 1e-6)

    print(f"    benders obj          : {bend_obj}  median={bend['median']}")
    print(f"    monolith (x,u fixed) : status={status} obj={mono_obj}")
    print(f"    feasible             : {feasible}")
    print(f"    objective matches    : {obj_match}")
    ok = feasible and obj_match
    print(f"    VALIDATION           : {'PASS' if ok else 'FAIL'}\n")
    return ok


if __name__ == "__main__":
    insts = sys.argv[1:] if len(sys.argv) > 1 else [
        "random/I_5_10_0.txt",
        "random/I_5_10_1.txt",
        "random/I_5_10_2.txt",
        "random/I_5_20_0.txt",
        "random/I_5_20_1.txt",
        "random/I_10_10_0.txt",
    ]
    results = [validate(i) for i in insts]
    print("=" * 50)
    print(f"All validations passed: {all(results)} ({sum(results)}/{len(results)})")
