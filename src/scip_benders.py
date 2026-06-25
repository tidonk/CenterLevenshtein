"""SCIP Benders' decomposition port of path_binary_digitalized.py.

Master: the median bits x and the length indicators u (+ length constraints).
SCIP adds one auxiliary variable phi_i per subproblem.
Subproblem i: a copy of x, u (continuous, fixed by the master) plus all of
string i's flow/conflict/sink constraints; its objective is the block arc cost.
Once x,u are fixed each subproblem is an LP (continuous flows) -> SCIP detects
it as convex and derives classical optimality cuts from the LP duals.
"""
import math
import os
import sys
import time

from pyscipopt import (Model, SCIP_PARAMSETTING, Benders, SCIP_RESULT,
                       SCIP_LPSOLSTAT)

from scip_common import (read_instance, instance_dims, add_x_vars, add_u_vars,
                         add_length_constraints, add_block, log_path)


class StringBenders(Benders):
    """Custom Benders plugin (mirrors the PySCIPOpt flp example).

    We solve each subproblem LP ourselves (via probing) and hand the framework
    the true objective, which it uses for cut generation and enforcement. This
    lets heuristics stay enabled: for every instance that SCIP solves to
    optimality the final incumbent is valid (the ENFO optimality cuts force it).

    Caveat: for a subproblem that is an LP after the linking vars are fixed,
    SCIP's convex CONSCHECK does not re-solve to verify a heuristic candidate's
    auxiliary variables, so a run *interrupted by the time limit* can report an
    incumbent whose phi_i under-values the true cost (the dual bound stays
    valid). The harness flags this via the validity check against the monolith.
    """

    def __init__(self, sigma, n, p, q, master_x, master_u, name, sub_indices):
        super().__init__()
        self.sigma, self.n, self.p, self.q = sigma, n, p, q
        self.name = name
        # Benders subproblem k (k=0..len-1) corresponds to string sub_indices[k];
        # strings kept in the master are not in this list.
        self.sub_indices = list(sub_indices)
        self.master_by_name = {}
        for v in master_x.values():
            self.master_by_name[v.name] = v
        for v in master_u.values():
            self.master_by_name[v.name] = v
        self.subproblems = {}
        self.sub_by_name = {}

    def benderscreatesub(self, probnumber):
        idx = self.sub_indices[probnumber]
        sub = build_sub(idx, self.sigma[idx], self.n, self.p,
                        self.q, link_vtype="B")
        self.subproblems[probnumber] = sub
        self.sub_by_name[probnumber] = {v.name: v for v in sub.getVars()}
        self.model.addBendersSubproblem(self, sub)

    def bendersgetvar(self, variable, probnumber):
        name = variable.name
        if probnumber == -1:
            # variable is a subproblem variable -> return the master variable
            return {"mappedvar": self.master_by_name.get(name)}
        # variable is a master variable -> return its copy in subproblem probnumber
        return {"mappedvar": self.sub_by_name[probnumber].get(name)}

    def _solve(self, solution, probnumber):
        sub = self.subproblems[probnumber]
        self.model.setupBendersSubproblem(probnumber, self, solution)
        sub.solveProbingLP()
        objective = sub.infinity()
        result = SCIP_RESULT.DIDNOTRUN
        stat = sub.getLPSolstat()
        if stat == SCIP_LPSOLSTAT.OPTIMAL:
            objective = sub.getObjVal()
            result = SCIP_RESULT.FEASIBLE
        elif stat == SCIP_LPSOLSTAT.INFEASIBLE:
            result = SCIP_RESULT.INFEASIBLE
        elif stat == SCIP_LPSOLSTAT.UNBOUNDEDRAY:
            result = SCIP_RESULT.UNBOUNDED
        return {"objective": objective, "result": result}

    def benderssolvesubconvex(self, solution, probnumber, onlyconvex):
        # LP relaxation solve, used for cut generation and the lower bound.
        return self._solve(solution, probnumber)

    def benderssolvesub(self, solution, probnumber):
        # "CIP" solve fallback. In practice SCIP never calls this: once the
        # linking vars are fixed the block has no integer variables, so the
        # framework always takes the convex (LP) path above. Implemented for
        # completeness / correctness if that ever changes.
        return self._solve(solution, probnumber)

    def bendersfreesub(self, probnumber):
        sub = self.subproblems[probnumber]
        if sub.inProbing():
            sub.endProbing()


def build_master(sigma, n, p, q, keep_indices=()):
    master = Model("path_binary_master")
    x = add_x_vars(master, n)
    u = add_u_vars(master, p, q)
    add_length_constraints(master, u, p, q)
    # Partial Benders decomposition: any string in keep_indices is retained in
    # the master -- its flow vars/constraints are built here and its arc cost is
    # added to the master objective. The rest become Benders subproblems, for
    # which SCIP adds the auxiliary phi_i vars.
    obj = 0
    for idx in keep_indices:
        obj = obj + add_block(master, x, u, idx, sigma[idx], n, p, q)
    master.setObjective(obj, "minimize")
    return master, x, u


def build_sub(idx, word, n, p, q, link_vtype="C"):
    sub = Model(f"sub_{idx}")
    # Copies of the linking variables, identical names. SCIP fixes them to the
    # master values when solving the subproblem. For the custom plugin they must
    # be binary so the raw subproblem is non-convex and the framework transforms
    # it (mirrors the flp example); after the linking vars are fixed the block
    # is a pure LP solved by probing.
    xs = {i: sub.addVar(vtype=link_vtype, lb=0.0, ub=1.0, name=f"x_{i}") for i in range(n)}
    us = {k: sub.addVar(vtype=link_vtype, lb=0.0, ub=1.0, name=f"u_{k}") for k in range(p, q + 1)}
    obj = add_block(sub, xs, us, idx, word, n, p, q)
    sub.setObjective(obj, "minimize")
    return sub


def solve_instance(path, time_limit=600, seed=2025, quiet=False, custom=False,
                   log_dir="logs", keep_in_master=None, log_label=None,
                   log_file=None, subprobfrac=None):
    """Solve one instance with SCIP Benders.

    custom=False: default Benders (initBendersDefault). The framework's
        CONSCHECK does not re-verify auxiliary-variable optimality for heuristic
        solutions, so heuristics are switched OFF to keep the incumbent sound.
    custom=True: custom StringBenders plugin whose solve callback runs on every
        candidate (including heuristic) solution, so heuristics stay ON.

    keep_in_master: optional partial decomposition. Either an int (retain the
        first k strings in the master) or an iterable of string indices. Those
        blocks are embedded in the master; only the rest become subproblems.
    """
    m, sigma = read_instance(path)
    n, p, q = instance_dims(sigma)

    # Resolve which strings stay in the master vs. become subproblems.
    if keep_in_master is None:
        keep = set()
    elif isinstance(keep_in_master, float):
        # fraction in (0,1] -> retain ceil(fraction * m) blocks
        k = math.ceil(keep_in_master * len(sigma))
        keep = set(range(min(k, len(sigma))))
    elif isinstance(keep_in_master, int):
        keep = set(range(min(keep_in_master, len(sigma))))
    else:
        keep = set(keep_in_master)
    sub_indices = [i for i in range(len(sigma)) if i not in keep]
    nsub = len(sub_indices)

    master, x, u = build_master(sigma, n, p, q, keep_indices=sorted(keep))

    # Recommended settings for Benders.
    master.setPresolve(SCIP_PARAMSETTING.OFF)
    master.setBoolParam("misc/allowstrongdualreds", False)
    master.setBoolParam("benders/default/cutstrengthenenabled", True)

    if nsub == 0:
        # Everything retained -> the master is the full monolith; no Benders.
        pass
    elif custom:
        master.setBoolParam("misc/allowweakdualreds", False)
        master.setBoolParam("benders/copybenders", False)
        bd = StringBenders(sigma, n, p, q, x, u, "stringbenders", sub_indices)
        master.includeBenders(bd, bd.name, "median Benders decomposition")
        master.includeBendersDefaultCuts(bd)
        master.activateBenders(bd, nsub)
        master.setBoolParam("constraints/benders/active", True)
        master.setBoolParam("constraints/benderslp/active", True)
        master.setBoolParam(f"benders/{bd.name}/updateauxvarbound", False)
        master.updateBendersLowerbounds({k: 0.0 for k in range(nsub)}, bd)
        if subprobfrac is not None:
            master.setRealParam(f"benders/{bd.name}/subprobfrac", subprobfrac)
    else:
        # The default Benders' CONSCHECK only verifies feasibility, not
        # auxiliary-variable optimality, so a heuristic can submit an (x, phi)
        # point whose phi under-values the true subproblem cost and have it
        # accepted as the incumbent. Disabling heuristics forces every incumbent
        # through the B&B where the optimality cuts are enforced.
        master.setHeuristics(SCIP_PARAMSETTING.OFF)
        subproblems = {k: build_sub(idx, sigma[idx], n, p, q)
                       for k, idx in enumerate(sub_indices)}
        master.initBendersDefault(subproblems)
        master.updateBendersLowerbounds({k: 0.0 for k in range(nsub)})

    master.setParam("limits/time", time_limit)
    master.setParam("randomization/randomseedshift", seed)
    method = log_label or ("benders_custom" if custom else "benders")
    lf = log_file or (log_path(path, method, time_limit, log_dir) if log_dir else None)
    if lf:
        os.makedirs(os.path.dirname(lf), exist_ok=True)
        master.setLogfile(lf)
    if quiet:
        master.hideOutput()

    t0 = time.time()
    master.optimize()
    wall = time.time() - t0

    if lf:
        # Append SCIP's full statistics (incl. the Benders Decomposition table
        # and other additional per-plugin statistics) to the per-run log.
        master.printStatistics()

    # The median only needs the master x,u (available from the best solution).
    # computeBestSolSubproblems re-solves subproblems for their internals; skip
    # it under the custom plugin to avoid re-entering the probing callbacks.
    if not custom and nsub > 0 and master.getNSols() > 0:
        master.computeBestSolSubproblems()

    res = {
        "instance": path,
        "status": master.getStatus(),
        "obj": master.getObjVal() if master.getNSols() > 0 else None,
        "bound": master.getDualbound(),
        "gap": master.getGap(),
        "nodes": master.getNNodes(),
        "lp_iters": master.getNLPIterations(),
        "solving_time": master.getSolvingTime(),
        "wall_time": wall,
    }

    median = ""
    if master.getNSols() > 0:
        for idx in range(n):
            if idx >= p and master.getVal(u[idx]) < 0.5:
                break
            median += str(int(round(master.getVal(x[idx]))))
        res["xvals"] = {i: round(master.getVal(x[i])) for i in range(n)}
        res["uvals"] = {k: round(master.getVal(u[k])) for k in range(p, q + 1)}
    res["median"] = median
    res["dims"] = (n, p, q)
    return res


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "random/I_5_10_0.txt"
    tl = float(sys.argv[2]) if len(sys.argv) > 2 else 600
    r = solve_instance(path, time_limit=tl, quiet=False)
    print("\n=== SCIP Benders ===")
    for k, v in r.items():
        print(f"  {k}: {v}")
