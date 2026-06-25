"""Shared model-building logic for the SCIP ports of path_binary_digitalized.py.

This replicates the arc/node generation and the per-string ("block") constraint
system of the original Gurobi model verbatim, so that the monolithic and the
Benders formulations are guaranteed to encode the same problem.
"""
import os

from pyscipopt import quicksum


def log_path(instance_path, method, time_limit, log_dir="logs"):
    """Return a per-run SCIP log file path inside log_dir (created if needed).

    e.g. logs/I_5_10_0_benders_custom_tl600.log
    """
    os.makedirs(log_dir, exist_ok=True)
    stem = os.path.splitext(os.path.basename(instance_path))[0]
    return os.path.join(log_dir, f"{stem}_{method}_tl{int(time_limit)}.log")


# ---------------------------------------------------------------------------
# Instance reading (mirrors path_binary_digitalized.py)
# ---------------------------------------------------------------------------
def read_instance(path):
    with open(path, "r") as f:
        lines = f.readlines()
    m = int(lines[1].split("=")[1].strip())
    sigma = [list(map(int, line.strip())) for line in lines[3:] if line.strip()]
    return m, sigma


def instance_dims(sigma):
    """Return (n, p, q): n = 2*maxlen (median grid width), p = min length, q = n."""
    n = 2 * max(len(s) for s in sigma)
    p = min(len(v) for v in sigma)
    q = n
    return n, p, q


# ---------------------------------------------------------------------------
# Arc / node generation (verbatim from the original)
# ---------------------------------------------------------------------------
def generate_y(nk, n, ray):
    y_vars = []
    for f_index in range(nk + 1):
        for s_index in range(n + 1):
            if abs(f_index - s_index) >= ray:
                continue
            if f_index < n and abs(f_index + 1 - s_index) < ray:
                y_vars.append(((f_index, s_index), (f_index + 1, s_index)))
            if s_index < n and abs(f_index - s_index - 1) < ray:
                y_vars.append(((f_index, s_index), (f_index, s_index + 1)))
            if f_index < n and s_index < n:
                y_vars.append(((f_index, s_index), (f_index + 1, s_index + 1)))
    return y_vars


def generate_ye(word, nk, n, ray):
    y_vars = []
    for f_index in range(nk + 1):
        for s_index in range(n + 1):
            if abs(f_index - s_index) >= ray:
                continue
            if f_index < nk and s_index < n:
                y_vars.append(((f_index, s_index), (f_index + 1, s_index + 1), word[f_index]))
    return y_vars


def generate_nodes(y):
    nodes = set()
    for arc in y:
        nodes.add(arc[0])
        nodes.add(arc[1])
    return list(nodes)


# ---------------------------------------------------------------------------
# Master (linking) variables
# ---------------------------------------------------------------------------
def add_x_vars(model, n):
    return {i: model.addVar(vtype="B", name=f"x_{i}") for i in range(n)}


def add_u_vars(model, p, q):
    return {k: model.addVar(vtype="B", name=f"u_{k}") for k in range(p, q + 1)}


def add_length_constraints(model, u, p, q):
    for k in range(p, q):
        model.addCons(u[k] >= u[k + 1], name=f"mono_{k}")
    model.addCons(u[p] == 1, name="min_length")


# ---------------------------------------------------------------------------
# One per-string block (flow vars + constraints). Returns the block objective
# expression (sum of unit-cost arc flows). x, u are the (possibly local) copies
# of the linking variables.
# ---------------------------------------------------------------------------
def add_block(model, x, u, idx, word, n, p, q):
    # x, u may be the master's binary linking variables or local copies; the
    # block constraints are identical either way.
    nk = len(word)
    string_length = n
    upper_bound = n

    cost_y = generate_y(nk, string_length, upper_bound)
    diagonal_cost_y = generate_ye(word, nk, string_length, upper_bound)
    arcs = list(cost_y)
    nodes = generate_nodes(cost_y)

    flow = {}
    for arc in arcs:
        flow[arc] = model.addVar(
            vtype="C", lb=0.0,
            name=f"flow{idx}_{arc[0][0]}_{arc[0][1]}_{arc[1][0]}_{arc[1][1]}")

    zero_cost_flow = {}
    for arc in diagonal_cost_y:
        zero_cost_flow[arc] = model.addVar(
            vtype="C", lb=0.0,
            name=f"ye{idx}_{arc[0][0]}_{arc[0][1]}_{arc[1][0]}_{arc[1][1]}")

    # column capacity (length gating) -- constraints (10)
    for arc in arcs:
        j = arc[1][1]
        if p <= j <= q:
            model.addCons(flow[arc] <= u[j], name=f"act_full_{idx}_{j}")
    for arc in diagonal_cost_y:
        j = arc[1][1]
        if p <= j <= q:
            model.addCons(zero_cost_flow[arc] <= u[j], name=f"act_diag_{idx}_{j}")

    # gating of free slants by the median bit -- constraints (5)/(6)
    for zcf in diagonal_cost_y:
        if zcf[2] == 0:
            model.addCons(zero_cost_flow[zcf] <= 1 - x[zcf[0][1]])
        else:
            model.addCons(zero_cost_flow[zcf] <= x[zcf[0][1]])

    # source
    source = quicksum(flow[arc] for arc in arcs if arc[0] == (0, 0)) \
        + quicksum(zero_cost_flow[arc] for arc in diagonal_cost_y if arc[0] == (0, 0))
    model.addCons(source == 1, name=f"source{idx}")

    # flow conservation
    for (i, j) in nodes:
        if (i, j) == (0, 0) or i == nk:
            continue
        outflow = quicksum(flow[arc] for arc in arcs if arc[0] == (i, j)) \
            + quicksum(zero_cost_flow[arc] for arc in diagonal_cost_y if arc[0] == (i, j))
        inflow = quicksum(flow[arc] for arc in arcs if arc[1] == (i, j)) \
            + quicksum(zero_cost_flow[arc] for arc in diagonal_cost_y if arc[1] == (i, j))
        model.addCons(outflow == inflow, name=f"conserv{idx}_{i}{j}")

    # horizontal/vertical conflict constraints
    arcset = set(arcs)
    for node in nodes:
        if node == (0, 0) or node == (nk, string_length):
            continue
        i, j = node
        arc_left = ((i, j - 1), (i, j))
        arc_down = ((i + 1, j), (i, j))
        arc_up = ((i - 1, j), (i, j))
        arc_right = ((i, j), (i, j + 1))
        if arc_left in arcset and arc_down in arcset:
            model.addCons(flow[arc_left] + flow[arc_down] <= 1)
        if arc_up in arcset and arc_right in arcset:
            model.addCons(flow[arc_up] + flow[arc_right] <= 1)

    # sink balance tied to length indicators -- constraint (11)
    for k in range(p, q):
        node = (nk, k)
        outflow = quicksum(flow[arc] for arc in arcs if arc[0] == node) \
            + quicksum(zero_cost_flow[arc] for arc in diagonal_cost_y if arc[0] == node)
        inflow = quicksum(flow[arc] for arc in arcs if arc[1] == node) \
            + quicksum(zero_cost_flow[arc] for arc in diagonal_cost_y if arc[1] == node)
        model.addCons(inflow - outflow == u[k] - u[k + 1], name=f"sink_{idx}_{k}")

    node = (nk, q)
    outflow = quicksum(flow[arc] for arc in arcs if arc[0] == node) \
        + quicksum(zero_cost_flow[arc] for arc in diagonal_cost_y if arc[0] == node)
    inflow = quicksum(flow[arc] for arc in arcs if arc[1] == node) \
        + quicksum(zero_cost_flow[arc] for arc in diagonal_cost_y if arc[1] == node)
    model.addCons(inflow - outflow == u[q], name=f"sink_last_{idx}")

    # block objective: every full arc costs 1, free slants cost 0
    return quicksum(flow[arc] for arc in arcs)
