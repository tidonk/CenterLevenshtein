import gurobipy as gb
from gurobipy import GRB
from tqdm import tqdm

def generate_y(nk, n, ray):
    y_vars = []
    for f_index in range(nk+1):
        for s_index in range(n+1):
            if abs(f_index - s_index) >= ray:
                continue
            if f_index < n and abs(f_index + 1 - s_index) < ray:
                y_vars.append(((f_index, s_index), (f_index+1, s_index)))
            if s_index < n and abs(f_index - s_index - 1) < ray:
                y_vars.append(((f_index, s_index), (f_index, s_index+1)))
            if f_index < n and s_index < n:
                y_vars.append(((f_index, s_index), (f_index+1, s_index+1)))
    return y_vars


def generate_ye(word, nk, n, ray):
    y_vars = []
    for f_index in range(nk+1):
        for s_index in range(n+1):
            if abs(f_index - s_index) >= ray:
                continue
            if f_index < nk and s_index < n:
                y_vars.append(((f_index, s_index), (f_index+1, s_index+1), word[f_index]))
    return y_vars


def generate_nodes(y):
    nodes = set()
    for arc in y:
        if arc[0] not in nodes:
            nodes.add(arc[0])
        if arc[1] not in nodes:
            nodes.add(arc[1])
    return list(nodes)

#num Instances
I = 3
#Set result file
with open(f"result_path_binary_2n.csv", "a") as file:
    file.write(f"Instance,Seed,Best Incumbent,BestBound,SolutionTime,GAP,Nodes,SimplexIterations,Median\n")

print("### STARTING PATH ###")

for stringlength in [5, 10, 15, 20]:
    for stringnumber in [10, 20, 30, 40, 50]:
        for it in range(I):
            for seed in [2025]:
                # -------------------------------------------------
                # Read input 
                # -------------------------------------------------
                n, m, sigma = None, None, None

                with open(f"random/I_{stringlength}_{stringnumber}_{it}.txt", "r") as f:
                    lines = f.readlines()

                m = int(lines[1].split("=")[1].strip())
                sigma = [list(map(int, line.strip())) for line in lines[3:]]
                sigmas = ["".join(map(str, row)) for row in sigma]

                n = max(len(s) for s in sigma)
                n = 2*n
                print(f"Instance_{stringlength}_{stringnumber}_{it}.txt")
                print(f"n = {n}")
                print(f"m = {m}")
                print("sigma =\n", sigma)
                
                model = gb.Model("path binary")

                string_length = n
                string_number = m
                stringpool = sigma
                upper_bound = n
                obj = 1 #median

                model = gb.Model()
                
                x = {}
                for i in range(n):
                    name = f"x_{i}"
                    x[i] = model.addVar(vtype=GRB.BINARY, name=name)

                median_lhs = gb.LinExpr()
                slant_arcs = []
                full_arcs = []
                slant_arcs_vars = []
                full_arcs_vars = []
                u = {}
                p = min(len(v) for v in stringpool)
                q = n
                for k in range(p, q+1):
                    u[k] = model.addVar(vtype=GRB.BINARY, name=f"u_{k}")

                for k in range(p, q):
                    model.addConstr(u[k] >= u[k+1], name=f"mono_{k}")
                model.addConstr(u[p] == 1, name="min_length")


                for idx, word in tqdm(enumerate(stringpool)):
                    nk = len(word)
                    cost_y = generate_y(nk, string_length, upper_bound) 
                    arcs, costs = gb.multidict({arc: 1 if arc[0][0] + 1 == arc[1][0] and arc[0][1] + 1 == arc[1][1] else 1 for arc in cost_y})
                    nodes = generate_nodes(cost_y)
                    diagonal_cost_y = generate_ye(word, nk, string_length, upper_bound)
                    ye, ye_costs = gb.multidict({arc: 0 for arc in diagonal_cost_y})

                    slant_arcs.append(ye)
                    full_arcs.append(arcs)

                    flow = model.addVars(
                    arcs,
                    name={(arc): f"flow{idx}_{arc[0][0]}_{arc[0][1]}_{arc[1][0]}_{arc[1][1]}" for arc in arcs},
                    vtype='C'
                    )

                    zero_cost_flow = model.addVars(
                    ye,
                    name={(arc): f"ye{idx}_{arc[0][0]}_{arc[0][1]}_{arc[1][0]}_{arc[1][1]}" for arc in ye},
                    vtype='C'
                    )
                    slant_arcs_vars.append(list(zero_cost_flow.values()))
                    full_arcs_vars.append(list(flow.values()))
                    for arc in arcs:
                        j = arc[1][1]  
                        if p <= j <= q:
                            model.addConstr(flow[arc] <= u[j], name=f"act_full_{idx}_{j}")

                    for arc in diagonal_cost_y:
                        j = arc[1][1]
                        if p <= j <= q:
                            model.addConstr(zero_cost_flow[arc] <= u[j], name=f"act_diag_{idx}_{j}")

                    for zcf in zero_cost_flow:
                        if zcf[2] == 0:
                            model.addConstr(zero_cost_flow[zcf] <= 1 - x[zcf[0][1]])
                        else:
                            model.addConstr(zero_cost_flow[zcf] <= x[zcf[0][1]])
                    
                    source = gb.LinExpr()
                    for arc in arcs:
                        if arc[0] == (0, 0):
                            source += flow[arc]
                    for arc in diagonal_cost_y:
                        if arc[0] == (0, 0):
                            source += zero_cost_flow[arc]
                    
                    model.addConstr(source == 1, name='source' + str(idx))
                    for i, j in nodes:
                        if (i, j) == (0, 0) or i == nk:
                            continue
                        outflow = gb.LinExpr()
                        inflow = gb.LinExpr()
                        for arc in arcs:
                            if arc[0] == (i, j):
                                outflow += flow[arc]
                            if arc[1] == (i, j):
                                inflow += flow[arc]
                        for arc in diagonal_cost_y:
                            if arc[0] == (i, j):
                                outflow += zero_cost_flow[arc]
                            if arc[1] == (i, j):
                                inflow += zero_cost_flow[arc]
                        model.addConstr(outflow == inflow, name='conserv' + str(idx) + '_' + str(i) + str(j))
                    # Horizontal/vertical conflict constraints
                    for node in nodes:
                        if node == (0, 0) or node == (nk, string_length):
                            continue

                        i, j = node

                        arc_left  = ((i, j-1), (i, j))
                        arc_down  = ((i+1, j), (i, j))
                        arc_up    = ((i-1, j), (i, j))
                        arc_right = ((i, j), (i, j+1))

                        if arc_left in arcs and arc_down in arcs:
                            model.addConstr(flow[arc_left] + flow[arc_down] <= 1)

                        if arc_up in arcs and arc_right in arcs:
                            model.addConstr(flow[arc_up] + flow[arc_right] <= 1)

                    for k in range(p, q):
                        outflow = gb.LinExpr()
                        inflow = gb.LinExpr()

                        node = (nk, k)

                        for arc in arcs:
                            if arc[0] == node:
                                outflow += flow[arc]
                            if arc[1] == node:
                                inflow += flow[arc]

                        for arc in diagonal_cost_y:
                            if arc[0] == node:
                                outflow += zero_cost_flow[arc]
                            if arc[1] == node:
                                inflow += zero_cost_flow[arc]

                        model.addConstr(inflow - outflow == u[k] - u[k+1],
                                        name=f"sink_{idx}_{k}")

                    outflow = gb.LinExpr()
                    inflow = gb.LinExpr()
                    node = (nk, q)

                    for arc in arcs:
                        if arc[0] == node:
                            outflow += flow[arc]
                        if arc[1] == node:
                            inflow += flow[arc]

                    for arc in diagonal_cost_y:
                        if arc[0] == node:
                            outflow += zero_cost_flow[arc]
                        if arc[1] == node:
                            inflow += zero_cost_flow[arc]

                    model.addConstr(inflow - outflow == u[q], name=f"sink_last_{idx}")

                    if obj:
                        ### Median problem ###
                        median_lhs += flow.prod(costs) + zero_cost_flow.prod(ye_costs)
                        model.setObjective(median_lhs)
                            
                    model.update()
                                    
                    flow_vars = []
                    ye_vars = []
                    for var in model.getVars():
                        if 'flow' in var.VarName:
                            flow_vars.append(var)
                        if 'ye' in var.VarName:
                            ye_vars.append(var)

                    
                    
                        model.update()

                    
                model.setParam("Seed", seed)
                model.setParam(GRB.Param.TimeLimit, 600)

                model.optimize()
                if model.SolCount > 0:
                    total_best_incumbent = model.ObjVal
                    total_center_string = ""
                    for idx, v in enumerate(x):
                        if idx >= p and u[idx].X < 0.5: break
                        total_center_string += str(int(x[v].X))

                    if model.Status in [GRB.OPTIMAL, GRB.TIME_LIMIT]:
                        total_best_bound =  model.ObjBound
                total_solution_time = model.Runtime
                total_gap = model.MIPGap
                total_nodes = model.NodeCount
                total_simplexiters = model.IterCount
                with open(f"result_path_binary_2n.csv", "a") as file:
                    # Write results in a single row format
                    file.write(f"I_{stringlength}_{stringnumber}_{it}.txt,{seed},{total_best_incumbent},{total_best_bound},{total_solution_time},{total_gap},{total_nodes},{total_simplexiters},{total_center_string}\n")
            
