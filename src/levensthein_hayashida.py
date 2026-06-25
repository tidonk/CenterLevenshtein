import gurobipy as gp
from gurobipy import GRB

def levenshtein(s1, s2):
    if len(s1) < len(s2):
        return levenshtein(s2, s1)

    if len(s2) == 0:
        return len(s1)

    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[
                             j + 1] + 1
            deletions = current_row[j] + 1  
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    return previous_row[-1]

print("### STARTING HAYASHIDA BINARY ###")

def readStrings(filepath):
    strings = []
    with open(filepath, "r") as f:
        f.readline()
        f.readline()
        f.readline()
        for line in f.readlines():
            l = line.strip()
            l = l.replace(" ", "")
            strings.append(l)
    return strings

#num Instances
I = 3

with open(f"result_hayashida_binary_sum.csv", "a") as file:
    file.write(f"Instance,Seed,BestIncumbent,BestBound,SolutionTime,GAP,Nodes,SimplexIterations,CenterString\n")

for stringlength in [5, 10, 15, 20]:
    for stringnumber in [10, 20, 30, 40, 50]:
        # skip if below 10_50
        if stringlength == 5 or (stringlength == 10 and stringnumber < 50):
            continue
        for it in range(I):
            for seed in [2025]:
                n, m, sigma = None, None, None
                
                strings = readStrings(f"random/I_{stringlength}_{stringnumber}_{it}.txt")
                alphabet = {1:"0", 2: "1"}
                print(f"I_{stringlength}_{stringnumber}_{it}.txt")
                Csub=1
                Cdel=1
                Cins=1

                A = {char: idx for idx, char in alphabet.items()}
                max_len = sum(len(s) for s in strings)
                gp.setParam("LogFile", "hayashida.log")

                model = gp.Model("ILPMed")
                model.setParam(GRB.Param.TimeLimit, 600)

                # Variables
                t = model.addVars(max_len+1, vtype=GRB.INTEGER, lb=1, ub=len(alphabet), name="t")
                l = model.addVar(vtype=GRB.INTEGER, lb=0, ub=max_len, name="l")

                x = {}
                y = {}
                z = {}
                g = {}
                h = {}

                for k, s in enumerate(strings):
                    nk = len(s)
                    for i in range(nk + 1):
                        for j in range(max_len + 1):
                            if i > 0:
                                x[k, i, j] = model.addVar(vtype=GRB.BINARY, name=f"x_{k}_{i}_{j}")
                            if j > 0:
                                y[k, i, j] = model.addVar(vtype=GRB.BINARY, name=f"y_{k}_{i}_{j}")
                            if i > 0 and j > 0:
                                z[k, i, j] = model.addVar(vtype=GRB.BINARY, name=f"z_{k}_{i}_{j}")
                                g[k, i, j] = model.addVar(vtype=GRB.BINARY, name=f"g_{k}_{i}_{j}")
                                h[k, i, j] = model.addVar(vtype=GRB.BINARY, name=f"h_{k}_{i}_{j}")

                # Objective
                obj = 0
                for k, s in enumerate(strings):
                    nk = len(s)
                    for i in range(1, nk + 1):
                        obj += Cdel * x[k, i, 0]
                    for j in range(1, max_len + 1):
                        obj += Cins * y[k, 0, j]
                    for i in range(1, nk + 1):
                        for j in range(1, max_len + 1):
                            obj += (Cdel * x[k, i, j] + Cins * y[k, i, j] + Csub * h[k, i, j])
                    obj -= Cins * (max_len - l)

                model.setObjective(obj, GRB.MINIMIZE)

                # Constraints
                for k, s in enumerate(strings):
                    nk = len(s)
                    model.addConstr(x[k,1,0] + y[k,0,1] + z[k,1,1] == 1, name=f"a1_{k}")
                    model.addConstr(x[k,nk,0] == y[k,nk,1], name=f"a3_{k}")
                    model.addConstr(x[k,1,max_len] == y[k,0,max_len], name=f"a5_{k}")
                    model.addConstr(x[k,nk,max_len] + y[k,nk,max_len] + z[k,nk,max_len] == 1, name=f"a9_{k}")
                    for i in range(1, nk):
                        model.addConstr(x[k,i,0] == x[k,i+1,0] + y[k,i,1] + z[k, i+1, 1], name=f"a2_{k}_{i}")
                        model.addConstr(x[k,i,max_len] + y[k,i,max_len] + z[k,i,max_len] == x[k,i+1,max_len], name=f"a8_{k}_{i}")
                        for j in range(1, max_len):
                            model.addConstr(x[k,i,j] + y[k,i,j] + z[k,i,j] == x[k,i+1,j] + y[k,i,j+1] + z[k, i+1, j+1], name=f"a6_{k}_{i}_{j}")
                    for i in range(1, nk+1):
                        for j in range(1, max_len+1):
                            model.addConstr(A[s[i-1]]-t[j] <= len(A)*g[k,i,j], name=f"c1_{k}_{i}_{j}")
                            model.addConstr(-A[s[i-1]]+t[j] <= len(A)*g[k,i,j], name=f"c2_{k}_{i}_{j}")
                            model.addConstr(h[k,i,j] >= z[k,i,j]+g[k,i,j]-1, name=f"d1_{k}_{i}_{j}")
                            model.addConstr(h[k,i,j] <= 0.5*(z[k,i,j]+g[k,i,j]), name=f"d2_{k}_{i}_{j}")
                    for j in range(1, max_len+1):
                        if j < max_len:
                            model.addConstr(y[k,0,j] == x[k,1,j] + y[k,0,j+1] + z[k,1,j+1], name=f"a4_{k}_{j}")
                            model.addConstr(x[k,nk,j]+y[k,nk,j]+z[k,nk,j] == y[k,nk,j+1], name=f"a7_{k}_{j}")
                        model.addConstr(y[k,nk,j] >= 1/max_len*(j-l), name=f"b_{k}_{j}")
                model.optimize()

                if model.SolCount > 0:

                    median_string = ''.join(alphabet[int(t[j].X)] for j in range(1,max_len) if j < int(l.X)+1)
                else:
                    median_string = ""
                with open(f"result_hayashida_binary_sum.csv", "a") as file:
                    # Collect results
                    best_incumbent = model.ObjVal if model.SolCount > 0 else "NFS" #no feasible solution
                    best_bound = model.ObjBound if model.Status in [GRB.OPTIMAL, GRB.TIME_LIMIT] else "N/A"
                    solution_time = model.Runtime
                    gap = model.MIPGap
                    nodes = model.NodeCount
                    simplexiters = model.IterCount
                    file.write(f"I_{stringlength}_{stringnumber}_{it}.txt,{seed},{best_incumbent},{best_bound},{solution_time},{gap},{nodes},{simplexiters},{median_string}\n")


