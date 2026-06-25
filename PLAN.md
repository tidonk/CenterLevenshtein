You're right, and that detail is exactly what dissolves the obstruction from my last message. I was analyzing the cost‑coupled diagonal (substitution cost $=\mathbb 1[x_k\neq\lambda^i_h]$ as an objective coefficient), which is what makes $\ell(x,\lambda^i)$ concave in $x$. Their path model does the opposite: costs are constant ($c_e=1$ everywhere, $c_e=0$ on the free slant arcs), and the median instead controls the **capacity** of the duplicated zero‑cost slants — each slant arc is duplicated into a costly copy and a free copy, and the free arc ending in $hk$ has capacity $1$ if $x_k=\lambda^i_h$ and $0$ otherwise, i.e. constraints (5) $y_e\le x_{k_e}$ and (6) $y_e\le 1-x_{k_e}$. So $x$ sits in the **RHS** of the subproblem, not the objective. That is the standard interdiction/gating trick, and it flips the value function from concave to convex piecewise‑linear. Scratch the "LBBD‑only" framing — this is ordinary LP‑Benders and the duals are exactly what you want.

Quick confirmation of the convexity on the toy case from before (median length 2 vs $\lambda^i=\texttt{0}$): the gated‑capacity LP gives $Q_i(x)=\max(1,\,x_1+x_2)$ — convex — agreeing with $\ell$ at all four integer points, where the cost‑coupled version gave the concave $1+\min(x_1,x_2)$.

## Extracting the dual

For fixed $(\hat x,\hat z)$ the subproblem is a min‑cost unit flow on $G_i$; the matrix is TU, so LP $=$ IP $=$ your DP. The dual is the shortest‑path dual:

- **Node potentials** $\pi^i_u$ for flow balance (3)–(4): these are just the forward DP distances, $\pi^i_u=D^i_u$ (run the recursion backward for $\overleftarrow D^i$, the suffix distances, at no extra cost).
- **Gating multipliers** $\alpha_e\ge0$ on (5) and $\beta_e\ge0$ on (6): these are the shadow prices of the free arcs. With $\pi=D^i$, dual feasibility on a free slant $e:(h\!-\!1)(k\!-\!1)\to hk$ (cost 0) forces $\alpha_e,\beta_e\ge \pi_{hk}-\pi_{(h-1)(k-1)}$, so the minimal optimal value is

$$\mu_e \;=\; \big(D^i_{hk}-D^i_{(h-1)(k-1)}\big)^+\in\{0,1\}.$$

$\mu_e=1$ **iff the free arc is closed at $\hat x$ (a mismatch) and its costly twin is locally tight**, i.e. lies on a shortest path — exactly the substitutions the alignment is currently paying for that a matching median bit would save. Open free arcs have $\mu_e=0$ automatically (free diagonal $\Rightarrow$ zero increment). So nothing beyond the DP tables is needed.

## The cut

With known length, the Benders optimality cut is $\theta_i\ge \pi^{i\top}b-\sum_e \mu_e\,U_e(x)$, which evaluates to

$$\theta_i \;\ge\; \hat z_i \;-\!\!\sum_{e\in F^i_1:\,\mu_e=1}\!\! x_{k_e}\;-\!\!\sum_{e\in F^i_0:\,\mu_e=1}\!\!\big(1-x_{k_e}\big),$$

where $F^i_1=\{e\in F^i:\lambda^i_{h_e}=1\}$ (gated by $x_{k_e}$) and $F^i_0$ (gated by $1-x_{k_e}$). Reading: *the distance to $\lambda^i$ drops by at most 1 for each substitution you currently pay that the median is allowed to turn into a match.* It's a genuine global lower bound (dual feasibility is $x$‑independent), tight at $\hat x$.

On the toy case: $\hat x=(1,1)$ gives both free slants closed with $\mu=1$, $\hat z_i=2$, so $\theta_i\ge 2-(1-x_1)-(1-x_2)=x_1+x_2$; $\hat x=(0,0)$ gives $\theta_i\ge1$. The two cuts reconstruct $\max(1,x_1+x_2)=Q_i(x)$ exactly.

The length variables just add linear terms from the same solve: the column caps (10) $y_e\le z_k$ contribute $-\sum_e\gamma_e z_{k_e}$ with $\gamma_e\ge0$ their multipliers, and the $z$‑dependent sink balance (11) contributes $\sum_k \pi^i_{n_ik}(z_k-z_{k+1})$. And since a unit path always exists through the uncapacitated costly arcs for any monotone $z$, the subproblem is never infeasible — you only ever generate optimality cuts, no feasibility cuts.

## Caveats that land in your wheelhouse

The forward‑potential dual above is valid but is the *lazy* dual, and it has two defects you'll want to manage in a B&BC:

- **Over‑crediting / density.** A single $x_k$ gates the whole column of free slants (all rows $h$). The minimal‑feasible dual can put $\mu_e=1$ on several locally‑tight closed arcs in that column, even though a monotone path uses at most one of them — so flipping $x_k$ gets credited more than the true $\le 1$ marginal. Valid (it only weakens the bound), but it yields loose, dense cuts.
- **Non‑uniqueness.** Shortest‑path duals are highly degenerate, so the cut you emit is a modeling choice.

Both point to Pareto‑optimal (Magnanti–Wong) selection, or — cheaper here — restricting the cut's support to the free arcs on one chosen optimal alignment and verifying feasibility against $\overleftarrow D^i$ before lifting, which both sparsifies and tightens. Given binary instances produce many near‑identical alignments, the dual you pick and how you age these cuts will matter more than the cut family itself.

Want me to write out the $z$‑augmented cut in full, or work through the Pareto/single‑alignment dual selection so the emitted cut is sparse and non‑dominated?

# Implement this with Benders in SCIP

I want to take this exact model and code and instances, but solve it using SCIP's Benders. Probably the SP solver needs to be implemented? And then we need a script to test the existing code on a few short instances, plus our SCIP Benders run and compare runtimes. Make a plan. What results do you expect?

Good news first: I pulled their repo. It's pure Python and the model `path_binary_digitalized.py` is **Gurobi** (`gurobipy`), implementing exactly §3.1 — `x[i]` median bits, `u[k]` as the length indicators $z_k$, costly `flow` arcs (cost 1) and `zero_cost_flow` free slants gated by `<= x` / `<= 1-x` (their (5)/(6)), column caps `<= u[j]` (10), and the $u_k-u_{k+1}$ sink balance (11). Instances are `random/I_{len}_{num}_{it}.txt` and `ecc/`. I also confirmed SCIP 10.0 / PySCIPOpt 6.2.1 is available with the full default-Benders API.

Two consequences shape the whole plan.

**The comparison has a confound you must neutralize.** Their baseline is Gurobi-monolith. "SCIP Benders vs their code" conflates *solver* (Gurobi→SCIP, usually a 2–10× slowdown on MIP) with *method* (monolith→Benders). So you need three configurations, not two: Gurobi-monolith (their script), **SCIP-monolith (control)**, and SCIP-Benders. Gurobi-mono vs SCIP-mono isolates the solver penalty; SCIP-mono vs SCIP-Benders isolates the decomposition effect, which is the thing you actually care about.

**You probably do *not* need to implement a subproblem solver for v1.** Once $(x,u)$ are fixed, each block is a continuous LP (the flow vars are `'C'`; the only integers are the linking $x,u$). That's a *convex* Benders subproblem, so SCIP solves it as an LP and `benderscut_opt` derives classical optimality cuts from the LP duals automatically — exactly the dual we discussed, no hand-coding. You flag it with `setBendersSubproblemIsConvex(benders, i, True)`. A custom subproblem (the $O(n_i m)$ DP instead of an LP solve) and a custom cut (our sparse single-alignment/Pareto $\mu$-cut) are *optimizations for later*, not requirements — and they're where your B&BC work actually plugs in.

## Plan

**Phase 0 — repro & instances.** Clone the repo, grab a handful of short instances (`I_5_10_*`, `I_5_20_*`, `I_10_10_*`). If you have a Gurobi license, run their `pat` script on those to get reference objective + `model.Runtime`. If not, skip Gurobi and treat SCIP-mono as the baseline.

**Phase 1 — SCIP monolith (control + correctness gate).** Port `path_binary_digitalized.py` to PySCIPOpt verbatim: same variables, same constraints, `vtype='C'` for flows, binaries for `x`,`u`, objective = total (don't divide by $t$ — argmin is identical). Gate: SCIP-mono optimum must equal Gurobi's on every solved instance (the median *string* may differ — symmetry/alternate optima — only the objective must match).

**Phase 2 — SCIP Benders.** Refactor the builder into `build_master()` and `build_sub(i)`:
- *Master*: `x_0..x_{n-1}`, `u_p..u_q`, the monotonicity `u[k] >= u[k+1]` and `u[p]==1`. SCIP adds the auxiliary $\varphi_i$ itself.
- *Sub $i$*: **copies of `x`,`u` with identical names** (default Benders links by variable name), the block's flow/`zero_cost_flow` vars, and all per-string constraints (source, conservation, sink-balance (11), conflict, caps (10), gating (5)/(6)). Objective = the block's arc costs.
- Wire: `master.initBendersDefault({i: sub_i})`; mark each convex; `updateBendersLowerbounds({i: 0.0})` (distance $\ge 0$ keeps the master bounded before cuts). Relatively-complete recourse holds — a unit path always exists through the uncapacitated costly arcs — so feasibility cuts shouldn't fire; keep them enabled defensively and assert the count stays 0.
- Gate: same optimum as Phase 1.

**Phase 3 — (research, optional).** Only if profiling justifies it: (a) custom subproblem via DP, and/or (b) a custom `benderscut` injecting the sparse $\theta_i \ge \hat z_i - \sum_{e:\mu_e=1}(\dots)$ cut from forward/backward tables with Pareto selection. This is the cut-quality lever; expect it to matter far more than (a).

**Phase 4 — harness.** A driver that runs the chosen configs on a fixed instance subset (time limit ~120 s), and logs, per run: **solver time only** — `model.Runtime` for Gurobi, `getSolvingTime()` for SCIP, *not* wall-clock (their Python build loop is $O(\text{strings}\times\text{grid})$ with per-arc `addConstr` and will dominate wall-clock on small solves). Also log objective, gap, nodes, LP iters, and for Benders: #optimality cuts, #subproblem solves, #feasibility cuts (expect 0). Cross-check objectives agree, emit a CSV mirroring their schema, print a table.

## What I expect

1. **Parity, not speedups.** All three agree on the objective; medians vary. That's result #1 and it's the validation.

2. **On the paper's short instances, Benders loses — probably by a wide margin.** Their monolith already closes `rnd-5-*` in ~0.02–0.3 s; these are trivial for branch-and-cut. Decomposition overhead (master re-solves + LP subproblem solves + cut loop) is pure cost when there's nothing to decompose away. Likely ordering: Gurobi-mono < SCIP-mono < SCIP-Benders.

3. **The bottleneck will be master iterations, not subproblems.** Subproblem LPs are tiny; the cost is how many times the NP-hard master is re-optimized, which is governed by cut quality. The default duals are exactly the degenerate, dense, over-crediting cuts we discussed (one $x_k$ gates a whole column; many locally-tight slants get weight though a monotone path uses one). Expect slow root-bound improvement, many cuts, and a large fraction going inactive — visible in the cut counts.

4. **Scaling in $t$ is the only place Benders might claw back.** Hold length at 5–10, sweep `stringnumber` 10→50. The monolith grows ~linearly in $t$; the Benders master stays compact (≈$2q$ binaries + $t$ aux vars + cuts). I expect the *gap* to the monolith to narrow as $t$ grows, but I'd bet against a clean win with default cuts within 120 s — iteration count tends to grow with $t$ too. Finding the crossover (if any) is the empirical payoff.

5. **The real value is as a cut-management testbed for your B&BC line.** Trivial, well-understood subproblems plus an *analytically known good cut* make this an unusually clean setting to benchmark cut aging/filtering — including your prospective-filtering work — against a ground-truth-optimal cut. My prediction: swapping the default cut for the sparse single-alignment/Pareto $\mu$-cut cuts iteration count substantially (the headline experiment), and the default run produces a large, mostly-stale cut pool that's ideal for aging studies.

SCIP is installed here and I have the model in hand, so I can write `scip_mono.py`, `scip_benders.py`, and the harness now and smoke-test the Benders wiring on one small instance to confirm parity before you run the full sweep. Want me to do that?