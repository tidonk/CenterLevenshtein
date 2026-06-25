# Integer programming models for the median of a 0-1 string set under Levenshtein distance

This repository contains the code of the formulations in  
**"Integer programming models for the median of a 0-1 string set under Levenshtein distance"**,  presented at the SEA 2026 conference.

## Experimental setup

### Hardware

All experiments were run on a cluster node equipped with
2&times;Intel Xeon L5630 (8 cores total), 16&thinsp;GB RAM.

### Software

| Component | Version |
|---|---|
| Python | 3.10.11 |
| SCIP | 10.0 |
| PySCIPOpt | 6.2.1 |
| Gurobi / gurobipy | 11.0 |

### Settings

Each instance was solved with a time limit of 600&thinsp;s per method.
The random seed was fixed to 2025 for all SCIP runs.
Gurobi results are pre-recorded reference solutions from the same seed.

The following methods are compared:

| Method | Description |
|---|---|
| Gurobi | Gurobi 11 monolithic MIP (reference) |
| SCIP (mono) | SCIP 10 monolithic MIP |
| Benders (full) | Custom Benders decomposition, all strings as subproblems |
| Benders (m/4) | Partial Benders, ceil(m/4) strings retained in master |
| Benders (2 rand.) | Partial Benders, 2 randomly chosen strings retained in master |
| Benders (adap. SP) | Benders with adaptive subproblem fraction (50% per round, `subprobfrac=0.5`) |

### Results

All 105 instances were run (75 random, 30 ECC). Summary over solved instances (shifted geometric mean of solving time, shift = 10 s):

| Method | # optimal / 105 | sh. geomean (s) |
|---|---|---|
| Gurobi | 55 | 18.16 |
| SCIP (mono) | 54 | 61.98 |
| Benders (full) | 50 | 39.58 |
| Benders (m/4) | 47 | 39.96 |
| Benders (2 rand.) | 54 | 49.45 |
| Benders (adap. SP) | 52 | 42.65 |

All four Benders variants are significantly faster than SCIP mono (Wilcoxon one-sided, p < 0.005).
Benders (full) and Benders (adap. SP) achieve the best shifted geometric mean (~40 s vs. 62 s for mono).
Benders (2 rand.) matches mono in number of instances solved while also being significantly faster.
No Benders variant significantly outperforms Gurobi.

Detailed results: [plots/instance_table_noGRB.md](plots/instance_table_noGRB.md) · [performance profile](plots/perf_profile_noGRB.png) · [interactive 3-D speedup plot](plots/corr_3d.html).

## Reproducing the experiments

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

`gurobipy` requires a valid Gurobi licence. Gurobi results are pre-recorded in `results/result_path_binary_2n.csv` and are not re-solved; SCIP runs require no licence.

### 2. Run solvers

**Single instance (local):**

```bash
python run.py random/I_10_30_0.txt 600
# logs  → logs/I_10_30_0/{GRB,mono,benders_full,benders_partial,benders_2rand,benders_adap_sp}.log
# result → results/I_10_30_0.csv
```

**Full benchmark (HTCondor cluster):**

```bash
# all 105 instances, all methods
condor_submit cluster/submit_full.sub

# benders_adap_sp in parallel (merges into existing CSVs)
condor_submit cluster/submit_adap_sp.sub

# monitor
condor_q -batch
```

### 3. Generate plots and tables

```bash
python3 visu/compare_analyze.py     # performance profiles + pairwise report
python3 visu/instance_table.py      # per-instance Markdown + LaTeX tables
python3 visu/correlation_plots.py   # interactive 3-D speedup plot (HTML)
```

All outputs are written to `plots/`.