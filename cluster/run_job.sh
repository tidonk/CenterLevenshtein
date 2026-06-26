#!/usr/bin/env bash
# HTCondor job runner — one job per instance, all four methods in sequence.
# Usage (invoked by HTCondor): run_job.sh <instance_path> [time_limit]
#   instance_path  relative to project root, e.g. random/I_10_30_0.txt
#   time_limit     seconds per method (default 600)
#
# Expects PROJECT_ROOT to be set via the submit file's environment stanza.
# Outputs (written into the project tree, which is NFS-mounted on all nodes):
#   logs/<instance_stem>/{GRB,mono,benders_full,benders_partial}.log
#   results/<instance_stem>.csv

set -euo pipefail

INSTANCE="${1:?first argument must be instance path}"
TIME_LIMIT="${2:-600}"
# optional: specific methods passed as further positional args (e.g. benders_adap_sp)
REQUESTED_METHODS=("${@:3}")

cd "${PROJECT_ROOT:?PROJECT_ROOT env var not set — check submit file}"

# Prefer the project venv; fall back to system python3 if the venv binary
# can't execute on this node (GLIBC version mismatch on older workers).
PYTHON="${PROJECT_ROOT}/.venv/bin/python3"
if ! "${PYTHON}" -c "pass" 2>/dev/null; then
    PYTHON=python3
fi

# Default methods must stay in sync with ALL_METHODS in src/run_methods.py.
if [[ ${#REQUESTED_METHODS[@]} -eq 0 ]]; then
    REQUESTED_METHODS=(GRB mono benders_full benders_partial benders_2rand benders_adap_sp)
fi

# Check which methods are missing from the results CSV.
STEM="$(basename "${INSTANCE}" .txt)"
CSV="${PROJECT_ROOT}/results/${STEM}.csv"

MISSING_METHODS=()
if [[ -f "${CSV}" ]]; then
    for method in "${REQUESTED_METHODS[@]}"; do
        if ! awk -F, -v m="${method}" 'NR>1 && $5==m {found=1; exit} END{exit !found}' "${CSV}"; then
            MISSING_METHODS+=("${method}")
        fi
    done
else
    MISSING_METHODS=("${REQUESTED_METHODS[@]}")
fi

if [[ ${#MISSING_METHODS[@]} -eq 0 ]]; then
    echo "=== skipping ${INSTANCE}: all methods already in ${CSV} ==="
    exit 0
fi

echo "=== job start ==="
echo "instance  : ${INSTANCE}"
echo "time_limit: ${TIME_LIMIT}s"
echo "methods   : ${MISSING_METHODS[*]}"
echo "host      : $(hostname)"
echo "python    : $(${PYTHON} --version 2>&1)"
echo "date      : $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo ""

"${PYTHON}" run.py "${INSTANCE}" "${TIME_LIMIT}" "${MISSING_METHODS[@]}"

echo ""
echo "=== job done: $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
