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

cd "${PROJECT_ROOT:?PROJECT_ROOT env var not set — check submit file}"

# Prefer the project venv; fall back to system python3 if the venv binary
# can't execute on this node (GLIBC version mismatch on older workers).
PYTHON="${PROJECT_ROOT}/.venv/bin/python3"
if ! "${PYTHON}" -c "pass" 2>/dev/null; then
    PYTHON=python3
fi

echo "=== job start ==="
echo "instance  : ${INSTANCE}"
echo "time_limit: ${TIME_LIMIT}s"
echo "host      : $(hostname)"
echo "python    : $(${PYTHON} --version 2>&1)"
echo "date      : $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo ""

"${PYTHON}" run.py "${INSTANCE}" "${TIME_LIMIT}"

echo ""
echo "=== job done: $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
