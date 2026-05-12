#!/bin/bash
set -e

# Define directories
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" &> /dev/null && pwd)"
DEFAULT_WORKSPACE_DIR="$SCRIPT_DIR"
WORKSPACE_DIR="${WORKSPACE_DIR:-${DEFAULT_WORKSPACE_DIR}}"
RAW_TRANSFER_DIR="${WORKSPACE_DIR}/raw_transfer"

export PYTHONPATH=$RAW_TRANSFER_DIR/bazel-bin:$PYTHONPATH

cd "${RAW_TRANSFER_DIR}"
echo "========================================"
echo "Running: test_import.py"
echo "========================================"
python test_import.py 2>&1 | tee ${WORKSPACE_DIR}/import.log

echo "========================================"
echo "Running: test_raw_transfer_perf.py"
echo "========================================"
python test_raw_transfer_perf.py 2>&1 | tee ${WORKSPACE_DIR}/perf_test.log