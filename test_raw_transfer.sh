#!/bin/bash
set -e

WORKSPACE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" &> /dev/null && pwd)"

# Point to the directory containing the compiled raw_transfer.so
export PYTHONPATH="${WORKSPACE_DIR}/bazel-bin/raw_transfer:${PYTHONPATH}"

# Change to the tests directory to avoid Python's local directory import shadowing
cd "${WORKSPACE_DIR}/raw_transfer"

echo "=== Running: test_import.py ==="
python test_import.py 2>&1 | tee "${WORKSPACE_DIR}/import.log"

echo "=== Running: test_raw_transfer_perf.py ==="
python test_raw_transfer_perf.py 2>&1 | tee "${WORKSPACE_DIR}/perf_test.log"
