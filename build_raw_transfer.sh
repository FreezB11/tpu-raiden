#!/bin/bash
set -e

# Define directories
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" &> /dev/null && pwd)"
DEFAULT_WORKSPACE_DIR="$SCRIPT_DIR"
WORKSPACE_DIR="${WORKSPACE_DIR:-${DEFAULT_WORKSPACE_DIR}}"
BAZEL_DISK_CACHE="${BAZEL_CACHE_DIR:-/mnt/disks/jcgu/bazel_cache/disk_cache}"
BAZEL_REPO_CACHE="${BAZEL_CACHE_DIR:-/mnt/disks/jcgu/bazel_cache/repo_cache}"
RAW_TRANSFER_DIR="${WORKSPACE_DIR}/raw_transfer"

echo "=== Navigating to raw_transfer_lib directory ==="
cd "${RAW_TRANSFER_DIR}"

echo "=== Building raw_transfer with Bazel ==="
bazel build -c opt --check_visibility=false //:raw_transfer_binaries --disk_cache=${BAZEL_DISK_DIR} --repository_cache=${BAZEL_REPO_DIR}

echo "=== Build Complete! ==="
echo "Artifacts are located in: ${RAW_TRANSFER_DIR}/bazel-bin/"

echo "=== Install Python Dependencies! ==="
cd ${WORKSPACE_DIR}
pip install -r requirements.txt

echo "=== Installation Complete! ==="
