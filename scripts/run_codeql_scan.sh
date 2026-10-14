#!/bin/bash
# run_codeql_scan.sh — Run CodeQL SAST scan for OSRB security approval
#
# Requirements:
#   1. Download CodeQL bundle (Linux):
#      wget https://mkmartifactory.amd.com:8443/artifactory/system-provisioning/CodeQL/bundles/latest/codeql-bundle-linux64.tar.gz
#      tar xzf codeql-bundle-linux64.tar.gz
#      export PATH=$PATH:$(pwd)/codeql
#
# Usage:
#   bash scripts/run_codeql_scan.sh [output_dir]

REPO_ROOT="/scratch/MPF/CppSim"
OUTPUT_DIR=${1:-"$REPO_ROOT/codeql_results"}
DB_DIR="$OUTPUT_DIR/db"

mkdir -p "$OUTPUT_DIR" "$DB_DIR"
cd "$REPO_ROOT"

echo "=== CodeQL SAST Scan for AI-GPU-Pauli-tensor-network ==="
echo "Repo: $REPO_ROOT"
echo "Output: $OUTPUT_DIR"
echo ""

# ── Step 1: C/C++ scan ────────────────────────────────────────────────────────
echo "--- Step 1: C/C++ database creation ---"
codeql database create "$DB_DIR/cpp_db" \
    --language=cpp \
    --source-root="$REPO_ROOT" \
    --command="make main_pauli_sweep_rect7x7" \
    --overwrite

echo ""
echo "--- Step 2: C/C++ analysis (cpp-security-extended) ---"
codeql database analyze "$DB_DIR/cpp_db" \
    --format=csv \
    --output="$OUTPUT_DIR/cpp_results.csv" \
    cpp-security-extended.qls \
    BannedFunctions.ql

echo "C/C++ results: $OUTPUT_DIR/cpp_results.csv"

# ── Step 2: Python scan ───────────────────────────────────────────────────────
echo ""
echo "--- Step 3: Python database creation ---"
codeql database create "$DB_DIR/python_db" \
    --language=python \
    --source-root="$REPO_ROOT" \
    --overwrite

echo ""
echo "--- Step 4: Python analysis (python-security-extended) ---"
codeql database analyze "$DB_DIR/python_db" \
    --format=csv \
    --output="$OUTPUT_DIR/python_results.csv" \
    python-security-extended.qls

echo "Python results: $OUTPUT_DIR/python_results.csv"

echo ""
echo "=== DONE ==="
echo "Results in: $OUTPUT_DIR/"
echo "Upload SARIF to GitHub or send CSV to dl.AppliedAI-security@amd.com"
