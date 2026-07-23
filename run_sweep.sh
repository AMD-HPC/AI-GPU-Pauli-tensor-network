#!/bin/bash
# run_sweep.sh — Systematic sweep of chi, cutoff, cycles for gamma/epsilon MPF
#
# Usage:
#   bash run_sweep.sh gamma   → gamma-MPF sweep
#   bash run_sweep.sh epsilon → epsilon-MPF sweep
#   bash run_sweep.sh all     → both
#
# Results saved to logs/sweep_YYYYMMDD_HHMMSS/

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODE="${1:-all}"
LOG_DIR="$SCRIPT_DIR/logs/sweep_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"

CHI_VALUES=(30 50 70 120)
CUTOFFS=(1e-2 1e-3 1e-4)
N_CYCLES=20

echo "=================================================="
echo "  CppSim MPF sweep — $(date)"
echo "  Mode: $MODE"
echo "  Chi values: ${CHI_VALUES[*]}"
echo "  Cutoffs: ${CUTOFFS[*]}"
echo "  Cycles: $N_CYCLES"
echo "  Logs: $LOG_DIR"
echo "=================================================="

run_experiment() {
    local binary=$1
    local chi=$2
    local cutoff=$3
    local tag="${binary}_chi${chi}_cut${cutoff}"
    local logfile="$LOG_DIR/${tag}.log"
    echo "  Running: $tag"
    "$SCRIPT_DIR/$binary" $chi $N_CYCLES $cutoff > "$logfile" 2>&1
    echo "  Done: $tag  ($(tail -1 $logfile))"
}

if [ "$MODE" = "gamma" ] || [ "$MODE" = "all" ]; then
    echo ""
    echo "── Gamma-MPF sweep ──────────────────────────────────"
    for chi in "${CHI_VALUES[@]}"; do
        for cutoff in "${CUTOFFS[@]}"; do
            run_experiment "main_heavyhex_gamma_mpf" $chi $cutoff
        done
    done
fi

if [ "$MODE" = "epsilon" ] || [ "$MODE" = "all" ]; then
    echo ""
    echo "── Epsilon-MPF sweep ────────────────────────────────"
    for chi in "${CHI_VALUES[@]}"; do
        for cutoff in "${CUTOFFS[@]}"; do
            run_experiment "main_heavyhex_eps_mpf" $chi $cutoff
        done
    done
fi

echo ""
echo "=================================================="
echo "  Sweep complete. Logs in: $LOG_DIR"
echo "=================================================="

# Generate CSV summary
echo ""
echo "Generating CSV summary..."
python3 - <<'EOF'
import os, re, sys, glob

log_dir = sorted(glob.glob('/scratch/MPF/CppSim/logs/sweep_*'))[-1]
print(f"Reading logs from: {log_dir}")

rows = []
for logfile in sorted(glob.glob(f"{log_dir}/*.log")):
    name = os.path.basename(logfile).replace('.log','')
    m = re.match(r'(main_heavyhex_\w+_mpf)_chi(\d+)_cut([\de.-]+)', name)
    if not m: continue
    binary, chi, cutoff = m.group(1), int(m.group(2)), m.group(3)
    mpf_type = 'gamma' if 'gamma' in binary else 'epsilon'

    with open(logfile) as f:
        content = f.read()

    for line in content.split('\n'):
        m2 = re.match(r'\s*(\d+)\s+([+-][\d.e+-]+)\s+([+-][\d.e+-]+)\s+([\d.e+-]+)', line)
        if not m2: continue
        step = int(m2.group(1))
        c_nl = float(m2.group(2))
        delta_nl = float(m2.group(4))

        # Extract C_mpf and |Δmpf|
        nums = re.findall(r'[+-]?[\d.]+e[+-]\d+|[+-][\d.]+', line)
        if len(nums) < 8: continue

        rows.append(f"{mpf_type},{chi},{cutoff},{step},{c_nl:.8f},{delta_nl:.2e}")

out = f"{log_dir}/sweep_results.csv"
with open(out, 'w') as f:
    f.write("mpf_type,chi,cutoff,step,C_nl,delta_nl\n")
    f.write('\n'.join(rows))
print(f"Written: {out}")
EOF
