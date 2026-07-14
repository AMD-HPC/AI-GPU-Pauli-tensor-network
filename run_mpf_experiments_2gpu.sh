#!/bin/bash
# run_mpf_experiments_2gpu.sh — Systematic MPF experiment suite (2-GPU variant)
#
# Same experiments as run_mpf_experiments.sh but using only 2 GPUs:
#   GPU 0: noiseless evolution
#   GPU 1: 3 noisy states (gamma=1.0, 1.4, 1.8)
#
# Intended for hardware with only 2 GPUs (e.g. vega20) or for
# running the same experiment matrix on a different GPU model.
#
# Naming: same as 3-GPU version for direct comparison.
#
# Usage: bash run_mpf_experiments_2gpu.sh [--dry-run]

DRY=${1:-""}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="$SCRIPT_DIR/logs/mpf_experiments_2gpu"
mkdir -p "$LOG_DIR"
echo "Logs directory: $LOG_DIR"

log() { echo "[$(date +%H:%M:%S)] $*"; }

# ── Build if needed ───────────────────────────────────────────────────────────
log "Building binaries..."
make PHYS_DIM=4 main_heavyhex_mpf main_heavyhex main_heavyhex_noisy -j3 2>&1 | tail -3

# ── Run one MPF experiment (2-GPU only) ──────────────────────────────────────
run_experiment() {
    local name=$1
    local cutoff=$2
    local center=$3
    local noise_power=$4
    local chi_nl=$5
    local chi_n=$6
    local steps=$7

    log "=== $name ==="
    log "  cutoff=$cutoff  center=$center  NP=$noise_power  chi_nl=$chi_nl  chi_n=$chi_n  steps=$steps"

    if [ "$DRY" = "--dry-run" ]; then
        log "  [DRY RUN — skipping]"
        return
    fi

    # Regenerate noise PTMs with correct NOISE_POWER
    NOISE_POWER=$noise_power python3 extract_noise_ptm.py > /dev/null 2>&1
    mv heavyhex_noise_data.hpp include/
    make PHYS_DIM=4 main_heavyhex_mpf main_heavyhex_noisy -j2 > /dev/null 2>&1

    local logfile="$LOG_DIR/${name}.log"
    local spatial_nl="$LOG_DIR/${name}_spatial_nl.dat"
    local spatial_n10="$LOG_DIR/${name}_spatial_n10.dat"
    local spatial_n14="$LOG_DIR/${name}_spatial_n14.dat"
    local spatial_n18="$LOG_DIR/${name}_spatial_n18.dat"

    # GPU 0+1: MPF run
    ROCR_VISIBLE_DEVICES=0,1 ./main_heavyhex_mpf \
        $chi_nl $chi_n $steps 0.1 1.0 $cutoff \
        --center=$center \
        > "$logfile" 2>&1
    log "  MPF done (exit $?)"

    # Noisy spatial weights — GPU 1
    for gamma in 1.0 1.4 1.8; do
        local sfile
        if   [ "$gamma" = "1.0" ]; then sfile="$spatial_n10"
        elif [ "$gamma" = "1.4" ]; then sfile="$spatial_n14"
        else                             sfile="$spatial_n18"
        fi
        ROCR_VISIBLE_DEVICES=1 ./main_heavyhex_noisy \
            $chi_n $steps 0.1 1.0 $cutoff \
            --gamma=$gamma --center=$center \
            --spatial="$sfile" \
            >> "$logfile.noisy" 2>&1
        log "    noisy gamma=$gamma done"
    done

    # Noiseless spatial — GPU 0
    ROCR_VISIBLE_DEVICES=0 ./main_heavyhex \
        $chi_nl $steps 0.1 1.0 $cutoff \
        --center=$center \
        --spatial="$spatial_nl" \
        >> "$logfile.nl" 2>&1
    log "    noiseless spatial done"

    # MPF extrapolation
    python3 mpf_extrapolate.py --data "$logfile" \
        --gammas 1.0 1.4 1.8 \
        >> "${logfile%.log}_mpf.log" 2>&1
    log "    MPF extrapolation done → ${name}_mpf.log"

    log "  === $name complete ==="
}

# ═══════════════════════════════════════════════════════════════════
# EXPERIMENT MATRIX (same as 3-GPU version, no GPU2 parallel run)
# ═══════════════════════════════════════════════════════════════════

# Wave 1: cutoff sensitivity
log "=== WAVE 1: Cutoff sensitivity ==="
run_experiment mpf_co3_ct2_np25_nl400_nn150_s10  1e-3  2  25  400  150  10
run_experiment mpf_co4_ct2_np25_nl400_nn150_s10  1e-4  2  25  400  150  10  # anchor
run_experiment mpf_co5_ct2_np25_nl400_nn150_s10  1e-5  2  25  400  150  10

# Wave 2: center site
log "=== WAVE 2: Center site ==="
run_experiment mpf_co4_ct13_np25_nl400_nn150_s10 1e-4  13  25  400  150  10
run_experiment mpf_co4_ct22_np25_nl400_nn150_s10 1e-4  22  25  400  150  10

# Wave 3: noise power + sanity
log "=== WAVE 3: Noise power + sanity ==="
run_experiment mpf_co4_ct2_np0_nl400_nn150_s10   1e-4   2   0  400  150  10
run_experiment mpf_co4_ct2_np1_nl400_nn150_s10   1e-4   2   1  400  150  10
run_experiment mpf_co4_ct2_np3_nl400_nn150_s10   1e-4   2   3  400  150  10
run_experiment mpf_co4_ct2_np10_nl400_nn150_s10  1e-4   2  10  400  150  10

# Wave 6: synthetic small gamma
log "=== WAVE 6: Synthetic small gamma ==="
run_experiment mpf_co4_ct2_npf2_nl400_nn150_s10  1e-4   2   2  400  150  10
run_experiment mpf_co4_ct2_npf5_nl400_nn150_s10  1e-4   2   5  400  150  10

# Wave 4: chi scaling
log "=== WAVE 4: Chi scaling ==="
run_experiment mpf_co4_ct2_np25_nl200_nn100_s10  1e-4   2  25  200  100  10
run_experiment mpf_co4_ct2_np25_nl300_nn120_s10  1e-4   2  25  300  120  10

# Wave 5: number of steps
log "=== WAVE 5: Steps ==="
run_experiment mpf_co4_ct2_np25_nl400_nn150_s20  1e-4   2  25  400  150  20
run_experiment mpf_co4_ct2_np25_nl400_nn150_s30  1e-4   2  25  400  150  30

log "=== ALL EXPERIMENTS COMPLETE ==="
log "Results in: $LOG_DIR/"
log ""
log "Analyze with:"
log "  python3 mpf_extrapolate.py --data $LOG_DIR/{name}.log"
log "  python3 mpf_extrapolate.py --spatial --nl ... --n10 ... --n14 ... --n18 ..."
