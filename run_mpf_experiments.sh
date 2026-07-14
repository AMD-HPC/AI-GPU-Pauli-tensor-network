#!/bin/bash
# run_mpf_experiments.sh — Systematic MPF experiment suite
#
# GPU 0+1: MPF run (noiseless + noisy x3 + overlaps + spatial weights)
# GPU 2:   Independent noiseless lightcone (different center or parameter)
#
# All runs save:
#   - C(t) + overlaps: logs/mpf_experiments/{name}.log
#   - Spatial Pauli weights noiseless: logs/mpf_experiments/{name}_spatial_nl.dat
#   - Spatial Pauli weights noisy γ=1.0/1.4/1.8: via main_heavyhex_noisy on GPU2
#
# Naming: mpf_co{cutoff_exp}_ct{center}_np{noise_power}_nl{chi_nl}_nn{chi_n}_s{steps}
#   e.g. mpf_co4_ct2_np25_nl400_nn150_s10
#
# Usage: bash run_mpf_experiments.sh [--dry-run]

DRY=${1:-""}
mkdir -p logs/mpf_experiments

log() { echo "[$(date +%H:%M:%S)] $*"; }

# ── Build if needed ───────────────────────────────────────────────────────────
log "Building binaries..."
make PHYS_DIM=4 main_heavyhex_mpf main_heavyhex main_heavyhex_noisy -j3 2>&1 | tail -3

# ── Run one MPF experiment ─────────────────────────────────────────────────
# GPU 0+1: MPF (noiseless + 3 noisy), GPU 2: noiseless spatial for extra center
run_experiment() {
    local name=$1
    local cutoff=$2
    local center=$3
    local noise_power=$4
    local chi_nl=$5
    local chi_n=$6
    local steps=$7
    local center2=${8:-13}   # GPU 2 center site (different from main center)

    log "=== $name ==="
    log "  cutoff=$cutoff  center=$center  NP=$noise_power  chi_nl=$chi_nl  chi_n=$chi_n  steps=$steps"
    log "  GPU2: noiseless center=$center2"

    if [ "$DRY" = "--dry-run" ]; then
        log "  [DRY RUN — skipping]"
        return
    fi

    # Regenerate noise PTMs with correct NOISE_POWER
    NOISE_POWER=$noise_power python3 extract_noise_ptm.py > /dev/null 2>&1
    mv heavyhex_noise_data.hpp include/
    make PHYS_DIM=4 main_heavyhex_mpf main_heavyhex_noisy -j2 > /dev/null 2>&1

    local logfile="logs/mpf_experiments/${name}.log"
    local spatial_nl="logs/mpf_experiments/${name}_spatial_nl.dat"
    local spatial_n10="logs/mpf_experiments/${name}_spatial_n10.dat"
    local spatial_n14="logs/mpf_experiments/${name}_spatial_n14.dat"
    local spatial_n18="logs/mpf_experiments/${name}_spatial_n18.dat"
    local spatial_gpu2="logs/mpf_experiments/${name}_spatial_gpu2_ct${center2}.dat"

    # GPU 0+1: MPF run (noiseless + 3 noisy overlaps)
    # We add spatial output for noiseless via main_heavyhex on GPU0 separately
    ROCR_VISIBLE_DEVICES=0,1 ./main_heavyhex_mpf \
        $chi_nl $chi_n $steps 0.1 1.0 $cutoff \
        --center=$center \
        > "$logfile" 2>&1 &
    PID_MPF=$!

    # GPU 2: noiseless spatial (different center for free)
    ROCR_VISIBLE_DEVICES=2 ./main_heavyhex \
        $chi_nl $steps 0.1 1.0 $cutoff \
        --center=$center2 \
        --spatial="$spatial_gpu2" \
        >> "$logfile.gpu2" 2>&1 &
    PID_GPU2=$!

    wait $PID_MPF
    log "  MPF done (exit $?)"
    wait $PID_GPU2
    log "  GPU2 noiseless done (exit $?)"

    # Now collect noisy spatial weights — run 3 gammas sequentially on GPU 1
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

    # Also save noiseless spatial from GPU 0 (center=center)
    ROCR_VISIBLE_DEVICES=0 ./main_heavyhex \
        $chi_nl $steps 0.1 1.0 $cutoff \
        --center=$center \
        --spatial="$spatial_nl" \
        >> "$logfile.nl" 2>&1
    log "    noiseless spatial done"

    # Run MPF extrapolation immediately
    python3 mpf_extrapolate.py --data "$logfile" \
        --gammas 1.0 1.4 1.8 \
        >> "${logfile%.log}_mpf.log" 2>&1
    log "    MPF extrapolation done → ${name}_mpf.log"

    log "  === $name complete ==="
}

# ═══════════════════════════════════════════════════════════════════
# EXPERIMENT MATRIX
# ═══════════════════════════════════════════════════════════════════
#
# Anchor: co4_ct2_np25_nl400_nn150_s10
# name                         cutoff  center NP   nl   nn   steps  gpu2_center
# ──────────────────────────── ─────── ────── ──── ──── ──── ─────  ───────────

# Wave 1: cutoff sensitivity
log "=== WAVE 1: Cutoff sensitivity ==="
run_experiment mpf_co3_ct2_np25_nl400_nn150_s10  1e-3  2  25  400  150  10  13
run_experiment mpf_co4_ct2_np25_nl400_nn150_s10  1e-4  2  25  400  150  10  13  # anchor
run_experiment mpf_co5_ct2_np25_nl400_nn150_s10  1e-5  2  25  400  150  10  13

# Wave 2: center site
log "=== WAVE 2: Center site ==="
run_experiment mpf_co4_ct13_np25_nl400_nn150_s10 1e-4  13  25  400  150  10   2
run_experiment mpf_co4_ct22_np25_nl400_nn150_s10 1e-4  22  25  400  150  10  13

# Wave 3: noise power
log "=== WAVE 3: Noise power ==="
run_experiment mpf_co4_ct2_np1_nl400_nn150_s10   1e-4   2   1  400  150  10  13
run_experiment mpf_co4_ct2_np3_nl400_nn150_s10   1e-4   2   3  400  150  10  13
run_experiment mpf_co4_ct2_np10_nl400_nn150_s10  1e-4   2  10  400  150  10  13

# Wave 4: chi scaling
log "=== WAVE 4: Chi scaling ==="
run_experiment mpf_co4_ct2_np25_nl200_nn100_s10  1e-4   2  25  200  100  10  13
run_experiment mpf_co4_ct2_np25_nl300_nn120_s10  1e-4   2  25  300  120  10  13

# Wave 5: number of steps
log "=== WAVE 5: Steps ==="
run_experiment mpf_co4_ct2_np25_nl400_nn150_s20  1e-4   2  25  400  150  20  13
run_experiment mpf_co4_ct2_np25_nl400_nn150_s30  1e-4   2  25  400  150  30  13

log "=== ALL EXPERIMENTS COMPLETE ==="
log "Results in: logs/mpf_experiments/"
log ""
log "Analyze with:"
log "  python3 mpf_extrapolate.py --data logs/mpf_experiments/{name}.log"
log "  python3 mpf_extrapolate.py --spatial --nl ... --n10 ... --n14 ... --n18 ..."
