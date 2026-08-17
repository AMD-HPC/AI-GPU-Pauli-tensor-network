#!/bin/bash
# run_chi_sweep.sh — Lightcone-aware chi maximization sweep
#
# Protocol:
#   Step 1: Reference run — largest affordable chi, NL only, full allocation
#   Step 2: Frontier calibration — find minimum chi for faithful frontier
#   Step 3: Production run — maximized chi with lightcone allocation
#   Step 4: Compare C(t) reference vs production
#
# Usage:
#   bash scripts/run_chi_sweep.sh \
#       --graph data/graphs/heavy_hex_3x3.json \
#       --center 13 \
#       --n_steps 10 \
#       --cutoff 1e-3 \
#       --chi_ref 400 \
#       --threshold 1e-4 \
#       --gpu_ref 0 \
#       --gpu_cal 1 \
#       --outdir logs/chi_sweep_$(date +%Y%m%d_%H%M%S)
#
# Output:
#   $outdir/step1_reference.log     — reference run (chi_ref, NL, full)
#   $outdir/step2_cal_chi*.log      — calibration runs
#   $outdir/step2_vhop.txt          — frontier calibration results
#   $outdir/step3_production_NL.log — production NL
#   $outdir/step3_production_N*.log — production noisy states
#   $outdir/step3_combined.log      — merged for MPF scripts
#   $outdir/summary.txt             — C(t) comparison + chi convergence

set -e

# ── Defaults ──────────────────────────────────────────────────────────────────
GRAPH="data/graphs/heavy_hex_3x3.json"
CENTER=13
N_STEPS=10
CUTOFF=1e-3
CHI_REF=400
THRESHOLD=1e-4
GPU_REF=0
GPU_CAL=1
GPU2=2
OUTDIR="logs/chi_sweep_$(date +%Y%m%d_%H%M%S)"
BINARY="./main_pauli_sweep"

# ── Parse arguments ───────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case $1 in
        --graph)    GRAPH=$2;     shift 2 ;;
        --center)   CENTER=$2;    shift 2 ;;
        --n_steps)  N_STEPS=$2;   shift 2 ;;
        --cutoff)   CUTOFF=$2;    shift 2 ;;
        --chi_ref)  CHI_REF=$2;   shift 2 ;;
        --threshold) THRESHOLD=$2; shift 2 ;;
        --gpu_ref)  GPU_REF=$2;   shift 2 ;;
        --gpu_cal)  GPU_CAL=$2;   shift 2 ;;
        --outdir)   OUTDIR=$2;    shift 2 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

mkdir -p "$OUTDIR"
echo "Output directory: $OUTDIR"
echo "Graph: $GRAPH  center=$CENTER  n_steps=$N_STEPS  cutoff=$CUTOFF"
echo "Chi_ref=$CHI_REF  threshold=$THRESHOLD"
echo ""

# ── Step 1: Reference run ─────────────────────────────────────────────────────
echo "============================================================"
echo "Step 1: Reference run (chi=$CHI_REF, NL only, full alloc)"
echo "============================================================"
REF_LOG="$OUTDIR/step1_reference.log"
ROCR_VISIBLE_DEVICES=$GPU_REF \
    $BINARY $CHI_REF $N_STEPS $CUTOFF \
    --center=$CENTER --state=NL \
    > "$REF_LOG"
echo "Reference log: $REF_LOG"
grep "^STEP" "$REF_LOG" | tail -5
echo ""

# ── Step 2: Frontier calibration ──────────────────────────────────────────────
echo "============================================================"
echo "Step 2: Frontier calibration (NL only, increasing chi)"
echo "============================================================"
VHOP_FILE="$OUTDIR/step2_vhop.txt"
echo "chi  max_shell  v_hop_eff  v_hop_slope" > "$VHOP_FILE"

PREV_SHELL=-1
CHI_FRONTIER=-1
# Calibration chi sequence — cheap, NL only
for CHI_CAL in 10 20 30 50 70 100; do
    CAL_LOG="$OUTDIR/step2_cal_chi${CHI_CAL}.log"
    echo -n "  chi=$CHI_CAL ... "
    ROCR_VISIBLE_DEVICES=$GPU_CAL \
        $BINARY $CHI_CAL $N_STEPS $CUTOFF \
        --center=$CENTER --state=NL \
        > "$CAL_LOG" 2>/dev/null

    # Extract v_hop from this run
    RESULT=$(python3 scripts/measure_vhop.py \
        --log "$CAL_LOG" \
        --graph "$GRAPH" \
        --center $CENTER \
        --threshold $THRESHOLD 2>/dev/null | grep "^  NL:")

    MAX_SHELL=$(echo "$RESULT" | grep -o 'max_shell=[0-9]*' | cut -d= -f2)
    V_EFF=$(echo "$RESULT"    | grep -o 'v_hop_eff=[0-9.]*' | cut -d= -f2)
    V_SLOPE=$(echo "$RESULT"  | grep -o 'v_hop_slope=[0-9.]*' | cut -d= -f2)

    echo "max_shell=$MAX_SHELL  v_hop_eff=$V_EFF"
    echo "$CHI_CAL  $MAX_SHELL  $V_EFF  $V_SLOPE" >> "$VHOP_FILE"

    # Convergence check: frontier stable for 2 consecutive chi values
    if [ "$MAX_SHELL" = "$PREV_SHELL" ] && [ "$CHI_FRONTIER" -lt 0 ]; then
        CHI_FRONTIER=$CHI_CAL
        V_HOP_CONVERGED=$V_EFF
        echo "  --> Frontier converged at chi=$CHI_FRONTIER (max_shell=$MAX_SHELL, v_hop=$V_HOP_CONVERGED)"
    fi
    PREV_SHELL=$MAX_SHELL
done

# Use last measured if never converged
if [ "$CHI_FRONTIER" -lt 0 ]; then
    CHI_FRONTIER=$CHI_CAL
    V_HOP_CONVERGED=$V_EFF
    echo "  --> Using chi=$CHI_FRONTIER (v_hop=$V_HOP_CONVERGED)"
fi
echo ""
cat "$VHOP_FILE"
echo ""

# ── Step 3: Production run ────────────────────────────────────────────────────
echo "============================================================"
echo "Step 3: Production run (shared lightcone v_hop=$V_HOP_CONVERGED)"
echo "============================================================"

# Estimate production chi — memory saved by lightcone allows larger chi
# Conservative estimate: use same chi_ref for now (user can tune)
CHI_PROD=$CHI_REF
echo "Production chi=$CHI_PROD  v_hop=$V_HOP_CONVERGED"
echo "(To maximize chi: estimate VRAM savings and scale chi accordingly)"
echo ""

# Run 3 states on 3 GPUs in parallel, then N3 sequential
echo "Running NL+N1+N2 in parallel on GPUs 0,1,2..."
ROCR_VISIBLE_DEVICES=0 \
    $BINARY $CHI_PROD $N_STEPS $CUTOFF \
    --center=$CENTER --state=NL --v_hop=$V_HOP_CONVERGED \
    > "$OUTDIR/step3_production_NL.log" &
PID_NL=$!

ROCR_VISIBLE_DEVICES=1 \
    $BINARY $CHI_PROD $N_STEPS $CUTOFF \
    --center=$CENTER --state=N1 --v_hop=$V_HOP_CONVERGED \
    > "$OUTDIR/step3_production_N1.log" &
PID_N1=$!

ROCR_VISIBLE_DEVICES=2 \
    $BINARY $CHI_PROD $N_STEPS $CUTOFF \
    --center=$CENTER --state=N2 --v_hop=$V_HOP_CONVERGED \
    > "$OUTDIR/step3_production_N2.log" &
PID_N2=$!

wait $PID_NL $PID_N1 $PID_N2
echo "NL+N1+N2 done. Running N3..."

ROCR_VISIBLE_DEVICES=0 \
    $BINARY $CHI_PROD $N_STEPS $CUTOFF \
    --center=$CENTER --state=N3 --v_hop=$V_HOP_CONVERGED \
    > "$OUTDIR/step3_production_N3.log"
echo "N3 done."

# Merge logs
COMBINED="$OUTDIR/step3_combined.log"
cat "$OUTDIR"/step3_production_*.log > "$COMBINED"
echo "Combined log: $COMBINED"
echo ""

# ── Step 4: Compare ───────────────────────────────────────────────────────────
echo "============================================================"
echo "Step 4: C(t) comparison — reference vs production"
echo "============================================================"

SUMMARY="$OUTDIR/summary.txt"
{
    echo "Chi sweep summary"
    echo "Graph: $GRAPH  center=$CENTER  n_steps=$N_STEPS"
    echo ""
    echo "Frontier calibration:"
    cat "$VHOP_FILE"
    echo "Converged: chi=$CHI_FRONTIER  v_hop=$V_HOP_CONVERGED"
    echo ""
    echo "C(t) comparison (NL state):"
    echo "Step   C_nl_ref         C_nl_prod        delta"
    paste \
        <(grep "^STEP" "$REF_LOG"       | awk '{print $2, $4}') \
        <(grep "^STEP" "$OUTDIR/step3_production_NL.log" | awk '{print $4}') | \
    awk '{printf "%-6s %-16s %-16s %+.2e\n", $1, $2, $3, $3-$2}'
    echo ""
    echo "Timing comparison:"
    echo "  Reference (full, NL, chi=$CHI_REF):"
    grep "STEP $N_STEPS " "$REF_LOG" | awk '{print "    Total:", $NF, "s"}'
    echo "  Production (lightcone, NL, chi=$CHI_PROD, v_hop=$V_HOP_CONVERGED):"
    grep "STEP $N_STEPS " "$OUTDIR/step3_production_NL.log" | awk '{print "    Total:", $NF, "s"}'
} | tee "$SUMMARY"

echo ""
echo "All done. Summary: $SUMMARY"
echo "MPF analysis: python3 scripts/plot_mpf_entropy.py --log $COMBINED ..."
