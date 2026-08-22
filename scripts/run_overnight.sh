#!/bin/bash
# Overnight run: HH 20 cycles + rect_7x7 dep data
# Run: bash scripts/run_overnight.sh
# Logs: logs/overnight_*/
# 3 GPUs used in parallel

set -e
mkdir -p logs

V_HOP_HH=0.500
V_HOP_R7=0.500
CENTER_HH=13
CENTER_R7=24
TS=$(date +%Y%m%d_%H%M%S)
LOGDIR="logs/overnight_${TS}"
mkdir -p $LOGDIR

echo "=== Overnight run started $(date) ==="
echo "Logs: $LOGDIR"

# ── Phase 3: Heavy-hex 20 cycles (all noise variants) ────────────────────────
echo "[HH 20c] Starting..."

# GPU 0: f32 hardware noise 20 cycles
ROCR_VISIBLE_DEVICES=0 ./main_pauli_sweep 200 20 1e-3 \
    --center=$CENTER_HH --v_hop=$V_HOP_HH \
    > $LOGDIR/A3_hh_200_20c_f32_hw.log 2>&1 &
PID0=$!

# GPU 1: f64 hardware noise 20 cycles
ROCR_VISIBLE_DEVICES=1 ./main_pauli_sweep_f64 200 20 1e-3 \
    --center=$CENTER_HH --v_hop=$V_HOP_HH \
    > $LOGDIR/A3_hh_200_20c_f64_hw.log 2>&1 &
PID1=$!

# GPU 2: f32 i.i.d. dep 20 cycles (this is what paper used — clean MPF)
ROCR_VISIBLE_DEVICES=2 ./main_pauli_sweep_hh_dep 200 20 1e-3 \
    --center=$CENTER_HH --v_hop=$V_HOP_HH \
    > $LOGDIR/A3_hh_200_20c_f32_dep.log 2>&1 &
PID2=$!

wait $PID0 $PID1 $PID2
echo "[HH 20c] DONE $(date)"

# ── Rect_7x7: use paper's working config (chi=50, dep noise, 10+20 cycles) ───
# rect_7x7 with hardware noise collapses at step 3.
# Paper used i.i.d. dep at chi=50 successfully.
echo "[R7 dep] Starting..."

# rect_7x7_noisy PTMs are already i.i.d. dep (IDEAL_KICK_EPS010_GAMMA1/2/3)
# so main_pauli_sweep_rect7x7 IS the dep version — no separate binary needed.

# GPU 0: r7 chi=50 10 cycles (i.i.d. dep — same as paper)
ROCR_VISIBLE_DEVICES=0 ./main_pauli_sweep_rect7x7 50 10 1e-3 \
    --center=$CENTER_R7 --v_hop=$V_HOP_R7 \
    > $LOGDIR/B2_r7_50_10c_f32_dep.log 2>&1 &
PID0=$!

# GPU 1: r7 chi=50 20 cycles
ROCR_VISIBLE_DEVICES=1 ./main_pauli_sweep_rect7x7 50 20 1e-3 \
    --center=$CENTER_R7 --v_hop=$V_HOP_R7 \
    > $LOGDIR/B3_r7_50_20c_f32_dep.log 2>&1 &
PID1=$!

# GPU 2: hh dep 10 cycles (shorter, for quick figure check)
ROCR_VISIBLE_DEVICES=2 ./main_pauli_sweep_hh_dep 200 10 1e-3 \
    --center=$CENTER_HH --v_hop=$V_HOP_HH \
    > $LOGDIR/A2_hh_200_10c_f32_dep.log 2>&1 &
PID2=$!

wait $PID0 $PID1 $PID2
echo "[R7 dep] DONE $(date)"

# ── Summary ───────────────────────────────────────────────────────────────────
echo ""
echo "=== Results Summary ==="
for f in $LOGDIR/*.log; do
    last=$(grep "^STEP" $f 2>/dev/null | tail -1)
    [ -z "$last" ] && echo "$(basename $f): ERROR/EMPTY" && continue
    echo "$(basename $f): $last"
done

echo ""
echo "=== Overnight run completed $(date) ==="
echo "Run analysis:"
echo "  source /scratch/.venv/bin/activate"
echo "  # HH dep figure (= paper Figure 7 at chi=200):"
echo "  python3 scripts/plot_mpf_entropy.py --log $LOGDIR/A2_hh_200_10c_f32_dep.log \\"
echo "      --graph data/graphs/heavy_hex_3x3.json --center 13 --threshold 1e-4 --layout graph"
echo "  # R7 dep figure:"
echo "  python3 scripts/plot_mpf_entropy.py --log $LOGDIR/B2_r7_50_10c_f32_dep.log \\"
echo "      --graph data/graphs/rect_7x7.json --center 24 --threshold 1e-4 --layout rect --nx 7 --ny 7"
echo ""
echo "  # Compare 10c vs 20c HH dep (frontier at different physical times):"
echo "  python3 scripts/analyze_frontier_mpf.py --log $LOGDIR/A3_hh_200_20c_f32_dep.log \\"
echo "      --graph data/graphs/heavy_hex_3x3.json --center 13 --threshold 1e-4"
