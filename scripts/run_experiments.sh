#!/bin/bash
# Frontier MPF Experiment Suite
# Usage: bash scripts/run_experiments.sh [phase]
# Phases: 1 (lightcone), 2 (hh_production), 3 (hh_long), 4 (hh_collapse), 5 (r7_probe)

set -e
mkdir -p logs

V_HOP_HH=0.500   # lightcone allocation for heavy-hex (2× physical frontier ~0.25)
V_HOP_R7=0.500   # same for rect_7x7 — adjust after B1 analysis

CENTER_HH=13
CENTER_R7=24

PHASE=${1:-all}

echo "=== CppSim Frontier MPF Experiments $(date) ==="
echo "Phase: $PHASE"

# ── Phase 1: Lightcone sweeps ─────────────────────────────────────────────────
run_phase1() {
    echo "[Phase 1] Lightcone sweeps — heavy-hex and rect_7x7"
    ROCR_VISIBLE_DEVICES=0 ./main_pauli_sweep 70 10 1e-3 \
        --center=$CENTER_HH --state=NL > logs/A1_hh_lightcone.log 2>&1 &
    ROCR_VISIBLE_DEVICES=1 ./main_pauli_sweep_rect7x7 70 10 1e-3 \
        --center=$CENTER_R7 --state=NL > logs/B1_r7_lightcone.log 2>&1 &
    ROCR_VISIBLE_DEVICES=2 ./main_pauli_sweep_f64 70 10 1e-3 \
        --center=$CENTER_HH --state=NL > logs/A1_hh_lightcone_f64.log 2>&1 &
    wait && echo "[Phase 1] DONE"
}

# ── Phase 2: Heavy-hex production (chi=200, 10 cycles) ───────────────────────
run_phase2() {
    echo "[Phase 2] Heavy-hex production runs — chi=200 delta=1e-3 10 cycles"
    # A2: f32 hardware noise
    ROCR_VISIBLE_DEVICES=0 ./main_pauli_sweep 200 10 1e-3 \
        --center=$CENTER_HH --v_hop=$V_HOP_HH \
        > logs/A2_hh_200_10c_f32.log 2>&1 &
    # A4: f64 hardware noise
    ROCR_VISIBLE_DEVICES=1 ./main_pauli_sweep_f64 200 10 1e-3 \
        --center=$CENTER_HH --v_hop=$V_HOP_HH \
        > logs/A4_hh_200_10c_f64.log 2>&1 &
    # A2_dep: f32 i.i.d. depolarizing
    ROCR_VISIBLE_DEVICES=2 ./main_pauli_sweep_hh_dep 200 10 1e-3 \
        --center=$CENTER_HH --v_hop=$V_HOP_HH \
        > logs/A2dep_hh_200_10c_f32.log 2>&1 &
    wait && echo "[Phase 2] DONE"
}

# ── Phase 3: Heavy-hex long runs (chi=200, 20 cycles) ────────────────────────
run_phase3() {
    echo "[Phase 3] Heavy-hex long runs — chi=200 delta=1e-3 20 cycles"
    ROCR_VISIBLE_DEVICES=0 ./main_pauli_sweep 200 20 1e-3 \
        --center=$CENTER_HH --v_hop=$V_HOP_HH \
        > logs/A3_hh_200_20c_f32.log 2>&1 &
    ROCR_VISIBLE_DEVICES=1 ./main_pauli_sweep_f64 200 20 1e-3 \
        --center=$CENTER_HH --v_hop=$V_HOP_HH \
        > logs/A3_hh_200_20c_f64.log 2>&1 &
    ROCR_VISIBLE_DEVICES=2 ./main_pauli_sweep_hh_dep 200 20 1e-3 \
        --center=$CENTER_HH --v_hop=$V_HOP_HH \
        > logs/A3dep_hh_200_20c_f32.log 2>&1 &
    wait && echo "[Phase 3] DONE"
}

# ── Phase 4: Heavy-hex collapse analysis (chi=380, delta=1e-4) ───────────────
run_phase4() {
    echo "[Phase 4] Heavy-hex collapse — chi=380 delta=1e-4 10 cycles"
    # A5: f32 hardware noise (known collapse at step 9)
    ROCR_VISIBLE_DEVICES=0 ./main_pauli_sweep 380 10 1e-4 \
        --center=$CENTER_HH --v_hop=$V_HOP_HH \
        > logs/A5_hh_380_10c_d1e4_f32.log 2>&1 &
    # A6: f32 hardware noise valid regime (delta=1e-3, no collapse)
    ROCR_VISIBLE_DEVICES=1 ./main_pauli_sweep 380 10 1e-3 \
        --center=$CENTER_HH --v_hop=$V_HOP_HH \
        > logs/A6_hh_380_10c_d1e3_f32.log 2>&1 &
    # A5_dep: i.i.d. collapse probe
    ROCR_VISIBLE_DEVICES=2 ./main_pauli_sweep_hh_dep 380 10 1e-4 \
        --center=$CENTER_HH --v_hop=$V_HOP_HH \
        > logs/A5dep_hh_380_10c_d1e4_f32.log 2>&1 &
    wait && echo "[Phase 4] DONE"
}

# ── Phase 5: Rect_7x7 probe (find safe chi) ──────────────────────────────────
run_phase5() {
    echo "[Phase 5] Rect_7x7 chi probe — find safe chi_max at delta=1e-3"
    # Probe different chi values to find safe operating point
    ROCR_VISIBLE_DEVICES=0 ./main_pauli_sweep_rect7x7 50 10 1e-3 \
        --center=$CENTER_R7 --v_hop=$V_HOP_R7 \
        > logs/B_r7_chi50_10c.log 2>&1 &
    ROCR_VISIBLE_DEVICES=1 ./main_pauli_sweep_rect7x7 40 10 1e-3 \
        --center=$CENTER_R7 --v_hop=$V_HOP_R7 \
        > logs/B_r7_chi40_10c.log 2>&1 &
    ROCR_VISIBLE_DEVICES=2 ./main_pauli_sweep_rect7x7 30 10 1e-3 \
        --center=$CENTER_R7 --v_hop=$V_HOP_R7 \
        > logs/B_r7_chi30_10c.log 2>&1 &
    wait && echo "[Phase 5] DONE"
    echo "Check C_nl for collapse:"
    for f in logs/B_r7_chi*.log; do
        echo "--- $f ---"
        grep "^STEP" $f | awk '{print $2,$4,"chi="$12,"gate_err="$16}'
    done
}

# ── Phase 7: Paper figure runs — NO --v_hop (full spatial signal required) ───
# IMPORTANT: MPF entropy figures require all sites active.
# --v_hop kills distant-site signal → entropy MPF = frontier MPF (degenerate).
# Use this phase for final paper-quality data.
run_phase7() {
    echo "[Phase 7] Paper figure runs — no v_hop, full spatial signal"

    # HH dep: the clean MPF figure (= paper Figure 7 at chi=200)
    ROCR_VISIBLE_DEVICES=0 ./main_pauli_sweep_hh_dep 200 10 1e-3 \
        --center=$CENTER_HH \
        > logs/PAPER_hh_dep_200_10c.log 2>&1 &

    # HH dep 20 cycles: new contribution (longer time evolution)
    ROCR_VISIBLE_DEVICES=1 ./main_pauli_sweep_hh_dep 200 20 1e-3 \
        --center=$CENTER_HH \
        > logs/PAPER_hh_dep_200_20c.log 2>&1 &

    # R7: paper-exact reproduction (chi=50, no v_hop — this IS the paper's log)
    # logs/spatial_mpf_rect7x7_chi50_1e-3_v2.log already exists!
    # Run at chi=50 for 20 cycles (new: longer time)
    ROCR_VISIBLE_DEVICES=2 ./main_pauli_sweep_rect7x7 50 20 1e-3 \
        --center=$CENTER_R7 \
        > logs/PAPER_r7_50_20c.log 2>&1 &

    wait && echo "[Phase 7] DONE"
    echo "Generate paper figures:"
    echo "  python3 scripts/plot_mpf_entropy.py --log logs/PAPER_hh_dep_200_10c.log --graph data/graphs/heavy_hex_3x3.json --center 13 --threshold 1e-4 --layout graph"
    echo "  python3 scripts/plot_mpf_entropy.py --log logs/PAPER_r7_50_20c.log --graph data/graphs/rect_7x7.json --center 24 --threshold 1e-4 --layout rect --nx 7 --ny 7"
}

# ── Phase 6: Rect_7x7 production (after safe chi known) ─────────────────────
run_phase6() {
    CHI_R7=${2:-40}  # pass safe chi as second arg
    echo "[Phase 6] Rect_7x7 production — chi=$CHI_R7 delta=1e-3"
    ROCR_VISIBLE_DEVICES=0 ./main_pauli_sweep_rect7x7 $CHI_R7 10 1e-3 \
        --center=$CENTER_R7 --v_hop=$V_HOP_R7 \
        > logs/B2_r7_${CHI_R7}_10c_f32.log 2>&1 &
    ROCR_VISIBLE_DEVICES=1 ./main_pauli_sweep_rect7x7 $CHI_R7 20 1e-3 \
        --center=$CENTER_R7 --v_hop=$V_HOP_R7 \
        > logs/B3_r7_${CHI_R7}_20c_f32.log 2>&1 &
    ROCR_VISIBLE_DEVICES=2 ./main_pauli_sweep_f64_rect7x7 $CHI_R7 10 1e-3 \
        --center=$CENTER_R7 --v_hop=$V_HOP_R7 \
        > logs/B4_r7_${CHI_R7}_10c_f64.log 2>&1 &
    wait && echo "[Phase 6] DONE"
}

# ── Summary ───────────────────────────────────────────────────────────────────
show_summary() {
    echo "=== Results Summary ==="
    for f in logs/*.log; do
        last=$(grep "^STEP" $f 2>/dev/null | tail -1)
        [ -z "$last" ] && continue
        echo "$(basename $f): $last"
    done
}

# ── Dispatch ──────────────────────────────────────────────────────────────────
case $PHASE in
    1)    run_phase1 ;;
    2)    run_phase2 ;;
    3)    run_phase3 ;;
    4)    run_phase4 ;;
    5)    run_phase5 ;;
    6)    run_phase6 $2 ;;
    7)    run_phase7 ;;
    summary) show_summary ;;
    all)
        run_phase1
        run_phase2
        run_phase3
        run_phase4
        run_phase5
        ;;
    *)  echo "Usage: $0 [1|2|3|4|5|6|summary]" ;;
esac

echo "=== ALL DONE $(date) ==="
