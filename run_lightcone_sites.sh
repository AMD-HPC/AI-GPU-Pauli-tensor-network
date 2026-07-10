#!/bin/bash
# run_lightcone_sites.sh — lightcone from 3 different center sites on 3 GPUs
#
# Sites chosen on heavy_hex_3x3 (0-based):
#   GPU 0: site 2  — degree-3 junction, graph center     (dist=0)
#   GPU 1: site 13 — degree-3 junction, mid-distance     (dist=8)
#   GPU 2: site 22 — degree-3 junction, far corner       (dist=12)
#
# Usage: bash run_lightcone_sites.sh [chi_max] [n_steps] [dt] [J] [cutoff]

CHI=${1:-200}
N_STEPS=${2:-10}
DT=${3:-0.1}
J=${4:-1.0}
CUTOFF=${5:-1e-4}
QR=${6:---qr=householder}

mkdir -p logs/lightcone_sites

echo "Starting lightcone sites: chi=$CHI  n_steps=$N_STEPS  dt=$DT  J=$J  cutoff=$CUTOFF"
echo "Centers: site2 (center), site13 (mid), site22 (far corner)"
echo "Logs: logs/lightcone_sites/site_{2,13,22}.log"

ROCR_VISIBLE_DEVICES=0 ./main_heavyhex $CHI $N_STEPS $DT $J $CUTOFF $QR \
    --spatial=logs/lightcone_sites/spatial_site2.dat \
    > logs/lightcone_sites/site2.log 2>&1 &
PID0=$!

ROCR_VISIBLE_DEVICES=1 ./main_heavyhex $CHI $N_STEPS $DT $J $CUTOFF $QR \
    --center=13 \
    --spatial=logs/lightcone_sites/spatial_site13.dat \
    > logs/lightcone_sites/site13.log 2>&1 &
PID1=$!

ROCR_VISIBLE_DEVICES=2 ./main_heavyhex $CHI $N_STEPS $DT $J $CUTOFF $QR \
    --center=22 \
    --spatial=logs/lightcone_sites/spatial_site22.dat \
    > logs/lightcone_sites/site22.log 2>&1 &
PID2=$!

echo "PIDs: site2:$PID0  site13:$PID1  site22:$PID2"

wait $PID0; echo "site2 done (exit $?)"
wait $PID1; echo "site13 done (exit $?)"
wait $PID2; echo "site22 done (exit $?)"

echo "All done. Results in logs/lightcone_sites/"
