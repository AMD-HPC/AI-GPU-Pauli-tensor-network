#!/bin/bash
# run_chi_scaling.sh — chi scaling benchmark on 3 GPUs simultaneously
# GPU 0: chi=200  GPU 1: chi=300  GPU 2: chi=400
# Usage: bash run_chi_scaling.sh [n_steps] [dt] [J] [cutoff]

N_STEPS=${1:-10}
DT=${2:-0.1}
J=${3:-1.0}
CUTOFF=${4:-1e-4}
QR=${5:---qr=householder}

mkdir -p logs/chi_scaling

echo "Starting chi scaling: chi=200,300,400  n_steps=$N_STEPS  dt=$DT  J=$J  cutoff=$CUTOFF"
echo "Logs: logs/chi_scaling/chi_{200,300,400}.log"

ROCR_VISIBLE_DEVICES=0 ./main_heavyhex 200 $N_STEPS $DT $J $CUTOFF $QR \
    --spatial=logs/chi_scaling/spatial_chi200.dat \
    > logs/chi_scaling/chi_200.log 2>&1 &
PID0=$!

ROCR_VISIBLE_DEVICES=1 ./main_heavyhex 300 $N_STEPS $DT $J $CUTOFF $QR \
    --spatial=logs/chi_scaling/spatial_chi300.dat \
    > logs/chi_scaling/chi_300.log 2>&1 &
PID1=$!

ROCR_VISIBLE_DEVICES=2 ./main_heavyhex 400 $N_STEPS $DT $J $CUTOFF $QR \
    --spatial=logs/chi_scaling/spatial_chi400.dat \
    > logs/chi_scaling/chi_400.log 2>&1 &
PID2=$!

echo "PIDs: chi=200:$PID0  chi=300:$PID1  chi=400:$PID2"
echo "Monitor: tail -f logs/chi_scaling/chi_200.log"

wait $PID0; echo "chi=200 done (exit $?)"
wait $PID1; echo "chi=300 done (exit $?)"
wait $PID2; echo "chi=400 done (exit $?)"

echo "All done. Results in logs/chi_scaling/"
