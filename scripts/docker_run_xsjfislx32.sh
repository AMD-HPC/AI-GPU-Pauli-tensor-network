#!/bin/bash
# docker_run_xsjfislx32.sh — Start CppSim docker session on xsjfislx32
# Usage: bash scripts/docker_run_xsjfislx32.sh [command]
# If no command given: interactive shell

CMD=${@:-bash}

docker run --rm -it \
    --device=/dev/kfd \
    --device=/dev/dri \
    --group-add video \
    -v /scratch/MPF/CppSim:/scratch/MPF/CppSim \
    -v /scratch/.venv:/scratch/.venv \
    -w /scratch/MPF/CppSim \
    rocm/dev-ubuntu-22.04:7.0-complete \
    $CMD
