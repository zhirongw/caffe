#!/usr/bin/env sh

GOOGLE_LOG_DIR=models/hierarchical10 \
mpirun -np 4 ./build/tools/caffe train \
    --solver=models/hierarchical10/solver.prototxt \
#    --gpu=3
