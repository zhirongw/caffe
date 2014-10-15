#!/usr/bin/env sh

#GOOGLE_LOG_DIR=models/two_layer_tree \
./build/tools/caffe train \
    --solver=models/mpi_two_layer_tree/solver.prototxt \
    --gpu=0
