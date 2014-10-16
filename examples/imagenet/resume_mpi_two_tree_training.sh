#!/usr/bin/env sh

GOOGLE_LOG_DIR=models/mpi_two_layer_tree \
mpirun -np 3 ./build/tools/caffe train \
    --solver=models/mpi_two_layer_tree/solver.prototxt \
    --snapshot=models/mpi_two_layer_tree/tree_train_iter_15000.solverstate \
