#!/usr/bin/env sh

GOOGLE_LOG_DIR=models/two_layer_tree \
./build/tools/caffe train \
    --solver=models/two_layer_tree/solver.prototxt \
    --snapshot=models/two_layer_tree/tree_train_iter_30000.solverstate \
    --gpu=3
