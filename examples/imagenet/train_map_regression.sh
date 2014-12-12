#!/usr/bin/env sh

GOOGLE_LOG_DIR=models/bvlc_reference_caffenet/log \
./build/tools/caffe train \
    --solver=models/bvlc_reference_caffenet/map_regress_solver.prototxt \
    --gpu=3
