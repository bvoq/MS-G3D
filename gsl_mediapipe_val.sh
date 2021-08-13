#!/bin/sh

python3 main2.py --config ./config/gsl_mediapipe/test_joint.yaml --work-dir pretrain_eval/gsl_mediapipe_wbg_newmethod --weights pretrained-models/gsl-mediapipe-wbg-newmethod-cross-joint.pt --half --amp-opt-level 2  --batch-size 16 --forward-batch-size 4 --device 0

python3 main2.py --config ./config/gsl_mediapipe/test_joint.yaml --work-dir pretrain_eval/gsl_mediapipe_wbg_newmethod --weights pretrained-models/worseweights45.pt --half --amp-opt-level 2  --batch-size 16 --forward-batch-size 4 --device 0

python3 main2.py --config ./config/gsl_mediapipe/test_joint.yaml --work-dir pretrain_eval/gsl_mediapipe_wbg_newmethod --weights pretrained-models/gsl-mediapipe-cross-joint.pt --half --amp-opt-level 2  --batch-size 16 --forward-batch-size 4 --device 0

