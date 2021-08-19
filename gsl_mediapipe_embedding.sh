#!/bin/sh

# python3 main2.py --config ./config/gsl_mediapipe/embed_val_joint.yaml --work-dir pretrain_eval/gsl_mediapipe_c2/joint --half --amp-opt-level 2  --batch-size 16 --forward-batch-size 8 --device 0
#gsl_mediapipe_wbg_newmethod

python3 main2.py --config ./config/gsl_mediapipe/embed_embed_joint.yaml --work-dir pretrain_eval/gsl_mediapipe_standard --half --amp-opt-level 2  --batch-size 16 --forward-batch-size 8 --device 0
