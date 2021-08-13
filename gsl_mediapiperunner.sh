#!/bin/sh

# ex: python3 main.py --config <...> --batch-size 32 --forward-batch-size 16 --device 0
#python3 main.py --config ./config/gsl_openpose/train_joint.yaml --work-dir pretrain_eval/gsl_openpose --half --amp-opt-level 2

# Note: This code requires A LOT of memory for batch size 32,64,128. batch-size 16 for training and 8 for forward seems to barely work with --half and opt-level 2

python3 main2.py --config ./config/gsl_mediapipe/train_joint.yaml --work-dir pretrain_eval/gsl_mediapipe_wbg_newmethod --half --amp-opt-level 2  --batch-size 16 --forward-batch-size 4 --device 0
#python3 main.py --config ./config/gsl_mediapipe/train_bone.yaml --work-dir pretrain_eval/gsl_mediapipe_cross/bone --half --amp-opt-level 2  --batch-size 16 --forward-batch-size 8 --device 0


# Kinetics Skeleton 400
#python3 main.py --config ./config/kinetics-skeleton/test_joint.yaml --work-dir pretrain_eval/kinetics/joint --weights pretrained-models/kinetics-joint.pt --half --amp-opt-level 2
#
#python3 main.py --config ./config/kinetics-skeleton/test_bone.yaml --work-dir pretrain_eval/kinetics/bone --weights pretrained-models/kinetics-bone.pt --half --amp-opt-level 2




