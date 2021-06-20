#!/bin/sh

# ex: python3 main.py --config <...> --batch-size 32 --forward-batch-size 16 --device 0
#python3 main.py --config ./config/gsl_openpose/train_joint.yaml --work-dir pretrain_eval/gsl_openpose --half --amp-opt-level 2

# Note: This code requires A LOT of memory for batch size 32,64,128. batch-size 16 for training and 8 for forward seems to barely work with --half and opt-level 2
python3 main.py --config ./config/nturgbd-cross-view/train_joint.yaml --work-dir pretrain_eval/newntu60xsub --half --amp-opt-level 2  --batch-size 16 --forward-batch-size 8 --device 0


