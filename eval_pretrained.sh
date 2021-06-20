## Generate test scores

# NTU 60 XSub
python3 main.py --config ./config/nturgbd-cross-subject/test_joint.yaml --work-dir pretrain_eval/ntu60/xsub/joint-fusion --weights pretrained-models/ntu60-xsub-joint-fusion.pt --half --amp-opt-level 2

python3 main.py --config ./config/nturgbd-cross-subject/test_bone.yaml --work-dir pretrain_eval/ntu60/xsub/bone --weights pretrained-models/ntu60-xsub-bone.pt --half --amp-opt-level 2



# NTU 60 XView
python3 main.py --config ./config/nturgbd-cross-view/test_joint.yaml --work-dir pretrain_eval/ntu60/xview/joint --weights pretrained-models/ntu60-xview-joint.pt --half --amp-opt-level 2


python3 main.py --config ./config/nturgbd-cross-view/test_bone.yaml --work-dir pretrain_eval/ntu60/xview/bone --weights pretrained-models/ntu60-xview-bone.pt --half --amp-opt-level 2



# NTU 120 XSub
python3 main.py --config ./config/nturgbd120-cross-subject/test_joint.yaml --work-dir pretrain_eval/ntu120/xsub/joint --weights pretrained-models/ntu120-xsub-joint.pt --half --amp-opt-level 2

python3 main.py --config ./config/nturgbd120-cross-subject/test_bone.yaml --work-dir pretrain_eval/ntu120/xsub/bone --weights pretrained-models/ntu120-xsub-bone.pt --half --amp-opt-level 2


# NTU 120 XSet
python3 main.py --config ./config/nturgbd120-cross-setup/test_joint.yaml --work-dir pretrain_eval/ntu120/xset/joint --weights pretrained-models/ntu120-xset-joint.pt --half --amp-opt-level 2

python3 main.py --config ./config/nturgbd120-cross-setup/test_bone.yaml --work-dir pretrain_eval/ntu120/xset/bone --weights pretrained-models/ntu120-xset-bone.pt --half --amp-opt-level 2


# Kinetics Skeleton 400
python3 main.py --config ./config/kinetics-skeleton/test_joint.yaml --work-dir pretrain_eval/kinetics/joint --weights pretrained-models/kinetics-joint.pt --half --amp-opt-level 2

python3 main.py --config ./config/kinetics-skeleton/test_bone.yaml --work-dir pretrain_eval/kinetics/bone --weights pretrained-models/kinetics-bone.pt --half --amp-opt-level 2



## Perform all ensembles at once

# NTU 60 XSub
printf "\nNTU RGB+D 60 XSub\n"
python3 ensemble.py --dataset ntu/xsub --joint-dir pretrain_eval/ntu60/xsub/joint-fusion --bone-dir pretrain_eval/ntu60/xsub/bone --half --amp-opt-level 2


# NTU 60 XView
printf "\nNTU RGB+D 60 XView\n"
python3 ensemble.py --dataset ntu/xview --joint-dir pretrain_eval/ntu60/xview/joint --bone-dir pretrain_eval/ntu60/xview/bone --half --amp-opt-level 2


# NTU 120 XSub
printf "\nNTU RGB+D 120 XSub\n"
python3 ensemble.py --dataset ntu120/xsub --joint-dir pretrain_eval/ntu120/xsub/joint --bone-dir pretrain_eval/ntu120/xsub/bone --half --amp-opt-level 2


# NTU 120 XSet
printf "\nNTU RGB+D 120 XSet\n"
python3 ensemble.py --dataset ntu120/xset --joint-dir pretrain_eval/ntu120/xset/joint --bone-dir pretrain_eval/ntu120/xset/bone --half --amp-opt-level 2


# Kinetics Skeleton 400
printf "\nKinetics Skeleton 400\n"
python3 ensemble.py --dataset kinetics --joint-dir pretrain_eval/kinetics/joint --bone-dir pretrain_eval/kinetics/bone --half --amp-opt-level 2

