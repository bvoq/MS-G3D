import os
import argparse
import numpy as np
from numpy.lib.format import open_memmap
from tqdm import tqdm

ntu_skeleton_bone_pairs = (
    (1, 2), (2, 21), (3, 21), (4, 3), (5, 21), (6, 5),
    (7, 6), (8, 7), (9, 21), (10, 9), (11, 10), (12, 11),
    (13, 1), (14, 13), (15, 14), (16, 15), (17, 1), (18, 17),
    (19, 18), (20, 19), (22, 23), (21, 21), (23, 8), (24, 25),(25, 12)
)


# openpose code refs: https://github.com/CMU-Perceptual-Computing-Lab/openpose/blob/master/doc/02_output.md
coco_pose = [ # 0-indexed
    (0,1),(0,14),(0,15),(14,16),(15,17),(1,5),(1,2),(1,8),(1,11),(2,3),(3,4),(5,6),(6,7),(8,9),(9,10),(11,12),(12,13)
]

body_25_openpose = [ # 0-indexed
    (0,1),(0,15),(0,16),(15,17),(16,18),(1,2),(1,5),(1,8),(2,3),(3,4),(5,6),(6,7),(8,9),(8,12),(9,10),(10,11),(11,24),(11,22),(22,23),(12,13),(13,14),(14,21),(14,19),(19,20)
]

hands_openpose = [ # 0-indexed
    (0,1),(0,5),(0,9),(0,13),(0,17),(1,2),(2,3),(3,4),(5,6),(6,7),(7,8),(9,10),(10,11),(11,12),(13,14),(14,15),(15,16),(17,18),(18,19),(19,20)
]

# modify this if you want to use the coco_pose instead
handsl_op = [ (t[0]+25,t[1]+25) for t in hands_openpose ]
handsl_op.append( (7,25) )
handsr_op = [ (t[0]+25+21,t[1]+25+21) for t in hands_openpose ]
handsr_op.append( (4,25+21) )

openpose_fullbody = body_25_openpose
openpose_fullbody.extend(handsl_op)
openpose_fullbody.extend(handsr_op)


# mediapipe skeletons
body_mediapipe = [  # 0-indexed
        (0,1),(1,2),(2,3),(3,7),
        (0,4),(4,5),(5,6),(6,8),
        (0,10),(0,9), # fake connections (head body)
        (10,9),
        (10,12),(9,11), # fake connections (head body)
        (12,14),(14,16),(16,22),(16,18),(16,20),(20,18),
        (12,11),(11,13),(13,15),(15,21),(15,17),(15,19),(19,17),
        (12,24),(11,23),(24,23),
        (24,26),(26,28),(28,32),(28,30),(30,32),
        (23,25),(25,27),(27,29),(29,31)
        ]

# int[] connectionsa = {0,1,2,3,0,4,5,6, 9,11,11,13,15,15,15,17,12,14,16,16,16,18,12,24,24,26,28,28,23,23,25,27,27,29};
# int[] connectionsb = {1,2,3,7,4,5,6,8,10,12,13,15,21,17,19,19,14,16,22,18,20,20,24,23,26,28,32,30,11,25,27,29,31,31};


hands_mediapipe = [ # 0-indexed
        (0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),(5,9),(9,10),(10,11),(11,12),(9,13),(13,14),(14,15),(15,16),(0,17),(13,17),(17,18),(18,19),(19,20)
        ]


handsl_mp = [ (t[0]+33,t[1]+33) for t in hands_mediapipe]
handsl_mp.append( (15,33+0) ) # connection between body and handsl
handsl_mp.append( (19,33+5) ) # connection between body and handsl
handsl_mp.append( (17,33+17) ) # connection between body and handsl
#optional
handsl_mp.append( (21,33+4) ) # connection between body and handsl

handsr_mp = [ (t[0]+33+21,t[1]+33+21) for t in hands_mediapipe ]
handsr_mp.append( (16,33+21+0) ) # connection between body and handsl
handsr_mp.append( (20,33+21+5) ) # connection between body and handsl
handsr_mp.append( (18,33+21+17) ) # connection between body and handsl
#optional
handsr_mp.append( (22,33+21+4) ) # connection between body and handsl

mediapipe_fullbody = body_mediapipe
mediapipe_fullbody.extend(handsl_mp)
mediapipe_fullbody.extend(handsr_mp)



bone_pairs = {
    'ntu/xview': ntu_skeleton_bone_pairs,
    'ntu/xsub': ntu_skeleton_bone_pairs,

    # NTU 120 uses the same skeleton structure as NTU 60
    'ntu120/xsub': ntu_skeleton_bone_pairs,
    'ntu120/xset': ntu_skeleton_bone_pairs,

    'kinetics': ( # 0-indexed
        (0, 0), (1, 0), (2, 1), (3, 2), (4, 3), (5, 1), (6, 5), (7, 6), (8, 2), (9, 8), (10, 9),
        (11, 5), (12, 11), (13, 12), (14, 0), (15, 0), (16, 14), (17, 15)
    ),
    'gsl_openpose': openpose_fullbody,
    'gsl_mediapipe': mediapipe_fullbody

}

benchmarks = {
    'ntu': ('ntu/xview', 'ntu/xsub'),
    'ntu120': ('ntu120/xset', 'ntu120/xsub'),
    'kinetics': ('kinetics',),
    'gsl_openpose': ('gsl_openpose',),
    'gsl_mediapipe': ('gsl_mediapipe',),
}

parts = { 'train', 'val' }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Bone data generation for NTU60/NTU120/Kinetics')
    parser.add_argument('--dataset', choices=['ntu', 'ntu120', 'kinetics', 'gsl_openpose', 'gsl_mediapipe'], required=True)
    args = parser.parse_args()

    for benchmark in benchmarks[args.dataset]:
        for part in parts:
            print(benchmark, part)
            try:
                data = np.load('../data/{}/{}_data_joint.npy'.format(benchmark, part), mmap_mode='r')
                N, C, T, V, M = data.shape
                fp_sp = open_memmap(
                    '../data/{}/{}_data_bone.npy'.format(benchmark, part),
                    dtype='float32',
                    mode='w+',
                    shape=(N, 3, T, V, M))

                fp_sp[:, :C, :, :, :] = data
                for v1, v2 in tqdm(bone_pairs[benchmark]):
                    if benchmark != 'kinetics' and benchmark != 'gsl_openpose' and benchmark != 'gsl_mediapipe':
                        v1 -= 1
                        v2 -= 1
                    fp_sp[:, :, :, v1, :] = data[:, :, :, v1, :] - data[:, :, :, v2, :]
            except Exception as e:
                print(f'Run into error: {e}')
                print(f'Skipping ({benchmark} {part})')
