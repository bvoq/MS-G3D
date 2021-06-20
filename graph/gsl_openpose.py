import sys
sys.path.insert(0, '')
sys.path.extend(['../'])

import numpy as np

from graph import tools

# Joint index:
# {0,  "Nose"}
# {1,  "Neck"},
# {2,  "RShoulder"},
# {3,  "RElbow"},
# {4,  "RWrist"},
# {5,  "LShoulder"},
# {6,  "LElbow"},
# {7,  "LWrist"},
# {8,  "RHip"},
# {9,  "RKnee"},
# {10, "RAnkle"},
# {11, "LHip"},
# {12, "LKnee"},
# {13, "LAnkle"},
# {14, "REye"},
# {15, "LEye"},
# {16, "REar"},
# {17, "LEar"},
body_25_openpose = [ # 0-indexed
    (0,1),(0,15),(0,16),(15,17),(16,18),(1,2),(1,5),(1,8),(2,3),(3,4),(5,6),(6,7),(8,9),(8,12),(9,10),(10,11),(11,24),(11,22),(22,23),(12,13),(13,14),(14,21),(14,19),(19,20)
]

hands_openpose = [ # 0-indexed
    (0,1),(0,5),(0,9),(0,13),(0,17),(1,2),(2,3),(3,4),(5,6),(6,7),(7,8),(9,10),(10,11),(11,12),(13,14),(14,15),(15,16),(17,18),(18,19),(19,20)
]

# modify this if you want to use the coco_pose instead
handsl = [ (t[0]+25,t[1]+25) for t in hands_openpose ] + [ (7,25) ]
handsr = [ (t[0]+25+21,t[1]+25+21) for t in hands_openpose ] + [ (4,25+21) ]

openpose_fullbody = body_25_openpose + handsl + handsr
print("a: ", body_25_openpose, " ", handsl, " ", handsr)
print('len: ', len(openpose_fullbody))

num_node = 67
self_link = [(i, i) for i in range(num_node)]
inward = openpose_fullbody
outward = [(j, i) for (i, j) in inward]
neighbor = inward + outward


class AdjMatrixGraph:
    def __init__(self, *args, **kwargs):
        self.num_nodes = num_node
        self.edges = neighbor
        self.self_loops = [(i, i) for i in range(self.num_nodes)]
        self.A_binary = tools.get_adjacency_matrix(self.edges, self.num_nodes)
        self.A_binary_with_I = tools.get_adjacency_matrix(self.edges + self.self_loops, self.num_nodes)


if __name__ == '__main__':
    graph = AdjMatrixGraph()
    A_binary = graph.A_binary
    import matplotlib.pyplot as plt
    print(A_binary)
    plt.matshow(A_binary)
    plt.show()
