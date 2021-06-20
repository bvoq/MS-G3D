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


num_node = 75
self_link = [(i, i) for i in range(num_node)]
inward = mediapipe_fullbody
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
