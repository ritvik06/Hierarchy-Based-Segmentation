import numpy as np
import numba
from numba import prange

#Function to update B and R for a single node by maximising benefit sum from left and right subtrees of the node
def node_update_b(k, indexes, child0, child1, T_node, T_child0, T_child1, B_child0, B_child1):
    alpha = min(k, T_node)
    R_min, R_max = np.maximum(1, indexes-T_child0), np.minimum(T_child1, indexes-1)
    B_ret, R_ret = np.zeros((alpha-1,)), np.zeros((alpha-1,), dtype=np.int32)
    
    for i in range(2, alpha+1):
        B_node = B_child1[R_min[i-2]-1:R_max[i-2]] + (B_child0[i-R_max[i-2]-1:i-R_min[i-2]])[::-1]
        # print(B_node)
        B_ret[i-2], R_ret[i-2] = np.max(B_node), R_min[i-2]+np.argmax(B_node)

    return B_ret, R_ret

#Function to update B and R for a single node by maximising benefit sum from left and right subtrees of the node
#Stores fg and bg rather storing Benefit, reduces floating point error
# TODO - Speedup
def node_update_b_vectorized(k, w, indexes, child0, child1, T_node, T_child0, T_child1, Bf_child0, Bf_child1, Bb_child0, Bb_child1):
    alpha = min(k, T_node)
    R_min, R_max = np.maximum(1, indexes-T_child0), np.minimum(T_child1, indexes-1)
    R_ret = np.zeros((alpha-1,), dtype=np.int32)
    Bf_ret, Bb_ret = np.zeros((alpha-1,), dtype=np.int32), np.zeros((alpha-1,), dtype=np.int32)
    
    for i in range(2, alpha+1):
        B_f = (Bf_child1[R_min[i-2]-1:R_max[i-2]]) + (Bf_child0[i-R_max[i-2]-1:i-R_min[i-2]])[::-1] 
        B_b = Bb_child1[R_min[i-2]-1:R_max[i-2]] + (Bb_child0[i-R_max[i-2]-1:i-R_min[i-2]])[::-1]
        B_node = B_f - (w*B_b)
        R_ret[i-2] = R_min[i-2]+np.argmax(B_node)
        Bf_ret[i-2], Bb_ret[i-2] = B_f[R_ret[i-2]-R_min[i-2]], B_b[R_ret[i-2]-R_min[i-2]]  
        
    return Bf_ret, Bb_ret, R_ret

#Function to update B and R for a single node by maximising benefit sum from left and right subtrees of the node
def node_update_c(k, indexes, child0, child1, T_node, T_child0, T_child1, B_child0, B_child1):
    alpha = min(k, T_node)
    R_min, R_max = np.maximum(0, indexes-T_child0), np.minimum(T_child1, indexes)
    B_ret, R_ret = np.zeros((alpha,)), np.zeros((alpha,), dtype=np.int32)

    for i in range(1, alpha+1):
        B_node = B_child1[R_min[i-1]:R_max[i-1]+1] + (B_child0[i-R_max[i-1]:i-R_min[i-1]+1])[::-1]
        B_ret[i-1], R_ret[i-1] = np.max(B_node), R_min[i-1]+np.argmax(B_node)
        
    return B_ret, R_ret

#Function to update B and R for a single node by maximising benefit sum from left and right subtrees of the node
def node_update_d(k, indexes, child0, child1, T_node, T_child0, T_child1, Bplus_child0, Bneg_child0, Bplus_child1, Bneg_child1):
    alpha = min(k, T_node)
    R_min, R_max = np.maximum(0, indexes-T_child0), np.minimum(T_child1, indexes)
    Bplus_ret, Rplus_ret = np.zeros((alpha,)), np.zeros((alpha,), dtype=np.int32)
    Bneg_ret, Rneg_ret = np.zeros((alpha,)), np.zeros((alpha,), dtype=np.int32)

    for i in range(1, alpha+1):
        Bplus_node = Bplus_child1[R_min[i-1]:R_max[i-1]+1] + (Bplus_child0[i-R_max[i-1]:i-R_min[i-1]+1])[::-1]
        Bneg_node = Bneg_child1[R_min[i-1]:R_max[i-1]+1] + (Bneg_child0[i-R_max[i-1]:i-R_min[i-1]+1])[::-1]
        Bplus_ret[i-1], Rplus_ret[i-1] = np.max(Bplus_node), R_min[i-1]+np.argmax(Bplus_node)
        Bneg_ret[i-1], Rneg_ret[i-1] = np.min(Bneg_node), R_min[i-1]+np.argmin(Bneg_node)

    return Bplus_ret, Bneg_ret, Rplus_ret, Rneg_ret

#Update Benefit and Belong (+ve and -ve) for a node
def b_belong_update_d(T_node, Attr, Bplus_node, Bneg_node, belongplus_node, belongneg_node, Rplus_node, Rneg_node):
    
    for i in range(T_node,0,-1):
        if(Attr-Bplus_node[i-1] >= Bneg_node[i]):
            belongneg_node[i-1] = False
        else:
            Bneg_node[i], Rneg_node[i], belongneg_node[i-1] = Attr-Bplus_node[i-1], Rplus_node[i-1], True

        if(Attr-Bneg_node[i-1] <= Bplus_node[i]):
            belongplus_node[i-1] = False
        else:
            Bplus_node[i], Rplus_node[i], belongplus_node[i-1] = Attr-Bneg_node[i-1], Rneg_node[i-1], True
            
    return Bplus_node, Bneg_node, belongplus_node, belongneg_node, Rplus_node, Rneg_node  
