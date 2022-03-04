import numpy as np
from numba import jit,njit
import cv2
import time
import higra as hg
import sys, os
import scipy
from numba_functions import *

try:
    from utils import * # imshow, locate_resource, get_sed_model_file
except: # we are probably running from the cloud, try to fetch utils functions from URL
    import urllib.request as request; exec(request.urlopen('https://github.com/higra/Higra-Notebooks/raw/master/utils.py').read(), globals())

# Returns atribute A(n) for a given node n
def attribute(node, tree, w, node_dimensions, node_area, is_compliment):
#     print(node)
    fn = node_dimensions[node]
    bn =  node_area[node] - fn
    
    if (not is_compliment):
        return (fn-(w*bn))
    else:
        if(w==0):
            w = sys.maxsize
        return bn - float(1/w)*fn

#Returns the measure jaccard index from calculated segmentation dimensions (fs,bs)
def jaccard_index(fs,bs,F):
    return float(fs/(F+bs))

############ COMPILE ALL HELPER NUMBA FUNCTIONS FOR AUXILIARY ALGORITHMS ############

#All Numba functions compiled here in nopython mode
start = time.time()
numba_func = jit(nopython=True)(node_update_b)
numba_temp = jit(nopython=True)(node_update_b_vectorized)
numba_cfunc = jit(nopython=True)(node_update_c)
numba_dfunc = jit(nopython=True)(node_update_d)
numba_dfunc2 = jit(nopython=True)(b_belong_update_d)
end = time.time()

print("\nCompilation for numba functions took %0.3f"% (end-start),"sec")

################ AUXILIARY ALGORITHMS ################

#Auxiliary Algorithm for B-Consistency
def best_bconsistency(k, tree, w, num_leaves, node_area, node_depth, node_dimensions, is_compliment=False, is_first=False):
    #2d array B (row:node----->columns:list of  benefits for each subset size)
    #2d array R (row:node----->columns:list of number of nodes in right subtree for each subtree size)
    #1d array T (for index:node----->max value of subset size to be considered for subtree rooted at node)

    #Initialise B and R, column (i-1) signifies subset size i
    B, R = np.zeros((tree.root()+1, min(k, int(num_leaves[tree.root()])))), np.zeros((tree.root()+1, min(k,int(num_leaves[tree.root()]))), dtype=np.int32)

    #Initialise T, index i signifies max subset size considered for node i
    T = np.zeros((tree.root()+1,), dtype=np.int32)  

    # Bottom up traversal of tree to construct B, R   
    for node in tree.leaves_to_root_iterator(include_leaves = True, include_root = True):   
        #Only nodes less than depth K make the best cut
        if (node_depth[node]<=k):
            fn = node_dimensions[node]
            B[node][0], R[node][0] = max(fn-(w*(node_area[node] - fn)), 0), 0
            T[node] = min(k, num_leaves[node])
            try:
                if(not tree.is_leaf(node) and node_depth[node]!=k):
                    #Update B,R for node by maximising benefit sum from left & right subtrees
                    child0, child1 = tree.child(0,node), tree.child(1,node)
                    B[node][1:T[node]], R[node][1:T[node]] = numba_func(k, np.arange(2, T[node]+1), child0, child1, T[node], T[child0], T[child1], B[child0], B[child1])
            except:
                "Investigate Issue with subset size 1"
            
    #If it's the first iteration while testing for multiple values of K, return R which is same for first iteration for all values of K if same w_init is used
    if (is_first):
        return R
    
    #H is the Best Subset of nodes, G is the subset of nodes with +ve attribute(foreground), Q is queue for top-down traversal of the tree
    H, G, Q = [], [], []
    
    
    Q.append((tree.root(), T[tree.root()]))
    
    node, i = Q.pop(-1)
    
    #If Right subtree of node has 0 nodes to be consider, put node into the best subset
    if (R[node][i-1]==0):
        #Add node to G if it has positive attribute
        if (attribute(node,tree,w,node_dimensions, node_area,False)>0):
            G.append(node)
        H.append(node)
    
    else:
        #If node is not considered, add both children of node to queue
        Q.insert(0, (tree.child(1, node), R[node][i-1]))
        Q.insert(0, (tree.child(0, node), (i-R[node][i-1])))

    #Continue till Queue is empty 
    while (len(Q)!=0):
        node, i = Q.pop(-1)
        
        #If Right subtree of node has 0 nodes to be consider, put node into the best subset
        if (R[node][i-1]==0):
            #Add node to G if it has positive attribute
            if (attribute(node,tree,w,node_dimensions, node_area, False)>0):
                G.append(node)
            H.append(node)

        else:
            #If node is not considered, add both children of node to queue
            Q.insert(0, (tree.child(1, node), R[node][i-1]))
            Q.insert(0, (tree.child(0, node), (i-R[node][i-1])))
        
    return (H, G)


# Returns H and G for first iteration while optimising measure for B-consistency
def first_iter_b(k, tree, w, R, num_leaves, node_area, node_depth, node_dimensions, is_compliment=False):
    
    T_root = min(k, num_leaves[tree.root()])
  
    H, G, Q = [], [], []
    
    Q.append((tree.root(), T_root))
    
    node, i = Q.pop(-1)
    
    if (R[node][i-1]==0):
        if (attribute(node,tree,w,node_dimensions, node_area, False)>0):
            G.append(node)
        H.append(node)
    
    else:
        Q.insert(0, (tree.child(1, node), R[node][i-1]))
        Q.insert(0, (tree.child(0, node), (i-R[node][i-1])))

    
    while (len(Q)!=0):
        node, i = Q.pop(-1)

        if (R[node][i-1]==0):
            if (attribute(node,tree,w,node_dimensions, node_area, False)>0):
                G.append(node)
            H.append(node)

        else:
            Q.insert(0, (tree.child(1, node), R[node][i-1]))
            Q.insert(0, (tree.child(0, node), (i-R[node][i-1])))
   
    return (H, G)

#Calculates updated weight for first iteration of optimisation of B-consistency
def b_init_w(k, tree, w, R, num_leaves, node_area, node_depth, node_dimensions, F, is_compliment=False):
    
    H_best, G_best = first_iter_b(k, tree, w, R, num_leaves, node_area, node_depth, node_dimensions, is_compliment=False)
    
    fs, bs = 0, 0

    for node in G_best:
        fn = node_dimensions[node]
        bn =  node_area[node] - fn
        fs+=fn
        bs+=bn

    return float(fs/(F+bs))

# Fully vectorized implementation of the auxiliary algorithm
def best_bconsistency_temp(k, tree, w, is_compliment):
    #Benefit 2d array B (row:node----->columns:list of  benefits for each subset size)

    #Right dictionary R (row:node----->columns:list of number of nodes in right subtree for each subtree size)

    #Initialise B and R, column (i-1) signifies subset size i
    
    start = time.time() 
    
    total_leaves = num_leaves[tree.root()]
    
    R = np.zeros((tree.root()+1, min(k,total_leaves)), dtype=np.int32)
    B_f = np.zeros((tree.root()+1, min(k,total_leaves)), dtype=np.int32)
    B_b = np.zeros((tree.root()+1, min(k,total_leaves)), dtype=np.int32)

    T = np.zeros((tree.root()+1,), dtype=np.int32)  

    # Bottom up traversal of tree to construct B, R 2d arrays    
    for node in tree.leaves_to_root_iterator(include_leaves = True, include_root = True):
        fn = node_dimensions[node]
        R[node][0] = 0
        if((fn-(w*(node_area[node] - fn)))>0):
            B_f[node][0], B_b[node][0] = fn, node_area[node]-fn
        
        T[node] = min(k, total_leaves)

        if(not tree.is_leaf(node)):
            child0, child1 = tree.child(0,node), tree.child(1,node)
            B_f[node][1:T[node]], B_b[node][1:T[node]], R[node][1:T[node]] = numba_temp(k, w, np.arange(2, T[node]+1),
                                                                                        child0, child1, T[node],
                                                                                        T[child0],T[child1],
                                                                                        B_f[child0],B_f[child1],
                                                                                        B_b[child0],B_b[child1])

    H, G, Q = [], [], []
    
    Q.append((tree.root(), T[tree.root()]))
    
    node, i = Q.pop(-1)
    
    if (R[node][i-1]==0):
        if (attribute(node,tree,w,is_compliment)>0):
            G.append(node)
        H.append(node)
    
    else:
        Q.insert(0, (tree.child(1, node), R[node][i-1]))
        Q.insert(0, (tree.child(0, node), (i-R[node][i-1])))

    
    while (len(Q)!=0):
        node, i = Q.pop(-1)

        if (R[node][i-1]==0):
            if (attribute(node,tree,w,is_compliment)>0):
                G.append(node)
            H.append(node)

        else:
            Q.insert(0, (tree.child(1, node), R[node][i-1]))
            Q.insert(0, (tree.child(0, node), (i-R[node][i-1])))
            
    end = time.time()
    
#     print("Time for iteration : %0.4f" % (end-start))

        
    return (H, G)

#Auxiliary Algorithm for C-consistency
def best_cconsistency(k, tree, w, num_leaves, node_area, node_depth, node_dimensions, is_compliment=False, is_first=False):
    #2d array B (row:node----->columns:list of  benefits for each subset size)
    #2d array R (row:node----->columns:list of number of nodes in right subtree for each subtree size)
    #Max Subset size 1, in R[node][1] store 1 for best node in right subtree, 0 for node in left subtree and -1 for the node itself
    #1d array T (for index:node----->max value of subset size to be considered for subtree rooted at node)
    
    B, R = np.zeros((tree.root()+1, min(k,int(num_leaves[tree.root()]))+1)), np.zeros((tree.root()+1, min(k,int(num_leaves[tree.root()]))+1), dtype=np.int32)

    T = np.zeros((tree.root()+1,), dtype=np.int32) 
    
    #Initialise B and R, in value (i)th index signifies subset size i
    for node in tree.leaves_to_root_iterator(include_leaves = True, include_root = True):
        fn = node_dimensions[node]

        #If calculating Jaccard Index
        if (not is_compliment):
            attr = fn-(w*(node_area[node] - fn))
        #If calculating complimentary Jaccard Index
        else:
            if(w!=0):
                attr = (node_area[node] - fn) - float(1/w)*fn
            else:
                attr = (node_area[node] - fn) - float(sys.maxsize)*fn
            
        T[node] = min(k, int(num_leaves[node]))

        #Update B,R for non-leaf nodes in tree in bottom - up traversal of tree
        if(not tree.is_leaf(node)):
            child0, child1 = tree.child(0,node), tree.child(1,node)
            B[node][1:T[node]+1], R[node][1:T[node]+1] = numba_cfunc(k, np.arange(1, T[node]+1),
                                                                          child0, child1, T[node],
                                                                          T[child0],T[child1],
                                                                          B[child0],B[child1])      
            

        #If leaf node has positive attribute, consider node
        if(tree.is_leaf(node) or attr>=B[node][1]):
            B[node][1], R[node][1] = attr, -1
            
    if (is_first):
        return R
        
    #H represents the best-subset by pruning the tree
    #G represents the best-subset with positive attribute nodes which represent foreground
    #Q behaves as a Queue 
    
    H, G, Q = [], [], []
    
    Q.append((tree.root(), T[tree.root()]))
    
    node, i = Q.pop(-1)
    
    #If node is chosen for best subset H
    if (R[node][i]==-1):
        #If node is of +ve attribute, add to G
        if(attribute(node,tree,w,node_dimensions, node_area,is_compliment)>=0):
            G.append(node)
        H.append(node)
    
    else:
        if(not tree.is_leaf(node)):
            #Nodes with more than 1 best nodes in Right subtree
            if(R[node][i] > 0):
                Q.insert(0, (tree.child(1, node), R[node][i]))
            #If all i best nodes are not from right subtree, add left child node to queue
            if(R[node][i]<i):
                Q.insert(0, (tree.child(0, node), (i-R[node][i])))
    
    #Continue till Queue is empty
    while (len(Q)!=0):
        node, i = Q.pop(-1)

        #If node is chosen for best subset H
        if (R[node][i]==-1):
            #If node is of +ve attribute, add to G
            if(attribute(node,tree,w,node_dimensions, node_area,is_compliment)>=0):
                G.append(node)
            H.append(node)

        else:
            if(not tree.is_leaf(node)):
                #Nodes with more than 1 best nodes in Right subtree
                if(R[node][i] > 0):
                    Q.insert(0, (tree.child(1, node), R[node][i]))
                #If all i best nodes are not from right subtree, add left child node to queue
                if(R[node][i] < i):
                    Q.insert(0, (tree.child(0, node), (i-R[node][i])))
    
    # If there are less than K nodes with positive attribute, assign some negative attribute nodes to foreground as well
    if(len(G)<k):
        print("WARNING - Chosen Negative Attribute Nodes")
        G = H
        
    return H,G


#Top-down traversal of tree to select best nodes for first iteration of optimising measure for C-consistency
def first_iter_c(k, tree, w, R, num_leaves, node_area, node_depth, node_dimensions, is_compliment=False):
    
    T_root = min(k, num_leaves[tree.root()])
    
    H, G, Q = [], [], []
    
    Q.append((tree.root(), T_root))
    
    node, i = Q.pop(-1)
    
    if (R[node][i]==-1):
        if(attribute(node,tree,w,node_dimensions, node_area, is_compliment)>=0):
            G.append(node)
        H.append(node)
    
    else:
        if(not tree.is_leaf(node)):
            if(R[node][i] > 0):
                Q.insert(0, (tree.child(1, node), R[node][i]))
            if(R[node][i]<i):
                Q.insert(0, (tree.child(0, node), (i-R[node][i])))
    
    while (len(Q)!=0):
        node, i = Q.pop(-1)

        if (R[node][i]==-1):
            if(attribute(node,tree,w,node_dimensions, node_area, is_compliment)>=0):
                G.append(node)
            H.append(node)

        else:
            if(not tree.is_leaf(node)):
                if(R[node][i] > 0):
                    Q.insert(0, (tree.child(1, node), R[node][i]))
                if(R[node][i] < i):
                    Q.insert(0, (tree.child(0, node), (i-R[node][i])))
    
    if(len(G)<k):
        print("WARNING - Chosen Negative Attribute Nodes")
        G = H
        
    return H,G

#Calculate new weight for 2nd iteration of optimisation measure(if needed)
def c_init_w(k, tree, w, R, num_leaves, node_area, node_depth, node_dimensions, F, is_compliment=False):
    
    H_best, G_best = first_iter_c(k, tree, w, R, num_leaves, node_area, node_depth, node_dimensions, is_compliment)
    
    fs, bs = 0, 0

    for node in G_best:
        fn = node_dimensions[node]
        bn =  node_area[node] - fn
        fs+=fn
        bs+=bn
        
    if(not is_compliment):
        return float(fs/(F+bs))
        
    else:
        return float((F-fs)/(node_area[tree.root()]-bs))    

#Auxiliary algorithm for D-consistency
def best_dconsistency(k, tree, w, num_leaves, node_area, node_depth, node_dimensions, is_compliment=False, first_iter=False):
    #2d array B_plus, B_neg (row:node----->columns:list of benefits for each subset size)
    #2d array R_plus, R_neg (row:node----->columns:list of number of nodes in right subtree for each subtree size)
    #1d array T (for index:node----->max value of subset size to be considered for subtree rooted at node)
    
    B_plus, B_neg = np.zeros((tree.root()+1, min(k,int(num_leaves[tree.root()]))+1)), np.zeros((tree.root()+1, min(k,int(num_leaves[tree.root()]))+1))
    R_plus, R_neg = np.zeros((tree.root()+1, min(k,int(num_leaves[tree.root()]))+1), dtype=np.int32), np.zeros((tree.root()+1, min(k,int(num_leaves[tree.root()]))+1), dtype=np.int32)
    belong_plus, belong_neg = np.zeros((tree.root()+1, min(k,int(num_leaves[tree.root()])))), np.zeros((tree.root()+1, min(k,int(num_leaves[tree.root()]))), dtype=np.int32)
    
    T = np.zeros((tree.root()+1,), dtype=np.int32) 
       
    #Bottom up Traversal of tree
    for node in tree.leaves_to_root_iterator(include_leaves = True, include_root = True): 
        fn = node_dimensions[node]
        #Calculating attribute for jaccard index
        if (not is_compliment):
            attr = fn-(w*(node_area[node] - fn))
        #Calculating attribute for complimentary jaccard index
        else:
            if(w!=0):
                attr = (node_area[node] - fn) - float(1/w)*fn
            else:
                attr = (node_area[node] - fn) - float(sys.maxsize)*fn
            
        T[node] = min(k, int(num_leaves[node]))
        
        if(tree.is_leaf(node)):
            B_plus[node][1], R_plus[node][1], belong_plus[node][0], B_neg[node][1], R_neg[node][1], belong_neg[node][0] = (attr, 0, True,
                                                                                                                        attr, 0, True)

        else:
            child0, child1 = tree.child(0,node), tree.child(1,node)
            B_plus[node][1:T[node]+1],B_neg[node][1:T[node]+1], R_plus[node][1:T[node]+1], R_neg[node][1:T[node]+1] = (
                                                                        numba_dfunc(k, np.arange(1, T[node]+1),
                                                                          child0, child1, T[node],
                                                                          T[child0],T[child1],
                                                                          B_plus[child0],B_neg[child0],
                                                                          B_plus[child1],B_neg[child1]))   
            
            if(node!=tree.root()):
                B_plus[node], B_neg[node], belong_plus[node], belong_neg[node], R_plus[node], R_neg[node] = (
                                                                                numba_dfunc2(T[node], attr,
                                                                                  B_plus[node], B_neg[node],
                                                                                  belong_plus[node], belong_neg[node],
                                                                                  R_plus[node], R_neg[node]))

            else:
                if(attr <= B_plus[tree.root()][1]):
                    belong_plus[tree.root()][0] = False
                else:
                    B_plus[tree.root()][1], R_plus[tree.root()][1], belong_plus[tree.root()][0] = attr, 0, True
 
    #If running the first iteration of D-consistency while optimising the measure
    if(first_iter):
        return belong_plus, belong_neg, R_plus, R_neg
                
        
    H, Q = [], []
    
    Q.append((tree.root(), T[tree.root()], 0))
    
    node, i, i_node = Q.pop(-1)
    
    if(i_node%2==0):
        belong, r = belong_plus[node][i-1], R_plus[node][i]
    else:
        belong, r = belong_neg[node][i-1], R_neg[node][i]
        
    if(belong):
        H.append((node, i_node))
        i_node+=1
    
    if (not tree.is_leaf(node)):
        if(r>0):
            Q.append((tree.child(1, node), r, i_node))

        if(r + belong < i):
            Q.append((tree.child(0, node),i-int(belong)-r, i_node))

    #Continue until queue is empty     
    while(len(Q)!=0):
        node, i, i_node = Q.pop(-1)

        if(i_node%2==0):
            belong, r = belong_plus[node][i-1], R_plus[node][i]
        else:
            belong, r = belong_neg[node][i-1], R_neg[node][i]

        if(belong):
            H.append((node, i_node))
            i_node+=1

        if(not tree.is_leaf(node)):
            if(r>0):
                Q.append((tree.child(1, node), r, i_node))

            if(r + belong < i):
                Q.append((tree.child(0, node),i-int(belong)-r, i_node))
    
    return H

#Top-down traversal of tree to select best nodes for first iteration of optimising measure for D-consistency
def first_iter_d(k, tree, w, belong_plus, belong_neg, R_plus, R_neg, num_leaves, node_area, node_depth, node_dimensions, is_compliment=False):
    
    T_root = min(k, num_leaves[tree.root()])
        
    H, Q = [], []
    
    Q.append((tree.root(), T_root, 0))
    
    node, i, i_node = Q.pop(-1)
    
    if(i_node%2==0):
        belong, r = belong_plus[node][i-1], R_plus[node][i]
    else:
        belong, r = belong_neg[node][i-1], R_neg[node][i]
        
    if(belong):
        H.append((node, i_node))
        i_node+=1
    
    if (not tree.is_leaf(node)):
        if(r>0):
            Q.append((tree.child(1, node), r, i_node))

        if(r + belong < i):
            Q.append((tree.child(0, node),i-int(belong)-r, i_node))
        
    while(len(Q)!=0):
        node, i, i_node = Q.pop(-1)

        if(i_node%2==0):
            belong, r = belong_plus[node][i-1], R_plus[node][i]
        else:
            belong, r = belong_neg[node][i-1], R_neg[node][i]

        if(belong):
            H.append((node, i_node))
            i_node+=1

        if(not tree.is_leaf(node)):
            if(r>0):
                Q.append((tree.child(1, node), r, i_node))

            if(r + belong < i):
                Q.append((tree.child(0, node),i-int(belong)-r, i_node))
    
    return H

#Returns weight for 2nd iteration while optimising the measure(if needed)
def d_init_w(k, tree, w, belong_plus, belong_neg, R_plus, R_neg, num_leaves, node_area, node_depth, node_dimensions, F, is_compliment=False):
    
    H_best = first_iter_d(k, tree, w, belong_plus, belong_neg, R_plus, R_neg, num_leaves, node_area, node_depth, node_dimensions, is_compliment) 

    fs, bs = 0, 0
    
    for node in H_best:
        fn = node_dimensions[node[0]]
        bn =  node_area[node[0]] - fn
        fs+= (fn)*(((-1)**node[1]))
        bs+= (bn)*(((-1)**node[1]))
        
    if(not is_compliment):
        return float(fs/(F+bs))
        
    else:
        return float((F-fs)/(node_area[tree.root()]-bs))


#Auxiliary algorithm for finding the best B-consistency segmentation specified by unlimited subset
def best_maximal(k, tree, w, num_leaves, node_area, node_depth, node_dimensions,  is_comp=False):
    #P stores actual maximam benefit for a node instead of the whole vector
    P = np.zeros((tree.root()+1,))
    
    for node in tree.leaves_to_root_iterator(include_leaves = True, include_root = True):
         
        if(tree.is_leaf(node)):
            P[node] = max(attribute(node, tree, w,node_dimensions, node_area, is_comp), 0)     
        else:
            P[node] = P[tree.child(0, node)] + P[tree.child(1, node)]
            
    H, G, Q = [], [], []
    
    Q.append(tree.root())
    
    node = Q.pop(-1)
    
    if (P[node]<=max(attribute(node, tree, w,node_dimensions, node_area, is_comp), 0)):
        if (attribute(node,tree,w,node_dimensions, node_area, is_comp)>0):
            G.append(node)
        H.append(node)
    
    else:
        Q.insert(0, tree.child(1, node))
        Q.insert(0, tree.child(0, node))


    while(len(Q)!=0):
        node = Q.pop(-1)

        if (P[node]<=max(attribute(node, tree, w,node_dimensions, node_area, is_comp), 0)):
            if (attribute(node,tree,w,node_dimensions, node_area,is_comp)>0):
                G.append(node)
            H.append(node)

        else:
            Q.insert(0, tree.child(1, node))
            Q.insert(0, tree.child(0, node))
            
    return H, G

