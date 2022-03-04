import numpy as np
import time
import higra as hg
from auxiliary_algorithms import *

#Have to check on best ways to initialise w
#Optimises weight w and accordingly calculates the best B/C consistent subset
def optimise_measure(consistency_func, k, tree, w_init, num_leaves, node_area, node_depth, node_dimensions, F, is_comp=False):
    start = time.time() 

    if(consistency_func==best_bconsistency):
        H_best,G_best = consistency_func(k, tree, w_init, num_leaves, node_area, node_depth, node_dimensions)
        
    else:
        H_best,G_best = consistency_func(k,tree, w_init, num_leaves, node_area, node_depth, node_dimensions, is_comp)

    #Calculate intersection of all nodes in G_best with GT foreground/background

    fs, bs = 0, 0

    for node in G_best:
        fn = node_dimensions[node]
        bn =  node_area[node] - fn
        fs+=fn
        bs+=bn

    #Compute Jaccard Index/New weights
    if(consistency_func==best_bconsistency or is_comp==False):
        w = float(fs/(F+bs))
        
    else:
        w = float((F-fs)/(node_area[tree.root()]-bs))
           
    print("W-",w)
    
    if (w==1 or w==w_init):
        end = time.time()
        total_t = end-start
#         print("\nTotal time : %0.4f\n" % (total_t))
        return (fs, F, bs, H_best, G_best, w,total_t)

    
    if(consistency_func==best_bconsistency):
        H_best,G_best = consistency_func(k, tree, w, num_leaves, node_area, node_depth, node_dimensions)
    else:
        H_best,G_best = consistency_func(k, tree, w, num_leaves, node_area, node_depth, node_dimensions, is_comp)

    w0 = w 

    #Calculate intersection of all nodes in G_best with GT foreground/background
    fs, bs = 0, 0

    for node in G_best:
        fn = node_dimensions[node]
        bn =  node_area[node] - fn
        fs+=fn
        bs+=bn

    #Compute Jaccard Index/New weights
    if(consistency_func==best_bconsistency or is_comp==False):
        w = float(fs/(F+bs))
        
    else:
        w = float((F-fs)/(node_area[tree.root()]-bs))
        
    print("W-",w)
    if (w==1):
        end = time.time() 
        total_t = end-start
#         print("\nTotal time : %0.4f\n" % (total_t))
        return (fs, F, bs, H_best, G_best, w,total_t)

    prev_Gbest = G_best
    prev_fs, prev_bs = fs, bs

    #Check for non-decreasing Jaccard Index
    while(w>w0):

        if(consistency_func==best_bconsistency):
            H_best,G_best = consistency_func(k, tree, w, num_leaves, node_area, node_depth, node_dimensions)

        else:
            H_best,G_best = consistency_func(k, tree, w, num_leaves, node_area, node_depth, node_dimensions, is_comp)

        w0 = w 

        fs, bs = 0, 0

        for node in G_best:
            fn = node_dimensions[node]
            bn =  node_area[node] - fn
            fs+=fn
            bs+=bn

        if(consistency_func==best_bconsistency or is_comp==False):
            w = float(fs/(F+bs))

        else:
            w = float((F-fs)/(node_area[tree.root()]-bs))
        
        if(w<w0):
            G_best = prev_Gbest
            fs, bs = prev_fs, prev_bs
            w = w0
        else:
            prev_Gbest = G_best
            prev_fs, prev_bs = fs, bs            
        print("W-",w)
        if (w==1):
            end = time.time() 
            total_t = end-start
#             print("\nTotal time : %0.4f\n" % (total_t))
            return (fs, F, bs, H_best, G_best, w,total_t)
        
    end = time.time() 

    total_t = end-start
#     print("\nTotal time : %0.4f\n" % (total_t))
    
    return (fs, F, bs, H_best, G_best, w,total_t)

#Generates output for B/C consistency with stats for multiple subset sizes
def generate_output(consistency_func,tree, w_init, num_leaves, node_area, node_depth, node_dimensions, F, start, stop, step=1, is_comp=False):
    
    for k in range(start,stop,step):
        print("Subset Size %d\n"% (k))
        if(consistency_func==best_bconsistency or is_comp==False):
            fs, F, bs, H_best, G_best, w, total_time = optimise_measure(consistency_func,k, tree, w_init, num_leaves, node_area, node_depth, node_dimensions,F)
            print("J: %0.12f, Nodes Used: %d, Nodes in foreground: %d, Total Time: %f\n"% (w, len(H_best), len(G_best), total_time))

        elif(consistency_func==best_cconsistency):
            fs, F, bs, H_best, G_best, w, time_i = optimise_measure(consistency_func,k, tree, w_init,False)
            fs_c, F_c, bs_c, H_best_c, G_best_c, w_c, time_c = optimise_measure(consistency_func,k, tree, w_init, num_leaves, node_area, node_depth, node_dimensions,F, True)
            
            total_time = (time_i+time_c)

            if(w>=w_c):
                print("J: %0.12f, Nodes Used: %d, Nodes in foreground: %d, Total Time: %f\n"% (w, len(H_best), len(G_best), total_time))
            else:
                print("Jc:%0.12f, Nodes Used: %d, Nodes in foreground: %d, Total Time: %f\n"% (w_c, len(H_best_c), len(G_best_c), total_time))
                 

#Optimises weight w and accordingly calculates the best D consistent subset
def d_optimise_measure(k, tree, w_init, num_leaves, node_area, node_depth, node_dimensions, F, is_comp=False):

    start = time.time()    

    H = best_dconsistency(k, tree, w_init, num_leaves, node_area, node_depth, node_dimensions, is_comp) 

    #Calculate intersection of all nodes in G_best with GT foreground/background
    fs, bs = 0, 0
    
    for node in H:
        fn = node_dimensions[node[0]]
        bn =  node_area[node[0]] - fn
        fs+= (fn)*(((-1)**node[1]))
        bs+= (bn)*(((-1)**node[1]))
        
    #Calculate weight using Jaccard Index/Complimentary Jaccard Index
    if(not is_comp):
        w = float(fs/(F+bs))
        
    else:
        w = float((F-fs)/(node_area[tree.root()]-bs))
        
    print("W -",w)
    
    if (w==1 or w==w_init):
        end = time.time()
        total_t = (end-start)
        return H,w,total_t
    
    prev_H = H
    w0 = w

    fs, bs = 0, 0
    
    H = best_dconsistency(k, tree, w, num_leaves, node_area, node_depth, node_dimensions, is_comp) 
    
    #Calculate intersection of all nodes in G_best with GT foreground/background
    for node in H:
        fn = node_dimensions[node[0]]
        bn =  node_area[node[0]] - fn
        fs+= (fn)*(((-1)**node[1]))
        bs+= (bn)*(((-1)**node[1]))

    #Calculate weight using Jaccard Index/Complimentary Jaccard Index 
    if(not is_comp):
        w = float(fs/(F+bs))
    else:
        w = float((F-fs)/(node_area[tree.root()]-bs))
        
    if(w0>=w):
        w = w0
        print("W -",w)
        end = time.time()
        total_t = (end-start)
        return prev_H,w,total_t
    
    print("W -",w)
    
    if (w==1):
        end = time.time()
        total_t = (end-start)
        return H,w,total_t
    
    #Check for non-decreasing jaccard index values
    while(w>w0):
        prev_H = H
        w0 = w
        H = best_dconsistency(k, tree, w, num_leaves, node_area, node_depth, node_dimensions, is_comp) 
        fs, bs = 0, 0

        for node in H:
            fn = node_dimensions[node[0]]
            bn =  node_area[node[0]] - fn
            fs+= (fn)*(((-1)**node[1]))
            bs+= (bn)*(((-1)**node[1]))

        if(not is_comp):
            w = float(fs/(F+bs))

        else:
            w = float((F-fs)/(node_area[tree.root()]-bs))
        
        if(w0>w):
            w = w0
#             print("W -",w)
            end = time.time()
            total_t = (end-start)
            return prev_H,w,total_t
        
        print("W -",w)

        if (w==1):
            end = time.time()
            total_t = (end-start)
            return H,w,total_t
    
    return H,w,total_t