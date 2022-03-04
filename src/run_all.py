import numpy as np
import time
import higra as hg
from optimisations import *

#Generate logs for all subset sizes - B consistency
def run_allk_b(tree, w_init, num_leaves, node_area, node_depth, node_dimensions, F):
    
    fs, Fg, bs, H_best, G_best, max_w, time_t = optimise_measure(best_maximal, 100, tree, w_init,\
                                                                 num_leaves, node_area, node_depth, node_dimensions, F, False)
    
    print("Maximal Benefit in atmost", len(H_best), "Nodes(B-consistency)\n")
    
    print("J_max=%0.12f\n"% (max_w)) 
    
    num_nodes = int((len(H_best)//1000 + 1)*1000)
    
    x_axis = np.arange(0,num_nodes+1,1,dtype=np.int32)
    y_axis = np.zeros((num_nodes+1, ))
    
    R = best_bconsistency(len(H_best), tree, 1, num_leaves, node_area, node_depth, node_dimensions, False, True)
    
    for i in range(1, len(H_best)+1):
        print("Subset Size %d\n"% (i))
        
        start = time.time()
        
        w_init = b_init_w(i, tree, 1, R[:,:i],num_leaves, node_area, node_depth, node_dimensions, F, is_compliment=False)
        print("w_init-" + str(w_init))
        
        fs, Fg, bs, H, G, w, total_time = optimise_measure(best_bconsistency, i, tree, w_init,\
                                                          num_leaves, node_area, node_depth, node_dimensions, F, False)

        end=time.time()

        y_axis[i] = w    

        print("J: %0.12f, Nodes Used: %d, Nodes in foreground: %d, Total Time: %0.3f\n"% (w, len(H), len(G), (end-start)))
        
        if w==max_w:
            break
        
    y_axis[len(H_best):] = np.ones((num_nodes-len(H_best)+1, ))*max_w
    
    return x_axis, y_axis, G

#Generate logs for all subset sizes - C consistency
def run_allk_c(tree, w_init, num_leaves, node_area, node_depth, node_dimensions, F):
    
    fs, Fg, bs, H_best, G_best, max_w, time_t = optimise_measure(best_maximal, 100, tree, w_init,\
                                                                 num_leaves, node_area, node_depth, node_dimensions, F, False)
    
    print("Maximal Benefit in atmost", len(H_best), "Nodes(B-consistency)\n")
    
    print("J_max=%0.12f\n"% (max_w)) 
    
    num_nodes = int((len(H_best)//1000 + 1)*1000)
    
    x_axis = np.arange(0,num_nodes+1,1,dtype=np.int32)
    y_axis = np.zeros((num_nodes+1, ))
    
    i = 1
    
    R = best_cconsistency(len(H_best),tree,1, num_leaves, node_area, node_depth, node_dimensions, False, True)
    R_c = best_cconsistency(len(H_best),tree,1, num_leaves, node_area, node_depth, node_dimensions, True, True)
    
    print("Subset Size %d\n"% (i))
    w_init = c_init_w(i, tree, 1, R[:,:i+1], num_leaves, node_area, node_depth, node_dimensions, F, is_compliment=False)
    print("w_init-" + str(w_init))
    fs, Fg, bs, H_best, G_best, w,total_t = optimise_measure(best_cconsistency, i, tree, w_init,\
                                                          num_leaves, node_area, node_depth, node_dimensions, F, False)
    
    y_axis[i] = w
    print("J: %0.12f, Nodes Used: %d, Nodes in foreground: %d, Total Time: %f\n"% (w, len(H_best), len(G_best), total_t))  
    
    while(w<max_w):
        i+=1
        print("Subset Size %d\n"% (i))
        w_init = c_init_w(i, tree, 1, R[:,:i+1], num_leaves, node_area, node_depth, node_dimensions, F, is_compliment=False)
        print("w_init-" + str(w_init))
        
        fs, Fg, bs, H_best, G_best, w,total_t = optimise_measure(best_cconsistency, i, tree, w_init,\
                                                          num_leaves, node_area, node_depth, node_dimensions, F, False)

        y_axis[i] = w
        print("J: %0.12f, Nodes Used: %d, Nodes in foreground: %d, Total Time: %f\n"% (w, len(H_best), len(G_best), total_t))  

    y_axis[i:] = np.ones((num_nodes - i + 1, ))*max_w
    
    return x_axis, y_axis

#Generate logs for all subset sizes - D consistency
def run_allk_d(tree, w_init, num_leaves, node_area, node_depth, node_dimensions, F):

    fs, Fg, bs, H_best, G_best, max_w, time_t = optimise_measure(best_maximal, 100, tree, w_init,\
                                                                 num_leaves, node_area, node_depth, node_dimensions, F, False)
    
    print("Maximal Benefit in atmost", len(H_best), "Nodes(B-consistency)\n")
    
    print("J_max=%0.12f\n"% (max_w)) 
    
    num_nodes = int((len(H_best)//1000 + 1)*1000)
    
    x_axis = np.arange(0,num_nodes+1,1,dtype=np.int32)
    y_axis = np.zeros((num_nodes+1, ))
    
    i = 1
    
    belong_plus, belong_neg, R_plus, R_neg = best_dconsistency(len(H_best), tree, 1, num_leaves, node_area, node_depth, node_dimensions, is_compliment=False, first_iter=True)
    belong_plus_c, belong_neg_c, R_plus_c, R_neg_c = best_dconsistency(len(H_best), tree, 1, num_leaves, node_area, node_depth, node_dimensions, is_compliment=True, first_iter=True)

    
    print("Subset Size %d\n"% (i))
    w_init = d_init_w(i, tree, 1, belong_plus[:,:i+1], belong_neg[:,:i+1], R_plus[:,:i+1], R_neg[:,:i+1], num_leaves, node_area, node_depth, node_dimensions, F, is_compliment=False)
    print("w_init-" + str(w_init))
    H_best, w, total_t = d_optimise_measure(i,tree,w_init,\
                                                          num_leaves, node_area, node_depth, node_dimensions, F, False)

    y_axis[i] = w
    print("J: %0.12f, Nodes Used: %d, Total Time: %f\n"% (w, len(H_best), total_t))   

    i+=1
    
    while(w<max_w):
        print("Subset Size %d\n"% (i))
        w_init = d_init_w(i, tree, 1, belong_plus[:,:i+1], belong_neg[:,:i+1], R_plus[:,:i+1], R_neg[:,:i+1], num_leaves, node_area, node_depth, node_dimensions, F, is_compliment=False)
        print("w_init-" + str(w_init))
        H_best, w, total_t = d_optimise_measure(i,tree,w_init,\
                                                          num_leaves, node_area, node_depth, node_dimensions, F, False)

        y_axis[i] = w
        print("J: %0.12f, Nodes Used: %d, Total Time: %f\n"% (w, len(H_best), total_t)) 

        i+=1
    
    y_axis[i:] = np.ones((num_nodes - i + 1, ))*max_w
    
    return x_axis, y_axis