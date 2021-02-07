from __future__ import division, print_function
import numpy as np
import matplotlib.pyplot as plt
import skfuzzy as fuzz

#clustering call method
# cluster_x_train = np.transpose(train_set)[:-1]
# cluster_y_train = np.transpose(train_set)[-1]
# arr_boundary = []
# for x in range(input_feature):
#     clustering(cluster_x_train[x:x+1])


def clustering(cluster_data):
    #try out clustering to get fpc value (fuzzy partition coefficient)
    max_fpc = 0
    best_cntr = []

    #finding the cluster center with best fpc
    for ncenters in range(3,10):
        cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(
            cluster_data, ncenters, 2, error=0.005, maxiter=1000, init=None)
        #print(ncenters,cntr,fpc)

        if fpc>max_fpc:
            max_fpc = fpc
            cntr_list = cntr.tolist()
            formatted_cntr = []
            for element in cntr_list:
                formatted_cntr += element
            best_cntr = formatted_cntr
            
    

    return best_cntr
    """
    #for seeing the plots
    import seaborn as sns
    sns.set_style('whitegrid')
    sns.kdeplot(cluster_x_train_set[1]) 
    plt.show()
    """

