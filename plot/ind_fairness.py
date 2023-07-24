import numpy as np
import utils.DataMatrix as ME
import utils.MLP as MLP
import utils.USVT as USVT
import utils.KNN as KNN
from utils.utils import *
from data.run_mlp import Network
import torch
import torch.nn as nn

##################################################################################################
# Hyperparameters
##################################################################################################

h = 'MLP'  # Inference algorithm h: choose either 'MLP' or 'KNN'

k = 10   # Number of nearest neighbors to use in KNN
train_batch = 128   # Batch size for training MLP
val_step = 100   # Validation step for training MLP
etas = [0.01] # Adjust eta used in USVT threshold
processed_data_filename = 'data/samples/p_1'

##################################################################################################
# MAIN
##################################################################################################

def get_IF_ratios(estimate, observed):
    ratios = []
    for i in range(estimate.shape[0]):
        for j in range(estimate.shape[0]):
            if i < j:
                a_diff = np.linalg.norm(estimate[i] - estimate[j])
                z_diff = np.linalg.norm(observed[i] - observed[j], 1)
                ratios.append(a_diff/z_diff)
    return ratios

if __name__ == "__main__":
    ratings_ME = ME.DataMatrix()
    ratings_ME.load_from_file(processed_data_filename)
    usvt = USVT.USVT(ratings_ME)
    mses, num_svs = usvt.run_USVT_and_get_results(etas)
    usvt_estimate = usvt.estimate

    if h == 'MLP':
        mlp = MLP.MLP(ratings_ME)
        train_X, train_y, val_X, val_y, test_X, test_y = mlp.split_train_val_test()
        n_in = train_X.shape[1]
        net = Network(n_in)
        test_mse = mlp.train_MLP(net, train_batch, val_step)
        A_hat_h = mlp.estimate
        ratios_h = get_IF_ratios(A_hat_h, ratings_ME.train)   # Get IF ratios for h

        mlp.data_matrix.train = usvt_estimate
        test_mse = mlp.train_MLP(net, train_batch, val_step)
        A_hat_f = mlp.estimate
        ratios_f = get_IF_ratios(A_hat_f, ratings_ME.train)


    elif h == 'KNN':
        knn = KNN.KNN(ratings_ME)
        mses = knn.run_KNN_and_get_results([k])
        A_hat_h = knn.estimate
        ratios_h = get_IF_ratios(A_hat_h, ratings_ME.train)  

        knn.data_matrix.train = usvt_estimate
        mses = knn.run_KNN_and_get_results([k])
        A_hat_f = knn.estimate
        ratios_f = get_IF_ratios(A_hat_f, ratings_ME.train) 

    else:
        print("Invalid inference algorithm h. Please choose either 'MLP' or 'KNN'.")

    # Plot histogram.
    x_max = max(ratios_h)
    plt.xlim(0, x_max)
    bin_list = [x_max/50.0 * i for i in range(51)]
    n, bins, patches = plt.hist(ratios_h, bins=bin_list, facecolor='g', label="without SVT pre-processing")
    bin_centers = 0.5 * (bins[:-1] + bins[1:])

    for c, p in zip(bin_centers, patches):
        plt.setp(p, color='#214da6', alpha=0.7)

    n, bins, patches = plt.hist(ratios_f, bins=bin_list, facecolor='g', label="with SVT pre-processing")
    bin_centers = 0.5 * (bins[:-1] + bins[1:])

    for c, p in zip(bin_centers, patches):
        plt.setp(p, color='#a32632', alpha=0.7)

    plt.xlabel(r"$D(f(a), f(b))/ d(a, b)$", fontsize=20)
    plt.ylabel("Frequency", fontsize=20)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    ax = plt.gca()
    handles, labels = ax.get_legend_handles_labels()
    plt.legend(reversed(handles), reversed(labels), loc="upper right", fontsize=18)
    plt.show()