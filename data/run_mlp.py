import numpy as np
import utils.DataMatrix as ME
import utils.MLP as MLP
from utils.utils import *
import torch
import torch.nn as nn

##################################################################################################
# Hyperparameters
##################################################################################################
dataset_folder_name = 'samples'
ps = [0.1, 0.2, 0.3]   # proportions of data to vary over
mses = []

seed = 42
train_batch = 128
val_step = 100

##################################################################################################
# MAIN
##################################################################################################
class Network(nn.Module):
    def __init__(self, n_in):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(n_in, 300),
            nn.ReLU(),
            nn.Linear(300, 100),
            nn.ReLU(),
            nn.Linear(100, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.layers(x) * 2 - 1   # scale [0, 1] to [0, 2] then shift down to [-1,1]

if __name__ == "__main__":

    for i in ps:
        ratings_ME = ME.DataMatrix()
        processed_data_filename = 'data/' + dataset_folder_name + '/p_' + str(int(i*10))
        ratings_ME.load_from_file(processed_data_filename)

        mlp = MLP.MLP(ratings_ME)
        train_X, train_y, val_X, val_y, test_X, test_y = mlp.split_train_val_test()
        n_in = train_X.shape[1]
        net = Network(n_in)
        test_mse = mlp.train_MLP(net, train_batch, val_step)
        mses.append(test_mse)
    print("Test MSEs: ", mses)
