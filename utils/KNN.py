import numpy as np
from utils.utils import *

class KNN:
    def __init__(self, data_matrix):
        self.data_matrix = data_matrix
        self.user_top_k = []
        self.estimate = []
        self.m = self.data_matrix.mat.shape[0]
        self.n = self.data_matrix.mat.shape[1]

    def run_KNN(self, k):
        """
        Performs user-based collaborative filtering (Sarwar et al., 2001)
        Uses weighted sum over top k neighbors to generate estimate
        """
        self.estimate = self.data_matrix.train
        for user_i in range(self.m):
            sim_i = self.data_matrix.user_sim[user_i, :]  # Similarity matrix (based on masked, noisy training data)
            sorted_sim = np.flip(np.argsort(sim_i))  # Indices of users sorted (descending) by similarity to user i
            for item_j in range(self.n):
                if self.data_matrix.mask[user_i][item_j] == 0:   # If user i has not rated item j
                    sum_sim, sum_ratings, num_neighbors, current_ind = 0, 0, 0, 0
                    while num_neighbors < k:   # Continue finding top k neighbors
                        neighbor_ind = sorted_sim[current_ind]
                        if self.data_matrix.mask[neighbor_ind][item_j] != 0:
                            num_neighbors += 1
                            sum_sim += max(0, sim_i[neighbor_ind]) 
                            sum_ratings += max(0, sim_i[neighbor_ind]) * self.data_matrix.train[neighbor_ind][item_j]
                        current_ind += 1
                    self.estimate[user_i][item_j] = sum_ratings/sum_sim

    def clip(self):
        max_val = self.data_matrix.mat.max()
        min_val = self.data_matrix.mat.min()
        return self.estimate.clip(min_val, max_val)

    def get_MSE(self):
        test_mask = 1 - self.data_matrix.mask
        test_data = self.data_matrix.mat * test_mask
        return np.sum((test_mask * self.clip() - test_data)**2)/np.sum(test_mask)

    def run_KNN_and_get_results(self, ks):
        num_ks = len(ks)
        mses = np.zeros(num_ks)
        for i in range(0, num_ks):
            self.run_KNN(k = ks[i])
            mses[i] = self.get_MSE()
            print('MSE for k = %.2f: ' % ks[i] + str(mses[i]))
        return mses



