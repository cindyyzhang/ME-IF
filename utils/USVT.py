import numpy as np
import math
from utils.utils import *

class USVT:
    def __init__(self,data_matrix):
        self.data_matrix = data_matrix
        self.singular_value_kept_indices = []
        self.estimate = []
        self.threshold = -np.inf
        self.m = self.data_matrix.mat.shape[0]
        self.n = self.data_matrix.mat.shape[1]

    def run_USVT(self,eta):
        """
        Accepts m x n matrix of observations, with entries in [-1,1]
        Performs USVT (Chatterjee et al., 2015)
        Outputs estimate of complete m x n matrix, SVD, and indices of singular values above the threshold
        """
        # Find set of singular values above threshold
        p = np.sum(self.data_matrix.mask)/(self.n * self.m)
        self.threshold = (2 + eta) * math.sqrt(max(self.n, self.m) * p * (1 - p) * self.data_matrix.max_var)

        self.singular_value_kept_indices = []
        self.S = np.zeros([self.m, self.n])

        for i in range(min(self.m, self.n)):
            if np.abs(self.data_matrix.s[i]) > self.threshold:
                self.singular_value_kept_indices.append(i)
                self.S[i,i] = self.data_matrix.s[i]

        # Generate estimate
        self.estimate = np.dot(self.data_matrix.U, np.dot(self.S, self.data_matrix.VT))
        self.estimate *= 1.0/p  
        self.estimate = np.clip(self.estimate, -1,1)

    def get_MSE(self):
        test_mask = 1 - self.data_matrix.mask
        test_data = self.data_matrix.mat * test_mask
        return np.sum((test_mask * self.estimate - test_data)**2)/np.sum(test_mask)

    def run_USVT_and_get_results(self, etas=[0.01]):
        num_etas = len(etas)
        mses = np.zeros(num_etas)
        num_svs = np.zeros(num_etas)   # number of singular values above threshold

        for i in range(0,num_etas):
            self.run_USVT(eta=etas[i])
            mses[i] = self.get_MSE()
            num_svs[i] = int(len(self.singular_value_kept_indices))
            print('MSE for eta = %.2f: ' % etas[i] + str(mses[i]))
            print('Number of singular values for eta = %.2f: ' % etas[i] + str(num_svs[i]))

            if num_svs[i] == 1:
                num_svs = num_svs[0:i+1]
                mses = mses[0:i+1]
                break

        return mses, num_svs

