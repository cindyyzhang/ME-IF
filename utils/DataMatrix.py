import numpy as np
import math

class DataMatrix:

    def __init__(self):
        self.mat = None
        self.mask = None
        self.normalized = False
        self.max_var = None
        self.U = []
        self.VT = []
        self.s = []
        self.user_sim = []

    def run_SVD_on_training_mat(self):
        self.U, self.s, self.VT = np.linalg.svd(self.train)

    def load_from_file(self, filename):
        with open(filename + '.npy', 'rb') as f:
            self.mat = np.load(f)
            self.mask = np.load(f)
            self.train = np.load(f)
            self.U = np.load(f)
            self.VT = np.load(f)
            self.s = np.load(f)
            self.max_var = np.load(f)
            self.user_sim = np.load(f)
        self.n_rows = self.mat.shape[0]
        self.n_cols = self.mat.shape[1]

    def save_to_file(self, filename):
        with open(filename + '.npy', 'wb') as f:
            np.save(f, self.mat)
            np.save(f, self.mask)
            np.save(f, self.train)
            np.save(f, self.U)
            np.save(f, self.VT)
            np.save(f, self.s)
            np.save(f, self.max_var)
            np.save(f, self.user_sim)

    def add_matrix_and_mask(self, mat, mask, var=0.1):
        self.mat = np.copy(mat)
        self.mask = np.array(np.copy(mask), dtype=np.uint32)
        self.max_var = var
        # Add noise to observed entries
        noise = np.random.normal(0, var, size=(self.mat.shape[0], self.mat.shape[1]))
        self.train = (self.mat + noise) * self.mask
        self.n_rows = self.mat.shape[0]
        self.n_cols = self.mat.shape[1]

    def get_matrix(self):
        return self.mat

    def get_mask(self):
        return self.mask

    def get_similarity(self, i, j): 
        """
        Returns adjusted cosine similarity between user i and user j
        based on the training data
        """
        user_i = self.train[i, :]
        user_j = self.train[j, :]
        mask_i = self.mask[i, :]
        mask_j = self.mask[j, :]
        sum, squared_sum_i, squared_sum_j = 0, 0, 0
        for k in range(self.n_cols):   # Loop over all possible items
            if mask_i[k] == 1 and mask_j[k] == 1:   # Only sum over items both users i and j have rated
                item_k = self.train[:,k]
                mask_k = self.mask[:,k]
                mean_k = np.sum(item_k) / np.sum(mask_k)
                sum += (user_i[k] - mean_k) * (user_j[k] - mean_k)
                squared_sum_i += (user_i[k] - mean_k)**2
                squared_sum_j += (user_j[k] - mean_k)**2
        if (math.sqrt(squared_sum_i) * math.sqrt(squared_sum_j)) == 0:
            return 1
        return (sum/(math.sqrt(squared_sum_i) * math.sqrt(squared_sum_j)))

    def get_user_sim_matrix(self):
        """
        Fill in the user_sim matrix such that user_sim[i][j] contains the
        adjusted cosine similarity between user i and user j
        """
        self.user_sim = np.zeros((self.n_rows, self.n_rows))
        for i in range(self.n_rows):
            for j in range(self.n_rows):
                if self.user_sim[i][j] == 0 and i != j:  
                    similarity = self.get_similarity(i, j)
                    self.user_sim[i][j] = similarity
                    self.user_sim[j][i] = similarity

    def prep_for_ME_methods(self):
        self.run_SVD_on_training_mat()
        self.get_user_sim_matrix()
