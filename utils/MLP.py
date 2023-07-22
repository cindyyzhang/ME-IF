import torch
import numpy as np
import random
import torch.optim as optim
from copy import deepcopy

class MLP(torch.nn.Module):
    def __init__(self, data_matrix):
        self.data_matrix = data_matrix
        self.estimate = []
        self.m = data_matrix.mat.shape[0]
        self.n = data_matrix.mat.shape[1]
        self.train_X = np.empty((0, self.m+self.n))
        self.train_y = []
        self.val_X = np.empty((0, self.m+self.n))
        self.val_y = []
        self.test_X = np.empty((0, self.m+self.n))
        self.test_y = []

    def split_train_val_test(self, train=0.8):
        train_X, val_X, test_X = [], [], []
        for user_i in range(self.m):
            for item_j in range(self.n):
                user = self.data_matrix.train[user_i, :]
                item = self.data_matrix.train[:,item_j]
                x = np.reshape(np.concatenate((user, item)), (1, self.m+self.n))
                num = random.random()
                if self.data_matrix.mask[user_i][item_j] == 1 and num < train:
                    train_X.append(x)
                    self.train_y.append(self.data_matrix.mat[user_i][item_j])
                elif self.data_matrix.mask[user_i][item_j] == 1 and num > train:
                    val_X.append(x)
                    self.val_y.append(self.data_matrix.mat[user_i][item_j])
                else:
                    test_X.append(x)
                    self.test_y.append(self.data_matrix.mat[user_i][item_j])
        self.train_X = np.vstack(train_X)
        self.val_X = np.vstack(val_X)
        self.test_X = np.vstack(test_X)
        self.train_y = np.array(self.train_y)
        self.val_y = np.array(self.val_y)
        self.test_y = np.array(self.test_y)
        return self.train_X, self.train_y, self.val_X, self.val_y, self.test_X, self.test_y

    def train_MLP(self, net, train_batch=128, val_step=100):
        opt = optim.Adam(net.parameters(), lr=1e-3)
        best_mse = np.inf
        best_state = None
        for step in range(2000):
            # sample batch
            i_batch = np.random.choice(len(self.train_X), size=train_batch, replace=False)

            pred = net(torch.tensor(self.train_X[i_batch]).float()).clamp(-1, 1).reshape(-1)
            loss = ((pred - torch.tensor(self.train_y[i_batch])) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()

            if (step + 1) % val_step == 0:
                with torch.no_grad():
                    pred = net(torch.tensor(self.val_X).float()).clamp(-1, 1).numpy().reshape(-1)
                    val_mse = ((pred - self.val_y) ** 2).mean()
                    print('step %s val mse %.5g' % (step + 1, val_mse))
                    # 0 and 1 for tie-break
                    best_mse, _, best_state = min((best_mse, 0, best_state), (val_mse, 1, deepcopy(net.state_dict())))
        
        with torch.no_grad():
            net.load_state_dict(best_state)
            pred = net(torch.tensor(self.test_X).float()).clamp(-1, 1).numpy().reshape(-1)

            A_hat = np.zeros((self.m, self.n))
            for i in range(self.m):
                for j in range(self.n):
                    user = self.data_matrix.train[i, :]
                    item = self.data_matrix.train[:, j]
                    x = np.reshape(np.concatenate((user, item)), (1, self.m + self.n))
                    A_hat[i][j] = net(torch.tensor(x).float()).clamp(-1, 1).detach().numpy()
            self.estimate = A_hat   # save the estimate

            test_mse = ((pred - self.test_y) ** 2).mean()
            print('test mse %.5g' % test_mse)
            return test_mse
