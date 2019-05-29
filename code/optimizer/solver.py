import numpy as np


class Solver(object):

    def learn(self, X, y):
        N, n = X.shape
        self.num_feat = n - 1

        def get_counts(X):
            N, n = X.shape
            p = np.zeros(tuple([2] * n))
            for i in range(N):
                p[tuple(X[i, :])] += 1
            return p

        Xy = np.hstack((X, y))
        self.p = get_counts(X) / N
        self.py = get_counts(X[Xy[:, -1] == 1]) / N / (self.p + 10e-9)

    def fit(self, alpha, fairness_type=None, fairness_def=None, num_stage=2):
        pass
