import numpy as np

from .base import BaseLAM

class SSA(BaseLAM):
    '''Singular spectrum analysis
    '''
    def __init__(self, l):
        super(SSA, self).__init__()
        self.l = l
        self.name = 'ssa_{}'.format(self.l)

    def k(self, row):
        return row.shape[0] - self.l + 1

    def get_phase_trajectory(self, row):
        return np.column_stack([row[i:i+self.l] for i in range(0, self.k(row))])

    def get_elementary(self, phase_trajectory, return_singular_values=True):
        d = np.linalg.matrix_rank(phase_trajectory)
        U, S, V = np.linalg.svd(phase_trajectory)
        V = V.T
        
        elem = np.array([S[i] * np.outer(U[:,i], V[:,i]) for i in range(0, d)])
        if return_singular_values:
            return elem, S
        else:
            return elem

    def hankelize(self, x):
        l, k = x.shape
        transpose = False
        if l > k:
            x = x.T
            l, k = k, l
            transpose = True

        hankel_matrix = np.zeros((l, k))
        for i in range(l):
            for j in range(k):
                s = i + j
                if 0 <= s <= k - 1:
                    for ll in range(0, s + 1):
                        hankel_matrix[i, j] += 1/(s + 1) * x[ll, s - ll]
                elif l <= s <= k - 1:
                    for ll in range(0, l - 1):
                        hankel_matrix[i, j] += 1/(l - 1) * x[ll, s - ll]
                elif k <= s <= k + l - 2:
                    for ll in range(s - k + 1, L):
                        hankel_matrix[i, j] += 1/(k + l - s - 1) * X[ll, s - ll]
        if transpose:
            return hankel_matrix.T
        else:
            return hankel_matrix

    def to_time_series(self, x):
        x_reversed = x[::-1]
        return np.array([x_reversed.diagonal(i).mean() for i in range(-x.shape[0] + 1, x.shape[1])])
        
    def fit_row(self, row):
        phase_trajectory = self.get_phase_trajectory(row)
        elem, params = self.get_elementary(phase_trajectory, return_singular_values=True)
        pred = self.to_time_series(elem.sum(axis=0))
        return (pred, params)