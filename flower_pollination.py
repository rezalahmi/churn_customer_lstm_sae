import numpy as np
from numpy.random import rand
from functionHO import Fun
import math
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


def init_position(lb, ub, N, dim):
    X = np.zeros([N, dim], dtype='float')
    for i in range(N):
        for d in range(dim):
            X[i, d] = lb[0, d] + (ub[0, d] - lb[0, d]) * rand()

    return X


def binary_conversion(X, thres, N, dim):
    Xbin = np.zeros([N, dim], dtype='int')
    for i in range(N):
        for d in range(dim):
            if X[i, d] > thres:
                Xbin[i, d] = 1
            else:
                Xbin[i, d] = 0

    return Xbin


def boundary(x, lb, ub):
    if x < lb:
        x = lb
    if x > ub:
        x = ub

    return x


def levy_distribution(beta, dim):
    # Sigma
    nume = math.gamma(1 + beta) * np.sin(np.pi * beta / 2)
    deno = math.gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2)
    sigma = (nume / deno) ** (1 / beta)
    # Parameter u & v
    u = np.random.randn(dim) * sigma
    v = np.random.randn(dim)
    # Step
    step = u / abs(v) ** (1 / beta)
    LF = step

    return LF


class flower_pollination_optimizer:
    def __init__(self, X_train, X_test, node_layer1, node_layer2, node_layer3):
        self.N = 10  # number of salps
        self.T = 100  # maximum number of iterations
        self.maxLt = 10  # maximum iteration for local search algorithm
        self.node_layer1 = node_layer1
        self.node_layer2 = node_layer2
        self.node_layer3 = node_layer3
        self.X_train = X_train
        self.X_test = X_test
        self.fold = {'xt': self.X_train, 'xv': self.X_test}
        self.opts = {'node_layer1': self.node_layer1,
                     'node_layer2': self.node_layer2,
                     'node_layer3': self.node_layer3,
                     'fold': self.fold,
                     'N': self.N,
                     'T': self.T,
                     'maxLt': self.maxLt}

    def jfs(self):
        # Parameters
        ub = 1
        lb = 0
        thres = 0.5
        gamma = 0.01
        beta = 1.5  # levy component
        P = 0.8  # switch probability

        N = self.opts['N']
        max_iter = self.opts['T']
        if 'P' in self.opts:
            P = self.opts['P']
        if 'beta' in self.opts:
            beta = self.opts['beta']

            # Dimension
        dim = np.size(self.X_train, 1)
        if np.size(lb) == 1:
            ub = ub * np.ones([1, dim], dtype='float')
            lb = lb * np.ones([1, dim], dtype='float')

        # Initialize position
        X = init_position(lb, ub, N, dim)

        # Binary conversion
        Xbin = binary_conversion(X, thres, N, dim)

        # Fitness at first iteration
        fit = np.zeros([N, 1], dtype='float')
        Xgb = np.zeros([1, dim], dtype='float')
        fitG = float('inf')

        for i in range(N):
            fit[i, 0] = Fun(self.X_train, Xbin[i, :], self.opts)
            if fit[i, 0] < fitG:
                Xgb[0, :] = X[i, :]
                fitG = fit[i, 0]

        # Pre
        curve = np.zeros([1, max_iter], dtype='float')
        t = 0

        curve[0, t] = fitG.copy()
        print("Generation:", t + 1)
        print("Best (MGFPA):", curve[0, t])
        t += 1

        while t < max_iter:
            Xnew = np.zeros([N, dim], dtype='float')

            for i in range(N):
                # Global pollination
                if rand() < P:
                    # Levy distribution (3)
                    L = levy_distribution(beta, dim)
                    for d in range(dim):
                        # --- update
                        if rand() < 0.5:
                            # Global pollination (2)
                            Xnew[i, d] = X[i, d] + gamma * L[d] * (Xgb[0, d] - X[i, d])
                        else:
                            # --- Different flower A, B in same species
                            R = np.random.permutation(N)
                            A = R[0]
                            B = R[1]
                            # --- Epsilon [0 to 1]
                            r2 = rand()
                            # --- Pollination (6)
                            Xnew[i, d] = (max(X[A, d], X[B, d]) - min(X[A, d], X[B, d])) * r2 + min(X[A, d], X[B, d])

                            # Boundary
                        Xnew[i, d] = boundary(Xnew[i, d], lb[0, d], ub[0, d])

                # Local pollination
                else:
                    # Different flower A, B in same species
                    R = np.random.permutation(N)
                    A = R[0]
                    B = R[1]
                    # Epsilon [0 to 1]
                    r1 = rand()
                    for d in range(dim):
                        # Local pollination (4)
                        Xnew[i, d] = X[i, d] + r1 * (X[A, d] - X[B, d])
                        # Boundary
                        Xnew[i, d] = boundary(Xnew[i, d], lb[0, d], ub[0, d])

            # Binary conversion
            Xbin = binary_conversion(Xnew, thres, N, dim)

            # Greedy selection
            for i in range(N):
                Fnew = Fun(self.X_train, Xbin[i, :], self.opts)
                if Fnew < fit[i, 0]:
                    X[i, :] = Xnew[i, :]
                    fit[i, 0] = Fnew

                if fit[i, 0] < fitG:
                    Xgb[0, :] = X[i, :]
                    fitG = fit[i, 0]

            # Store result
            curve[0, t] = fitG.copy()
            print("Generation:", t + 1)
            print("Best (MGFPA):", curve[0, t])
            t += 1

            # Best feature subset
        Gbin = binary_conversion(Xgb, thres, 1, dim)
        Gbin = Gbin.reshape(dim)
        pos = np.asarray(range(0, dim))
        sel_index = pos[Gbin == 1]
        num_feat = len(sel_index)
        # Create dictionary
        mgfpa_data = {'sf': sel_index, 'c': curve, 'nf': num_feat}

        return mgfpa_data
