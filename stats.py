import sys
sys.path.append('../')

import numpy as np
from math import sqrt
from sklearn.metrics import r2_score
from sklearn.decomposition import PCA
from statsmodels.multivariate.cancorr import CanCorr
import matplotlib.pyplot as plt


def pearson_corr(a, b):
    mu1 = a.mean()
    mu2 = b.mean()
    return ((a - mu1) * (b - mu2)).mean() / sqrt(((a - mu1)**2).mean() * ((b - mu2)**2).mean())


def pearson_corr_list(vecs1, vecs2):
    """
    returns the mean Pearson correlation between 2 lists of matching vectors, with corrected signs
    """
    vecs1_norm = np.vstack([v / np.linalg.norm(v) for v in vecs1])
    signs = [np.sign(np.mean(v1 * v2)) for v1, v2, in zip(vecs1, vecs2)]
    vecs2_norm = np.vstack([s * v / np.linalg.norm(v) for s, v in zip(signs, vecs2)])
    return pearson_corr(vecs1_norm, vecs2_norm)


def var_exp_ratio(y, yest):
    sse = np.sum((y-yest)**2)
    tot = np.sum(y**2)
    return 1-(sse/tot)


def flatten_trajectory(X):
    """
    Assumes a shape trials x timesteps x neurons.
    Returns a shape (trials x timesteps) x neurons
    """
    if len(X.shape) == 3:
        n_neurons = X.shape[-1]
        X = X.transpose((2, 0, 1)).reshape((n_neurons, -1)).T
    return X


def pca_fit(X, n_components=20):
    pca = PCA(n_components=n_components)
    pca.fit(X)
    return pca


def pca_cumvar(X, n_components=20, label=None, cross_validate=False):
    X = flatten_trajectory(X)
    print(X.shape)
    if cross_validate:
        expvars, optd, optv, V = cvPCA(X, n_components=n_components)
        plt.plot(np.arange(n_components + 1), np.concatenate([[0], expvars]), marker='o', label=label)
        plt.ylim(0, 1.05)
        return {'expvar': expvars, 'optd': optd, 'optv': optv, 'V': V}
    else:
        pca = pca_fit(X, n_components)
        plt.plot(np.arange(n_components + 1), np.concatenate([[0], np.cumsum(pca.explained_variance_ratio_)]), marker='o',
                 label=label)
        plt.ylim(0, 1)
        return pca


def align(X1, X2, m, pca1=None, pca2=None):
    if pca1 is None:
        pca1 = PCA(n_components=m).fit(X1)
    if pca2 is None:
        pca2 = PCA(n_components=m).fit(X2)

    # Define latent spaces
    proj1 = pca1.components_[:m]
    L1 = X1 @ proj1.T
    proj2 = pca2.components_[:m]
    L2 = X2 @ proj2.T

    # Find aligned projections
    cc = CanCorr(L1, L2)
    al1 = cc.y_cancoef
    al2 = cc.x_cancoef
    Ltilde1 = L1 @ al1
    Ltilde2 = L2 @ al2

    return Ltilde1, Ltilde2, proj1.T @ al1, proj2.T @ al2, cc.cancorr


#%% Cross-validated PCA
"""
Adapted from Joao Barbosa, https://colab.research.google.com/drive/1BPdZZvw-h1mAZkPK6_5LjBBfgezqSGaY?usp=sharing#scrollTo=klugDpK70WoJ
"""

def cvpca_do_split_(D, frac=0.9, shuffle=True):
    n_sep = int(frac * D.shape[0])
    m_sep = int(frac * D.shape[1])
    if shuffle:
        idx_row_train = np.random.choice(np.arange(D.shape[0]), n_sep, replace=False)
        idx_col_train = np.random.choice(np.arange(D.shape[1]), m_sep, replace=False)
        idx_row_test = np.setdiff1d(np.arange(D.shape[0]), idx_row_train)
        idx_col_test = np.setdiff1d(np.arange(D.shape[1]), idx_col_train)
    else:
        idx_row_train = np.arange(n_sep)
        idx_col_train = np.arange(m_sep)
        idx_row_test = np.arange(n_sep, D.shape[0])
        idx_col_test = np.arange(m_sep, D.shape[1])
    X = D[:, idx_col_train]
    Y = D[:, idx_col_test]
    X_train, X_test = X[idx_row_train, :], X[idx_row_test, :]
    Y_train, Y_test = Y[idx_row_train, :], Y[idx_row_test, :]
    return X_train, Y_train, X_test, Y_test, idx_row_train, idx_col_train, idx_row_test, idx_col_test


def cvPCA(D, frac=0.9, n_components=None, shuffle=True):
    X_train, Y_train, X_test, Y_test, idx_row_train, idx_col_train, idx_row_test, idx_col_test = \
        cvpca_do_split_(D, frac, shuffle)
    n_sep, m_sep = X_train.shape
    _, s, V_train = np.linalg.svd(np.concatenate([X_train, Y_train], axis=1), full_matrices=False)

    if n_components is None:
        n_components = D.shape[1]

    var_exps = []
    for dim in range(1, n_components + 1):
        V_train_x = V_train[:dim, :m_sep]
        V_train_y = V_train[:dim, m_sep:]

        pX = X_test @ V_train_x.T  # shape n_test x dim
        y_hat = pX @ V_train_y  # shape n_test x m_test
        var_exps.append(r2_score(Y_test.ravel(), y_hat.ravel()))

    optimal_dim = np.argmax(var_exps) + 1
    idx_V = np.argsort(np.concatenate([idx_col_train, idx_col_test]))
    return var_exps, optimal_dim, var_exps[optimal_dim - 1], V_train[:, idx_V][:n_components]
