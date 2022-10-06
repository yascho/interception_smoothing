import numpy as np
import scipy.sparse as sp
from sparse_smoothing.utils import split
from sklearn.model_selection import train_test_split


def load_dataset(name='cora_ml', path="data/graphs/", seed=42):
    return inductive_split(*load_graph(name, path), seed)


def load_graph(name, path):
    graph = np.load(f"{path}{name}.npz")
    A = sp.csr_matrix((np.ones(graph['A'].shape[1]).astype(int), graph['A']))
    data = (np.ones(graph['X'].shape[1]), graph['X'])
    X = sp.csr_matrix(data, dtype=np.float32).todense()
    y = graph['y']
    return A, X, y


def inductive_split(A, X, y, seed):
    n, d = X.shape
    nc = y.max() + 1

    # semi-supervised inductive setting
    idx_train, idx_valid, idx_unlabelled = split(
        labels=y, n_per_class=20, seed=seed)
    idx_unlabelled, idx_test = train_test_split(
        idx_unlabelled, test_size=0.1, random_state=seed)

    # graph splitting
    idx = np.hstack([idx_unlabelled, idx_train])
    train = (A[np.ix_(idx, idx)], X[idx, :], y[idx_train])

    idx = np.hstack([idx_unlabelled, idx_train, idx_valid])
    valid = (A[np.ix_(idx, idx)], X[idx, :], y[idx_valid])

    idx = np.hstack([idx_unlabelled, idx_train, idx_valid, idx_test])
    test = (A[np.ix_(idx, idx)], X[idx, :], y[idx_test])

    len1 = len(idx_unlabelled)
    len2 = len1+len(idx_train)
    len3 = len2+len(idx_valid)
    len4 = len3+len(idx_test)

    final_idx_train = np.arange(len1, len2)
    final_idx_valid = np.arange(len2, len3)
    final_idx_test = np.arange(len3, len4)

    return [A, X, y, n, d, nc, train, valid, test,
            final_idx_train, final_idx_valid, final_idx_test]
