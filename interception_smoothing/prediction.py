from sparse_smoothing.cert import p_lower_upper_from_votes
import torch.nn.functional as F
from scipy.stats import binom_test
from tqdm.auto import tqdm
import numpy as np
import torch


def smoothed_gnn(model, data, nc, n_samples, progress_bar=True):
    """
    Estimate smoothed classifier by drawing MC samples.

    Parameters
    ----------
    model : nn.Module
        Base classifier GNN
    data : torch_geometric.data.data.Data
        Graph dataset
    nc : int
        Number of classes
    n_samples : int
        Number of MC samples used to estimate smoothed classifier
    progress_bar : bool
        Whether you want to show the estimation progress

    Returns
    -------
    votes : np.array [n, nc]
        Votes per class for each node
    """
    A, x, edge_idx, edge_attr = data.A, data.x, data.edge_index, data.edge_attr
    n = A.shape[0]

    votes = torch.zeros(n, nc, dtype=torch.int32)
    for _ in tqdm(range(n_samples), disable=not progress_bar):
        predictions = model(x, edge_idx, edge_attr).argmax(1).cpu()
        votes += F.one_hot(predictions, int(nc))
    return votes


def predict(pre_votes, alpha=0.01):
    """See (Cohen et al., 2019) - predict."""
    top2, top2idx = torch.topk(pre_votes, k=2)
    p_values = [binom_test(top2[idx, 0], top2[idx, 0]+top2[idx, 1], 0.5)
                for idx in range(top2.shape[0])]
    p_values = np.array(p_values)
    abstain = p_values > alpha
    return top2idx[:, 0].numpy(), abstain


def predict_estimate(votes, pre_votes, n_samples, alpha=0.01):
    """Multiclass estimation - (Bojchevski, 2020)."""
    y_hat = pre_votes.argmax(1).numpy()
    p_lower, p_upper = p_lower_upper_from_votes(
        votes, pre_votes, conf_alpha=alpha, n_samples=n_samples)
    abstain = p_lower <= p_upper
    return y_hat, p_lower, p_upper, abstain
