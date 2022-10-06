import numpy as np
import networkx as nx
from tqdm.auto import tqdm

from .prediction import predict_estimate
from .cert import compute_bounds


def compute_certificates(model, all_paths, distances, pre_votes, votes,
                         data, n, nc, idx, hparams, neighborhood):
    p_nodes = hparams["p_eval_nodes"]
    p_edges = hparams["p_eval_edges"]
    y = data.y.cpu().numpy()

    if not all_paths or not distances:
        all_paths, distances = compute_distances(data.A, idx)

    # post-processing distances in case of disconnected nodes
    for t in idx:
        if t not in distances:
            distances[t] = {0: [t]}

        if 1 not in distances[t]:
            distances[t][1] = []

        if 2 not in distances[t]:
            distances[t][2] = []

    # evaluate smoothed classifier
    y_hat, p_lower, p_upper, abstain = predict_estimate(
        votes, pre_votes, n_samples=hparams["n1"], alpha=hparams['alpha'])

    nodes = None
    if hparams["with_skip"]:
        if neighborhood == 1:
            nodes = {t: distances[t][1] + distances[t][2] for t in idx}
        elif neighborhood == 2:
            nodes = {t: distances[t][neighborhood] for t in idx}
    else:
        if neighborhood == 0:
            nodes = {t: distances[t][0] + distances[t]
                     [1] + distances[t][2] for t in idx}
        elif neighborhood == 1:
            nodes = {t: distances[t][1] + distances[t][2] for t in idx}
        elif neighborhood == 2:
            nodes = {t: distances[t][2] for t in idx}

    # max num of certifiable nodes
    sizes = np.array([len(nodes[t]) for t in idx])

    # compute bounding constant \Delta
    bounds = compute_bounds(all_paths, nodes, idx, p_nodes, p_edges)

    # compute certificates
    cert_ratios = {0: (1-abstain[idx]).mean()}
    cert_accs = {0: accuracy(y, y_hat[idx], abstain[idx], idx)}
    full_robustness = {0: np.array((1-abstain[idx]).tolist(), dtype=bool)}

    for radius in range(1, sizes.max()+1):
        deltas = np.array(
            [bounds[t][:radius][-1] if len(bounds[t]) > 0 else 0 for t in idx])
        robust = p_lower[idx] - deltas > p_upper[idx] + deltas
        full_robustness[radius] = robust.tolist()
        robust = np.logical_and(robust, ~abstain[idx])[sizes >= radius]
        cert_ratios[radius] = robust.sum(
        )/robust.shape[0] if robust.shape[0] > 0 else 0
        cert_accs[radius] = certified_accuracy(
            y[sizes >= radius], y_hat[idx][sizes >= radius], robust, idx)

    cert_accs_plot = list(zip(*cert_accs.items()))
    cert_accs_plot = [np.arange(0, len(cert_accs_plot[0]), 1/100),
                      np.repeat(cert_accs_plot[1], 100)]

    max_radii = np.stack(list(full_robustness.values())).sum(0)-1
    max_radii = np.minimum(max_radii, sizes).tolist()

    receptive_field_normalized_ratio = {}
    receptive_field_normalized_acc = {}
    for percentage in np.arange(0, 1+0.001, 0.001):
        certified_ratio = (max_radii >= (percentage*sizes))
        receptive_field_normalized_ratio[percentage] = certified_ratio.mean()
        robust = [full_robustness[r][i]
                  for i, r in enumerate(np.ceil(percentage*sizes))]
        acc = certified_accuracy(y, y_hat[idx], robust, idx)
        receptive_field_normalized_acc[percentage] = acc
    ratio_curve = list(zip(*receptive_field_normalized_ratio.items()))
    acc_curve = list(zip(*receptive_field_normalized_acc.items()))

    results = {
        "clean_acc": cert_accs[0],
        "abstain_ratio": abstain[idx].mean(),
        "cert_ratio": cert_ratios,
        "cert_acc": cert_accs,
        "cert_acc_curve": acc_curve,
        "cert_ratio_curve": ratio_curve
    }

    return results


def compute_distances(A, idx, num_layers=2):
    """
    Compute distances from source nodes to all target nodes, \
        respecting the corresponding receptive fields of the GNN.

    Parameters
    ----------
    A : sp.csr_matrix
        Adjacency matrix of the graph
    idx : np.array
        Indices of target nodes
    num_layers : int
        Number of GNN layers

    Returns
    -------
    paths : dict
        Simple paths from source to target nodes
        (First level: Target nodes, Second level: Source nodes)
    distances : dict
        Distances from source to target nodes
    """
    G = nx.DiGraph()
    G.add_edges_from(list((int(e[0]), int(e[1])) for e in zip(*A.nonzero())))
    distances = nx.all_pairs_shortest_path_length(G.reverse(),
                                                  cutoff=num_layers)
    distances = {k[0]: k[1] for k in list(distances)}

    for v in distances:
        new_dict = {}
        for w in distances[v]:
            new_dict.setdefault(distances[v][w], []).append(w)
        distances[v] = new_dict

    paths = {}
    for target in tqdm(distances):
        if target not in idx:
            continue
        paths[target] = {}
        for n in distances[target]:
            for source in distances[target][n]:
                if source == target:
                    paths[target][source] = []
                    continue
                paths[target][source] = list(nx.all_simple_edge_paths(
                    G, source=source, target=target, cutoff=num_layers))
    return paths, distances


def accuracy(y, y_hat, abstain, idx):
    y = y[~abstain]
    y_hat = y_hat[~abstain]
    return (y == y_hat).sum()/idx.shape[0]


def certified_accuracy(y, y_hat, robust_not_abstained, idx):
    cert_and_accurate = y[robust_not_abstained] == y_hat[robust_not_abstained]
    return (cert_and_accurate).sum()/idx.shape[0]
