import numpy as np


def compute_delta_single_source(paths, p_nodes, p_edges):
    """
    Compute single source multiplicative bound (Theorem 2).

    Parameters
    ----------
    paths : dict
        Paths from source to target nodes
    p_nodes : float
        Node feature ablation probability
    p_edges : float
        Edge deletion probability

    Returns
    -------
    delta : float
        Delta of Theorem 2 (probability to receive messages from single node)
    """
    if len(paths) == 0:
        return 1-p_nodes
    # single source check
    assert len(set([p_edges[0][0] for p_edges in paths])) == 1
    delta = (1-p_nodes)*(1-np.prod([1-(1-p_edges)**len(p) for p in paths]))
    return delta


def compute_bounds(all_paths, nodes, idx, p_nodes, p_edges):
    """
    Compute generalized multiplicative bound (Theorem 3).

    Parameters
    ----------
    all_paths : dict
        Paths from source to all target nodes in idx
        (First level: Target nodes, Second level: Source nodes)
    nodes : dict
        Dictionary containing source nodes for all target nodes
    idx : np.array
        Array of target node indices for which we compute delta

    Returns
    -------
    delta : np.array
        Array containing deltas of Theorem 3 for each target node
        (Upper bound on probability to receive adversarial messages)
    """
    multiplicative_bounds = {}

    for t in idx:
        individual_deltas = {}

        for s in nodes[t]:
            paths = get_paths(all_paths, t, [s])
            individual_deltas[s] = compute_delta_single_source(
                paths, p_nodes, p_edges)

        # determine worst-case nodes
        wc_nodes = sorted(individual_deltas.keys(),
                          key=lambda x: individual_deltas[x],
                          reverse=True)

        inner_term = [1-individual_deltas[v] for v in wc_nodes]
        multiplicative_bounds[t] = 1 - np.cumprod(inner_term)

    return multiplicative_bounds


def get_paths(all_paths, target, selected):
    result = []
    for s in selected:
        result.extend(all_paths[target][s])
    return result
