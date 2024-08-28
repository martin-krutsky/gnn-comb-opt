from itertools import islice

import torch


def qubo_dict_to_torch(nx_g, q, torch_dtype=None, torch_device=None):
    """
    Output q matrix as torch tensor for given q in dictionary format.

    Input:
        q: QUBO matrix as defaultdict
        nx_g: graph as networkx object (needed for node lables can vary 0,1,... vs 1,2,... vs a,b,...)
    Output:
        q: QUBO as torch tensor
    """
    # get number of nodes
    n_nodes = len(nx_g.nodes)

    # get QUBO q as torch tensor
    q_mat = torch.zeros(n_nodes, n_nodes)
    for (x_coord, y_coord), val in q.items():
        q_mat[x_coord][y_coord] = val

    if torch_dtype is not None:
        q_mat = q_mat.type(torch_dtype)

    if torch_device is not None:
        q_mat = q_mat.to(torch_device)

    return q_mat


def gen_combinations(combs, chunk_size):
    yield from iter(lambda: list(islice(combs, chunk_size)), [])
