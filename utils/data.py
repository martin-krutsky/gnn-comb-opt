import os
from typing import Optional

import networkx as nx
import torch
from torch_geometric.data import Dataset, Data, InMemoryDataset
from torch_geometric.utils.convert import from_networkx

import domains
from domains.abstract.AbstractCODomain import AbstractCODomain
from utils.transform import qubo_dict_to_torch

DATASET_DIR = 'datasets/'


def generate_graph(n: int, d: int = None, p: float = None, graph_type: str = 'reg', random_seed: int = 0) -> nx.Graph:
    """
    Helper function to generate a NetworkX random graph of specified type,
    given specified parameters (e.g. d-regular, d=3). Must provide one of
    d or p, d with graph_type='reg', and p with graph_type in ['prob', 'erdos'].

    Input:
        n: Problem size
        d: [Optional] Degree of each node in graph
        p: [Optional] Probability of edge between two nodes
        graph_type: Specifies graph type to generate
        random_seed: Seed value for random generator
    Output:
        nx_graph: NetworkX OrderedGraph of specified type and parameters
    """
    if graph_type == 'reg':
        print(f'Generating d-regular graph with n={n}, d={d}, seed={random_seed}')
        nx_temp = nx.random_regular_graph(d=d, n=n, seed=random_seed)
    elif graph_type == 'prob':
        print(f'Generating p-probabilistic graph with n={n}, p={p}, seed={random_seed}')
        nx_temp = nx.fast_gnp_random_graph(n, p, seed=random_seed)
    elif graph_type == 'erdos':
        print(f'Generating erdos-renyi graph with n={n}, p={p}, seed={random_seed}')
        nx_temp = nx.erdos_renyi_graph(n, p, seed=random_seed)
    else:
        raise NotImplementedError(f'!! Graph type {graph_type} not handled !!')

    # Networkx does not enforce node order by default
    nx_temp = nx.relabel.convert_node_labels_to_integers(nx_temp)
    # nx Graph guarantees order for Python >0 3.7
    nx_graph = nx.Graph()
    nx_graph.add_nodes_from(sorted(nx_temp.nodes()))
    nx_graph.add_edges_from(nx_temp.edges)
    return nx_graph


def get_dataset(domain_name: str, data_size: int = 1, problem_size: int = 10, node_degree: int = 3, graph_type: str = 'reg',
                dtype: str = 'float32', device: str = 'cpu') -> Dataset:
    dataset_path = os.path.join(DATASET_DIR, f'{domain_name}.pkl')
    if not os.path.isfile(dataset_path):
        os.makedirs(DATASET_DIR, exist_ok=True)
        dtype = getattr(torch, dtype)

        try:
            domain_cls: AbstractCODomain = getattr(domains, domain_name)
        except AttributeError:
            raise AttributeError('Unknown CO domain class')

        data_list = []
        for i in range(data_size):
            nx_graph = generate_graph(n=problem_size, d=node_degree, graph_type=graph_type, random_seed=i)
            q_dict = domain_cls.gen_q_dict(nx_graph)
            edge_index = qubo_dict_to_torch(nx_graph, q_dict, torch_dtype=torch.int64, torch_device=device)
            data = Data(x=torch.range(0, problem_size-1, dtype=torch.int), edge_index=edge_index)
            data_list.append(data)
        InMemoryDataset.save(data_list, dataset_path)
    dataset: Dataset = InMemoryDataset()
    dataset.load(dataset_path)
    os.remove(dataset_path)
    return dataset
