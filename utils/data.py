import os
from typing import Optional

import matplotlib.pyplot as plt
import networkx as nx
import torch
from torch_geometric.data import Dataset, Data, InMemoryDataset
from torch_geometric.utils.convert import from_networkx

import domains
from domains.abstract.co_domain import CODomain
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


def visualize_graph(nx_graph: nx.Graph, bitstrings: Optional[torch.Tensor] = None):
    pos = nx.kamada_kawai_layout(nx_graph)
    if bitstrings is None:
        nx.draw(nx_graph, pos=pos, with_labels=True)
    else:
        color_map = ['orange' if (bitstrings[node] == 0) else 'lightblue' for node in nx_graph.nodes]
        nx.draw(nx_graph, pos=pos, with_labels=True, node_color=color_map)
    plt.show()


def create_data(domain_cls: type[CODomain], rnd_seed: int = 1, problem_size: int = 10, node_degree: int = 3, graph_type: str = 'reg',
                dtype: torch.dtype = torch.float64, device: str = 'cpu', visualize: bool = False) -> Data:
    nx_graph = generate_graph(n=problem_size, d=node_degree, graph_type=graph_type, random_seed=rnd_seed)
    if visualize:
        visualize_graph(nx_graph)

    q_dict = domain_cls.gen_q_dict(nx_graph)
    q_torch = qubo_dict_to_torch(nx_graph, q_dict, torch_dtype=dtype, torch_device=device)

    data = from_networkx(nx_graph).to(device)
    data.x = torch.arange(0, problem_size, dtype=torch.int)
    data.q_matrix = q_torch
    data.nx_graph = nx_graph
    return data


def get_dataset(domain_name: str, data_size: int = 1, problem_size: int = 10, node_degree: int = 3, graph_type: str = 'reg',
                dtype: torch.dtype = torch.float64, device: str = 'cpu', save_to_file: bool = True) -> Dataset:
    try:
        domain_cls: type[CODomain] = getattr(domains, domain_name)
    except AttributeError:
        raise AttributeError('Unknown CO domain class')

    dataset_path = os.path.join(DATASET_DIR, f'{domain_name}.pkl')
    if not os.path.isfile(dataset_path):
        os.makedirs(DATASET_DIR, exist_ok=True)

        data_list: [Data] = []
        for i in range(1, data_size + 1):
            data: Data = create_data(domain_cls, i, problem_size, node_degree, graph_type, dtype, device)
            data_list.append(data)
        InMemoryDataset.save(data_list, dataset_path)
    dataset: Dataset = InMemoryDataset()
    dataset.load(dataset_path)
    dataset.domain = domain_cls

    if not save_to_file:
        os.remove(dataset_path)
    return dataset
