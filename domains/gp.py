from collections import defaultdict

import networkx as nx
import torch
from time import time

from domains import MaxCut
from domains.abstract.co_domain import CODomain


class GraphPartition(MaxCut):
    num_classes = 1
    criterion_name = "Nr of divided edges"
    maximization = False

    @staticmethod
    def gen_q_dict(nx_g: nx.Graph, penalty: int = 10) -> defaultdict:
        """
        Helper function to generate QUBO matrix for Graph Partition as minimization problem.

        Input:
            nx_g: graph as networkx graph object (assumed to be unweigthed)
        Output:
            Q_dict: QUBO as defaultdict
        """
        # Initialize our Q matrix
        q_dict = super().gen_q_dict(nx_g, penalty=penalty)

        for (u, v) in nx_g.edges:
            if v > u:
                q_dict[(u, v)] -= penalty * 2

        nr_of_nodes = len(nx_g)
        for u in nx_g.nodes:
            q_dict[(u, u)] += penalty * (nr_of_nodes-1)

        return q_dict

    @staticmethod
    def run_solver(nx_graph: nx.Graph) -> (list, int, int):
        """
        helper function to run traditional solver for MaxCut.

        Input:
            nx_graph: networkx Graph object
        Output:
            ind_set_bitstring_nx: bitstring solution as list
            ind_set_cut_size: size of the cut defined by node index (int)
            number_violations: number of violations of ind_set condition
        """
        raise NotImplementedError

    @staticmethod
    def postprocess_gnn(best_bitstring: torch.Tensor, nx_graph: nx.Graph) -> (list, int, int):
        """
        helper function to postprocess MaxCut results

        Input:
            best_bitstring: bitstring as torch tensor
        Output:
            size_mis: Size of MaxCut (int)
            ind_set: one of the set defined by MaxCut (list of integers)
            number_violations: number of violations of ind_set condition
        """
        raise NotImplementedError
