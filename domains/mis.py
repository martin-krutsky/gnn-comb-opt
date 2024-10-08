from collections import defaultdict
from itertools import combinations

import networkx as nx
import torch
from networkx import maximal_independent_set as mis
from time import time

from domains.abstract.co_domain import CODomain
from utils.transform import gen_combinations


class MIS(CODomain):
    num_classes = 1
    criterion_name = "Independence number"

    @staticmethod
    def gen_q_dict(nx_g: nx.Graph, penalty: int = 2) -> defaultdict:
        """
        Helper function to generate QUBO matrix for MIS as minimization problem.

        Input:
            nx_g: graph as networkx graph object (assumed to be unweigthed)
        Output:
            Q_dict: QUBO as defaultdict
        """
        # Initialize our Q matrix
        q_dict = defaultdict(int)

        # Update Q matrix for every edge in the graph
        # all off-diagonal terms get penalty
        for (u, v) in nx_g.edges:
            q_dict[(u, v)] = penalty

        # all diagonal terms get -1
        for u in nx_g.nodes:
            q_dict[(u, u)] = -1

        return q_dict

    @staticmethod
    def run_solver(nx_graph: nx.Graph) -> (list, int, int):
        """
        helper function to run traditional solver for MIS.

        Input:
            nx_graph: networkx Graph object
        Output:
            ind_set_bitstring_nx: bitstring solution as list
            ind_set_nx_size: size of independent set (int)
            number_violations: number of violations of ind_set condition
        """
        # compare with traditional solver
        t_start = time()
        ind_set_nx = mis(nx_graph)
        t_solve = time() - t_start
        ind_set_nx_size = len(ind_set_nx)

        # get bitstring list
        nx_bitstring = [1 if (node in ind_set_nx) else 0 for node in sorted(list(nx_graph.nodes))]
        edge_set = set(list(nx_graph.edges))

        # Updated to be able to handle larger scale
        print('Calculating violations...')
        # check for violations
        number_violations = 0
        for ind_set_chunk in gen_combinations(combinations(ind_set_nx, 2), 100000):
            number_violations += len(set(ind_set_chunk).intersection(edge_set))

        return nx_bitstring, ind_set_nx_size, number_violations, t_solve

    @staticmethod
    def postprocess_gnn(best_bitstring: torch.Tensor, nx_graph: nx.Graph) -> (list, int, int):
        """
        helper function to postprocess MIS results

        Input:
            best_bitstring: bitstring as torch tensor
        Output:
            size_mis: Size of MIS (int)
            ind_set: MIS (list of integers)
            number_violations: number of violations of ind_set condition
        """
        # get bitstring as list
        bitstring_list = list(best_bitstring)

        # compute cost
        size_mis = sum(bitstring_list)

        # get independent set
        ind_set = set([node for node, entry in enumerate(bitstring_list) if entry == 1])
        edge_set = set(list(nx_graph.edges))

        print('Calculating violations...')
        # check for violations
        number_violations = 0
        for ind_set_chunk in gen_combinations(combinations(ind_set, 2), 100000):
            number_violations += len(set(ind_set_chunk).intersection(edge_set))

        return size_mis, ind_set, number_violations
