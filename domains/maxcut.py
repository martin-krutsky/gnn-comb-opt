from collections import defaultdict
from time import time

import cvxgraphalgs.algorithms.max_cut as mc
import networkx as nx
import torch

from domains.abstract.co_domain import CODomain


class MaxCut(CODomain):
    num_classes = 1
    criterion_name = "Cut size"
    maximization = True

    @staticmethod
    def goemans_williamson_weighted(graph):
        """
        Runs the Goemans-Williamson randomized 0.87856-approximation algorithm for
        MAX-CUT on the graph instance, returning the cut.

        :param graph: (nx.classes.graph.Graph) An undirected graph with no
            self-loops or multiple edges. The graph can either be weighted or
            unweighted, where each edge present is assigned an equal weight of 1.
        """
        adjacency = nx.linalg.adjacency_matrix(graph)
        adjacency = adjacency.toarray()
        solution = mc._solve_cut_vector_program(adjacency)
        sides = mc._recover_cut(solution)

        # nodes = list(graph.nodes)
        # left = {vertex for side, vertex in zip(sides, nodes) if side < 0}
        # right = {vertex for side, vertex in zip(sides, nodes) if side >= 0}
        # return mc.Cut(left, right)
        return sides

    @staticmethod
    def calc_max_cut_size(assignment: list[int], nx_graph: nx.Graph):
        size = 0
        for (u, v) in nx_graph.edges:
            size += bool(assignment[u]) ^ bool(assignment[v])
        return size

    @staticmethod
    def gen_q_dict(nx_g: nx.Graph, penalty: int = 2) -> defaultdict:
        """
        Helper function to generate QUBO matrix for Maximum Cut as minimization problem.

        Input:
            nx_g: graph as networkx graph object (assumed to be unweigthed)
        Output:
            Q_dict: QUBO as defaultdict
        """
        # Initialize our Q matrix
        q_dict = defaultdict(int)

        # Update Q matrix for every edge in the graph
        for (u, v) in nx_g.edges:
            q_dict[(u, v)] = penalty
            q_dict[(v, u)] = penalty

            q_dict[(u, u)] -= 1
            q_dict[(v, v)] -= 1

        return q_dict

    @classmethod
    def run_solver(cls, nx_graph: nx.Graph) -> (list, int, int):
        """
        helper function to run traditional solver for MaxCut.

        Input:
            nx_graph: networkx Graph object
        Output:
            ind_set_bitstring_nx: bitstring solution as list
            ind_set_cut_size: size of the cut defined by node index (int)
            number_violations: number of violations of ind_set condition
        """
        # compare with traditional solver
        t_start = time()
        nx_bitstring = cls.goemans_williamson_weighted(nx_graph)

        t_solve = time() - t_start
        nx_bitstring[nx_bitstring == -1] = 0
        ind_set_nx_size = cls.calc_max_cut_size(nx_bitstring, nx_graph)
        return nx_bitstring, ind_set_nx_size, 0, t_solve

    @classmethod
    def postprocess_gnn(cls, best_bitstring: torch.Tensor, nx_graph: nx.Graph) -> (int, set, int):
        """
        helper function to postprocess MaxCut results

        Input:
            best_bitstring: bitstring as torch tensor
        Output:
            size_mis: Size of MaxCut (int)
            selected_set: one of the set defined by MaxCut (list of integers)
            number_violations: number of violations of ind_set condition
        """
        # get bitstring as list
        bitstring_list = list(best_bitstring)

        # compute cost
        size_maxcut = cls.calc_max_cut_size(bitstring_list, nx_graph)

        # get independent set
        selected_set = set([node for node, entry in enumerate(bitstring_list) if entry == 1])

        return size_maxcut, selected_set, 0
