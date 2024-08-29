from abc import ABC, abstractmethod, abstractproperty


class AbstractCODomain(ABC):
    @property
    @abstractmethod
    def num_classes(self):
        pass

    @staticmethod
    @abstractmethod
    def gen_q_dict(nx_g, penalty=2):
        """
        Helper function to generate QUBO matrix for a CO domain as minimization problem.

        Input:
            nx_g: graph as networkx graph object (assumed to be unweigthed)
        Output:
            Q_dict: QUBO as defaultdict
        """
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def run_solver(nx_graph):
        """
        helper function to run traditional solver for a CO domain.

        Input:
            nx_graph: networkx Graph object
        Output:
            ind_set_bitstring_nx: bitstring solution as list
            ind_set_nx_size: size of independent set (int)
            number_violations: number of violations of ind.set condition
        """
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def postprocess_gnn(best_bitstring, nx_graph):
        """
        helper function to postprocess results

        Input:
            best_bitstring: bitstring as torch tensor
        Output:
            size: Size of problem (int)
            ind_set: definition of a problem (list of integers)
            number_violations: number of violations of ind_set condition
        """
        raise NotImplementedError