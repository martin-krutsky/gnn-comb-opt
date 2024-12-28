from abc import abstractmethod
from typing import Callable

import torch

from utils.fuzzy_ops import AndMin, AndProd, AndLuk


def loss_qubo(probs: torch.Tensor, q_mat: torch.Tensor, conjunction: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
              has_multiple: bool = False) -> torch.Tensor:
    """
    Function to compute cost value for given fuzzy degree of spin, using given conjunction and predefined q matrix.

    Input:
        probs: (Fuzzy) degree of each node belonging to each class, as a vector
        q_mat: QUBO as torch tensor
    """
    problem_size = q_mat.shape[1]
    if has_multiple:
        probs_ = probs.reshape(-1, problem_size)
        probs_x = probs_.unsqueeze(1).repeat(1, problem_size, 1)
        probs_y = probs_.unsqueeze(2).repeat(1, 1, problem_size)
        q_mat_ = q_mat.reshape(-1, problem_size, problem_size)
        cost = (q_mat_ * conjunction(probs_x, probs_y)).sum(dim=[1,2])
        cost = cost.mean()
    else:
        probs_x = probs.unsqueeze(0).repeat(problem_size, 1)
        probs_y = probs.unsqueeze(1).repeat(1, problem_size)
        # minimize cost = q * max(0, x + y - 1)
        cost = (q_mat * conjunction(probs_x, probs_y)).sum()
    return cost


class QUBOLoss:
    @property
    @abstractmethod
    def conjunction(self):
        raise NotImplementedError

    def __init__(self, regularization: Callable[[torch.Tensor], torch.Tensor] | None = None):
        self.regularization = regularization

    def __call__(self, probs: torch.Tensor, q_mat: torch.Tensor, has_multiple: bool = False):
        qubo = loss_qubo(probs, q_mat, conjunction=self.conjunction, has_multiple=has_multiple)
        if self.regularization is not None:
            qubo += self.regularization(probs)
        return qubo


class MinQUBOLoss(QUBOLoss):
    conjunction = AndMin()


class ProductQUBOLoss(QUBOLoss):
    conjunction = AndProd()


class LukasiewiczQUBOLoss(QUBOLoss):
    conjunction = AndLuk()

