import torch
from torch import Tensor


def loss_func(probs: Tensor, q_mat: Tensor) -> Tensor:
    """
    Function to compute cost value for given probability of spin [prob(+1)] and predefined q matrix.

    Input:
        probs: Probability of each node belonging to each class, as a vector
        q_mat: QUBO as torch tensor
    """
    probs_ = torch.unsqueeze(probs, 1)
    # minimize cost = x.T * q * x
    cost = (probs_.T @ q_mat @ probs_).squeeze()
    return cost
