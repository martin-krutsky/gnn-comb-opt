import torch
from torch import Tensor


def loss_func(probs: Tensor, q_mat: Tensor, is_batch: bool = False) -> Tensor:
    """
    Function to compute cost value for given probability of spin [prob(+1)] and predefined q matrix.

    Input:
        probs: Probability of each node belonging to each class, as a vector
        q_mat: QUBO as torch tensor
    """
    if is_batch:
        problem_size = q_mat.shape[1]
        probs_ = probs.reshape(-1, problem_size)
        q_mat_ = q_mat.reshape(-1, problem_size, problem_size)
        temp_mat = torch.einsum('ijk,ik->ij', q_mat_, probs_)
        cost = torch.einsum('ij,ij->i', probs_, temp_mat)
        cost = cost.mean()
    else:
        probs_ = torch.unsqueeze(probs, 1)
        # minimize cost = x.T * q * x
        cost = (probs_.T @ q_mat @ probs_).squeeze()
    return cost
