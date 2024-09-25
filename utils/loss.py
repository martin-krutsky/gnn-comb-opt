import torch


def loss_qubo(probs: torch.Tensor, q_mat: torch.Tensor, is_batch: bool = False) -> torch.Tensor:
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


def loss_linear_interp(probs: torch.Tensor, q_mat: torch.Tensor, is_batch: bool = False) -> torch.Tensor:
    if is_batch:
        problem_size = q_mat.shape[1]
        probs_ = probs.reshape(-1, problem_size)
        q_mat_ = q_mat.reshape(-1, problem_size, problem_size)

        max_q = torch.maximum(torch.zeros(q_mat_.size()), q_mat_)
        min_q = torch.minimum(torch.zeros(q_mat_.size()), q_mat_)

        x_rows = probs_.unsqueeze(1).repeat(1, probs_.shape[1], 1)
        x_cols = probs_.unsqueeze(2).repeat(1, 1, probs_.shape[1])
        max_xs = torch.maximum(torch.zeros(q_mat_.size()), x_rows + x_cols - 1)
    else:
        max_q = torch.maximum(torch.zeros(q_mat.size()), q_mat)
        min_q = torch.minimum(torch.zeros(q_mat.size()), q_mat)

        x_rows = probs.repeat(probs.shape[0], 1)
        x_cols = probs.repeat(1, probs.shape[0])
        max_xs = torch.maximum(torch.zeros(q_mat.size()), x_rows + x_cols - 1)

    mat_res = (max_q + min_q) * max_xs
    return mat_res.sum()

