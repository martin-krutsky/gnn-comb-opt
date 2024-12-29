import numpy as np
import torch
from torch.autograd import Function
from torch.nn import Sigmoid, Module


class SigmoidTempAnnealing(Module):
    min_mult: int = 1
    max_mult: int = 100

    def __init__(self, schedule='linear', training_steps=None):
        super(SigmoidTempAnnealing, self).__init__()
        if schedule == 'linear':
            self.schedule = np.linspace(self.min_mult, self.max_mult, training_steps)
        elif schedule == 'logarithmic':
            self.schedule = np.logspace(self.min_mult, self.max_mult, training_steps)
        elif schedule == 'geometric':
            self.schedule = np.geomspace(self.min_mult, self.max_mult, training_steps)
        else:
            raise Exception('Unsupported temperature annealing schedule name')

    def forward(self, x: torch.Tensor, time_idx: int):
        return torch.sigmoid(x * self.schedule[time_idx])


class SignSTE(Function):
    @staticmethod
    def forward(ctx, x):
        """
        Forward pass: Binarized sigmoid as a step function.
        """
        ctx.save_for_backward(x)
        x = (x >= 0).float()
        return x

    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward pass: Use the gradient of the straight-through estimation.
        """
        act_input, = ctx.saved_tensors
        mask = act_input.ge(-0.5) & act_input.le(0.5)
        grad_input = torch.where(
            mask, grad_output, torch.zeros_like(grad_output))
        return grad_input


class SignSigmoid(Function):
    @staticmethod
    def forward(ctx, x):
        """
        Forward pass: Binarized sigmoid as a step function.
        """
        ctx.save_for_backward(x)
        x = (x >= 0).float()
        return x

    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward pass: Use the gradient of the standard sigmoid function.
        """
        act_input, = ctx.saved_tensors

        # Compute the sigmoid for the gradient
        sigmoid_grad = torch.sigmoid(act_input) * (1 - torch.sigmoid(act_input))
        return grad_output * sigmoid_grad


def l1(x: torch.Tensor, alpha: float = 0.1) -> torch.Tensor:
    return alpha * torch.norm(x, 1)


def entropy(x: torch.Tensor, alpha: float = 0.1) -> torch.Tensor:
    return alpha * torch.nn.functional.l1_loss(x, x)
