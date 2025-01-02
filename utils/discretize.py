import numpy as np
import torch
from torch.autograd import Function
from torch.nn import Sigmoid, Module


def get_schedule(name, min_mult, max_mult, steps):
    if name == 'linear':
        schedule = np.linspace(min_mult, max_mult, num=steps)
    elif name == 'logarithmic':
        schedule = np.log(np.linspace(2, 2**max_mult, num=steps))
    elif name == 'geometric':
        schedule = np.geomspace(min_mult, max_mult, num=steps)
    elif name == 'inversed':
        schedule = 1 / np.geomspace(min_mult, max_mult, num=steps)
    # elif schedule == 'constant':  # for debugging purposes only
    #     self.schedule = np.ones(training_steps)
    else:
        raise Exception('Unsupported temperature annealing schedule name')
    return schedule


class SigmoidTempAnnealing(Module):
    min_mult: int = 1
    max_mult: int = 10

    def __init__(self, schedule='linear', training_steps=None):
        super(SigmoidTempAnnealing, self).__init__()
        self.schedule = get_schedule(schedule, self.min_mult, self.max_mult, training_steps)

    def forward(self, x: torch.Tensor, time_idx: int):
        return torch.sigmoid(x * self.schedule[time_idx])


class _SigmoidBackwardAnnealing(Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, temperature: torch.Tensor):
        ctx.save_for_backward(x, temperature)
        output = torch.sigmoid(x)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        act_input, temperature = ctx.saved_tensors

        # Compute the gradient of sigmoid with temperature
        sigmoid_grad = temperature * torch.sigmoid(temperature * act_input) * (1 - torch.sigmoid(temperature * act_input))
        return grad_output * sigmoid_grad, torch.zeros_like(temperature)


class SigmoidBackwardAnnealing(Module):
    min_mult: int = 1
    max_mult: int = 10

    def __init__(self, schedule='linear', training_steps=None):
        super(SigmoidBackwardAnnealing, self).__init__()
        self.schedule = get_schedule(schedule, self.min_mult, self.max_mult, training_steps)

    def forward(self, x: torch.Tensor, time_idx: int):
        output = _SigmoidBackwardAnnealing.apply(x, torch.tensor(self.schedule[time_idx]))
        return output


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

        # Compute the gradient of sigmoid
        sigmoid_grad = torch.sigmoid(act_input) * (1 - torch.sigmoid(act_input))
        return grad_output * sigmoid_grad


def l1(x: torch.Tensor, alpha: float = 0.1) -> torch.Tensor:
    return alpha * torch.norm(x, 1)


def entropy(x: torch.Tensor, alpha: float = 0.1) -> torch.Tensor:
    return alpha * torch.nn.functional.l1_loss(x, x)
