import torch
from torch.autograd import Function
from torch.nn import Sigmoid


def temp_sigmoid(x, temp):
    return torch.sigmoid(x/(temp))


class SignSTE(Function):
    @staticmethod
    def forward(ctx, x):
        """
        Forward pass: Binarized sigmoid as a step function.
        """
        ctx.save_for_backward(x)
        x = x.sign()
        return x

    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward pass: Use the gradient of the straight-through estimation.
        """
        act_input, = ctx.saved_tensors
        mask = act_input.ge(-1) & act_input.le(1)
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
        x = x.sign()
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

