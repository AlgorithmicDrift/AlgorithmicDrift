import torch
import torch.nn as nn
from torch.autograd import Function


class GRL_(Function):
    """
    Gradient reversal functional
    Unsupervised Domain Adaptation by Backpropagation - Yaroslav Ganin, Victor Lempitsky
    https://arxiv.org/abs/1409.7495
    """

    @staticmethod
    def forward(ctx, input, grad_scaling):
        ctx.grad_scaling = grad_scaling
        return input

    @staticmethod
    def backward(ctx, grad_output):
        # need to return a gradient for each input parameter of the forward() function
        # for parameters that don't require a gradient, we have to return None
        # see https://stackoverflow.com/a/59053469
        return -ctx.grad_scaling * grad_output, None


grl = GRL_.apply


class GradientReversalLayer(nn.Module):
    def __init__(self, grad_scaling):
        """
        Gradient reversal layer
        Unsupervised Domain Adaptation by Backpropagation - Yaroslav Ganin, Victor Lempitsky
        https://arxiv.org/abs/1409.7495
        :param grad_scaling: the scaling factor that should be applied on the gradient in the backpropagation phase
        """
        super().__init__()
        self.grad_scaling = grad_scaling

    def forward(self, input):
        return grl(input, self.grad_scaling)

    def extra_repr(self) -> str:
        return f"grad_scaling={self.grad_scaling}"


class Regressor(nn.Module):

    def __init__(self, n_features, n_hidden=32, p_dropout=0.2, grad_scaling=None):
        super(Regressor, self).__init__()

        if grad_scaling is None:
            self.network = nn.Sequential(
                nn.Linear(n_features, n_hidden),
                nn.ReLU(),
                nn.Dropout(p_dropout),
                nn.Linear(n_hidden, n_hidden),
                nn.ReLU(),
                nn.Dropout(p_dropout),
                nn.Linear(n_hidden, n_hidden),
                nn.ReLU(),
                nn.Dropout(p_dropout),
                nn.Linear(n_hidden, 1),
            )
        else:
            self.network = nn.Sequential(
                GradientReversalLayer(grad_scaling),
                nn.Linear(n_features, n_hidden),
                nn.ReLU(),
                nn.Dropout(p_dropout),
                nn.Linear(n_hidden, n_hidden),
                nn.ReLU(),
                nn.Dropout(p_dropout),
                nn.Linear(n_hidden, n_hidden),
                nn.ReLU(),
                nn.Dropout(p_dropout),
                nn.Linear(n_hidden, 1),
            )

    def forward(self, x):
        return self.network(x)
