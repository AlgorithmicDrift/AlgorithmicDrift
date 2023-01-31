import torch
import torch.nn as nn

from regressor import GradientReversalLayer

class Classifier(nn.Module):

    def __init__(self, n_features, n_hidden=32, p_dropout=0.2, grad_scaling=None):
        super(Classifier, self).__init__()

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
        return torch.sigmoid(self.network(x))
