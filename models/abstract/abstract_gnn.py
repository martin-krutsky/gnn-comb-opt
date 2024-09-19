from abc import ABC, abstractmethod

import torch
import torch.nn as nn

from torch_geometric.data import Data
from torch_geometric.nn.conv import MessagePassing


class AbstractGNN(ABC, nn.Module):
    def __init__(self, gnn_layer_cls: type[MessagePassing], n_nodes: int, in_feats: int, hidden_channels: int,
                 number_classes: int, dropout: float, device: torch.device):
        """
        Initialize a new instance of the GNN model of provided size.
        Dropout is added in forward step.

        Inputs:
            in_feats: Dimension of the input (embedding) layer
            hidden_channels: Hidden layer size
            dropout: Fraction of dropout to add between intermediate layer. Value is cached for later use.
            device: Specifies device (CPU vs GPU) to load variables onto
        """
        super(AbstractGNN, self).__init__()
        self.gnn_layer_cls = gnn_layer_cls
        self.n_nodes = n_nodes
        self.in_feats = in_feats
        self.hidden_channels = hidden_channels
        self.number_classes = number_classes
        self.dropout_frac = dropout
        self.device = device


    @abstractmethod
    def forward(self, graph_data: Data) -> torch.Tensor:
        """
        Run forward propagation step of instantiated model.

        Input:
            graph_data: pyg graph data object containing feature matrix and edge index
        Output:
            h: Output layer activations
        """
        raise NotImplementedError
