import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.data import Data
from torch_geometric.nn.conv import MessagePassing


class GNN(nn.Module):
    def __init__(self, gnn_layer_cls: MessagePassing, in_feats: int, hidden_channels: int, number_classes: int,
                 dropout: float, device: torch.device):
        """
        Initialize a new instance of the GNN model of provided size.
        Dropout is added in forward step.

        Inputs:
            in_feats: Dimension of the input (embedding) layer
            hidden_channels: Hidden layer size
            dropout: Fraction of dropout to add between intermediate layer. Value is cached for later use.
            device: Specifies device (CPU vs GPU) to load variables onto
        """
        super(GNN, self).__init__()

        self.dropout_frac = dropout
        self.conv1 = gnn_layer_cls(in_feats, hidden_channels).to(device)
        self.conv2 = gnn_layer_cls(hidden_channels, number_classes).to(device)

    def forward(self, graph_data: Data):
        """
        Run forward propagation step of instantiated model.

        Input:
            graph_data: pyg graph data object containing feature matrix and edge index
        Output:
            h: Output layer weights
        """

        # input step
        h = self.conv1(graph_data.x, graph_data.edge_attr)
        h = torch.relu(h)
        h = F.dropout(h, p=self.dropout_frac)

        # output step
        h = self.conv2(h, graph_data.edge_attr)
        h = torch.sigmoid(h)

        return h
