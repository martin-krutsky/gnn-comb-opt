import torch
from torch import nn
import torch.nn.functional as F

from torch_geometric.data import Data
from torch_geometric.nn.conv import MessagePassing

from models.abstract.abstract_gnn import AbstractGNN


class GNN(AbstractGNN):
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
        super(GNN, self).__init__(gnn_layer_cls, n_nodes, in_feats, hidden_channels, number_classes, dropout, device)
        self.embed = nn.Embedding(n_nodes, in_feats)
        self.conv1 = gnn_layer_cls(in_feats, hidden_channels, add_self_loops=False).to(device)
        self.conv2 = gnn_layer_cls(hidden_channels, number_classes, add_self_loops=False).to(device)

    def forward(self, graph_data: Data):
        """
        Run forward propagation step of instantiated model.

        Input:
            graph_data: pyg graph data object containing feature matrix and edge index
        Output:
            h: Output layer weights
        """
        # input step
        embed = self.embed(graph_data.x)
        h = self.conv1(x=embed, edge_index=graph_data.edge_index)
        h = torch.relu(h)
        h = F.dropout(h, p=self.dropout_frac)

        # output step
        h = self.conv2(x=h, edge_index=graph_data.edge_index)
        h = torch.sigmoid(h)

        return h
