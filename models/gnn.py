from typing import Any

import torch
from torch import nn
import torch.nn.functional as F

from torch_geometric.data import Data
from torch_geometric.nn.conv import MessagePassing

from models.abstract.abstract_gnn import AbstractGNN


class GNN(AbstractGNN):
    def __init__(self, gnn_layer_cls: type[MessagePassing], n_layers: int, n_nodes: int, in_feats: int, hidden_channels: int,
                 number_classes: int, dropout: float, device: torch.device, gcn_layer_kwargs: dict[str, Any] = None):
        """
        Initialize a new instance of the GNN model of provided size.
        Dropout is added in forward step.

        Inputs:
            in_feats: Dimension of the input (embedding) layer
            hidden_channels: Hidden layer size
            dropout: Fraction of dropout to add between intermediate layer. Value is cached for later use.
            device: Specifies device (CPU vs GPU) to load variables onto
        """
        super(GNN, self).__init__(gnn_layer_cls, n_layers, n_nodes, in_feats, hidden_channels, number_classes, dropout, device)
        self.embed = nn.Embedding(n_nodes, in_feats)
        self.conv_layers = nn.ModuleList()
        for i in range(n_layers):
            if i == 0:
                in_channels = in_feats
                out_channels = hidden_channels
            elif i == n_layers - 1:
                in_channels = hidden_channels
                out_channels = number_classes
            else:
                in_channels = hidden_channels
                out_channels = hidden_channels
            layer = gnn_layer_cls(in_channels=in_channels, out_channels=out_channels, **gcn_layer_kwargs).to(device)
            self.conv_layers.append(layer)

    def forward(self, graph_data: Data) -> torch.Tensor:
        """
        Run forward propagation step of instantiated model.

        Input:
            graph_data: pyg graph data object containing feature matrix and edge index
        Output:
            h: Output layer activations
        """
        # input step
        h = self.embed(graph_data.x)
        for i in range(self.n_layers):
            h = self.conv_layers[i](x=h, edge_index=graph_data.edge_index)
            if i != self.n_layers - 1:
                h = torch.relu(h)
                h = F.dropout(h, p=self.dropout_frac)
        h = torch.sigmoid(h)

        return h
