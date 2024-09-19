import os
from argparse import Namespace

import onnx2torch
import torch
from torch.nn import Module
import torch_geometric.nn as pyg_nn
from torch_geometric.data import Data
from torch_geometric.nn.conv import MessagePassing

import domains
import models
from domains.abstract.co_domain import CODomain
from models.abstract.abstract_gnn import AbstractGNN
from utils.data import create_data


# save/load state_dict only (requires recreating model instance)
def save_model_state_dict(model: Module, filepath: str):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    torch.save(model.state_dict(), filepath)


def load_model_state_dict(model: Module, filepath: str) -> Module:
    model.load_state_dict(torch.load(filepath))
    return model


# save/load state_dict with metadata about the model and hyperparams
def save_model_with_metadata(model: AbstractGNN, hyperparams: dict, filepath: str):
    save_data = {
        'model_state_dict': model.state_dict(),
        'model_cls_name': type(model).__name__,
        'gcn_cls_name': type(model.gnn_layer_cls).__name__,
        'hyperparams': hyperparams
    }
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    torch.save(save_data, filepath)


def load_model_with_metadata(filepath: str, device: str) -> Module:
    loaded_data = torch.load(filepath)

    try:
        model_cls: type[AbstractGNN] = getattr(models, loaded_data['model_cls_name'])
    except AttributeError:
        raise AttributeError('Cannot load model: Unknown GNN model class')

    try:
        gcn_class: type[MessagePassing] = getattr(pyg_nn, loaded_data['model_cls_name'])
    except AttributeError:
        raise AttributeError('Cannot load model: Unknown GCN layer class')

    model = model_cls(gcn_class, **loaded_data['hyperparams'], device=device)
    model.load_state_dict(loaded_data['model_state_dict'])
    return model


# save/load using the ONNX format (may not be compatible with all pyg operations)
def create_dummy_data_input(args: Namespace) -> Data:
    domain_cls: type[CODomain] = getattr(domains, args.domain)
    data: Data = create_data(domain_cls, rnd_seed=42, problem_size=args.problem_size, node_degree=args.node_degree,
                             graph_type=args.graph_type, dtype=args.data_type, device=args.device, visualize=False)
    return data


def save_model_onnx(model: Module, filepath: str, args: Namespace):
    dummy_data = create_dummy_data_input(args)
    onnx_program = torch.onnx.dynamo_export(model, dummy_data)
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    onnx_program.save(filepath)


def load_model_onnx(filepath: str) -> Module:
    model = onnx2torch.convert(filepath)
    return model
