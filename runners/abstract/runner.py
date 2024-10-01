import json
import random
from abc import ABC, abstractmethod
from argparse import Namespace
import hashlib
from typing import Callable

import numpy as np
import torch
from torch.nn import Module
from torch.optim import Optimizer
import torch_geometric.nn as pyg_nn
from torch_geometric.data import Data, Dataset
from torch_geometric.nn.conv import MessagePassing

import models
from models.abstract.abstract_gnn import AbstractGNN
from utils.data import visualize_graph
from utils.model_io import save_model_with_metadata


class Runner(ABC):
    @staticmethod
    def get_torch_classes(model_cls_name: str, conv_layer_cls_name: str) -> (type[AbstractGNN], type[MessagePassing]):
        try:
            model_class: type[AbstractGNN] = getattr(models, model_cls_name)
        except AttributeError:
            raise AttributeError('Unknown GNN model class')
        try:
            gcn_class: type[MessagePassing] = getattr(pyg_nn, conv_layer_cls_name)
        except AttributeError:
            raise AttributeError('Unknown GCN layer class')
        return model_class, gcn_class

    @staticmethod
    def train_step(model: Module, loss_fn: Callable, optimizer: Optimizer, data_batch: Data,
                   is_batch: bool = False) -> float:
        model.train()
        optimizer.zero_grad()

        out = model(data_batch)[:, 0]
        loss = loss_fn(out, data_batch.q_matrix, is_batch=is_batch)  # the edge index stores the Q matrix

        loss.backward()
        optimizer.step()

        return loss.detach().item()

    @staticmethod
    def predict(model: Module, data: Data, prob_threshold: float) -> torch.Tensor:
        model.eval()
        with torch.no_grad():
            out = model(data)
            pred = (out >= prob_threshold).int()
            return pred.detach().cpu().T

    @staticmethod
    def hash_dict(dictionary: dict, hash_len: int = 6):
        sha1 = hashlib.sha1()
        sha1.update(json.dumps(dictionary, sort_keys=True).encode())
        return sha1.hexdigest()[:hash_len]

    @classmethod
    def hash_save_model(cls, model: AbstractGNN, model_hyperparams: dict, optimizer_params: dict, args: Namespace,
                        seed: int):
        model_hyperparams['model_cls'] = args.model_cls
        model_hyperparams['gcn_cls'] = args.gcn_cls
        hyperparam_hash = cls.hash_dict(model_hyperparams)
        optparam_hash = cls.hash_dict(optimizer_params)
        save_path = args.save_path.format(domain=args.domain,
                                          domain_params=f'n{args.problem_size}_d{args.node_degree}_{args.graph_type}',
                                          hyperparam_hash=hyperparam_hash, optparam_hash=optparam_hash,
                                          rnd_seed=seed)
        save_model_with_metadata(model, model_hyperparams, save_path)

    @staticmethod
    def set_seed(seed: int):
        # Set the seed for everything
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    @staticmethod
    def postprocess(dataset: Dataset, prediction, visualize: bool = False):
        improvement = []
        prediction = prediction.reshape(len(dataset), -1)
        for datapoint, pred in zip(dataset, prediction):
            # compute correctness of the neural prediction
            size_mis, ind_set, number_violations = dataset.domain.postprocess_gnn(pred, datapoint.nx_graph)
            print(f'{dataset.domain.criterion_name} found by GNN is {size_mis} with {number_violations} violations')

            if visualize:  # visualize the prediction
                visualize_graph(datapoint.nx_graph, pred)

            # compute an approximate solution using a solver
            ind_set_bitstring_nx, ind_set_nx_size, nx_number_violations, t_solve = dataset.domain.run_solver(
                datapoint.nx_graph)
            print(
                f'{dataset.domain.criterion_name} found by solver is {ind_set_nx_size} with {number_violations} violations')

            improvement.append(size_mis - ind_set_nx_size)
        return improvement

    @classmethod
    @abstractmethod
    def train(cls,  *args, **kwargs) -> (float, torch.Tensor):
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def run(cls,  *args, **kwargs):
        raise NotImplementedError
