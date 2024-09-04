import json
from argparse import Namespace
import hashlib
import random
from typing import Callable

import git
import numpy as np

import torch
from torch.nn import Module
from torch.optim import Optimizer
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader
import torch_geometric.nn as pyg_nn
from torch_geometric.nn.conv import MessagePassing

import models
from models.abstract.abstract_gnn import AbstractGNN
from scripts.parser import get_parser
from utils.data import get_dataset, visualize_graph
from utils.loss import loss_func
from utils.model_io import save_model_with_metadata


def train_step(model: Module, loss_fn: Callable, optimizer: Optimizer, data_batch: Data, is_batch: bool = False) -> float:
    model.train()
    optimizer.zero_grad()

    out = model(data_batch)[:, 0]
    loss = loss_fn(out, data_batch.q_matrix, is_batch=is_batch)  # the edge index stores the Q matrix

    loss.backward()
    optimizer.step()

    return loss.detach().item()


def predict(model: Module, data: Data, prob_threshold: float) -> [int]:
    model.eval()
    with torch.no_grad():
        out = model(data)
        pred = (out >= prob_threshold).int()
        return pred.detach().cpu().numpy().squeeze().tolist()


def hash_dict(dictionary: dict, hash_len: int = 6):
    sha1 = hashlib.sha1()
    sha1.update(json.dumps(dictionary, sort_keys=True).encode())
    return sha1.hexdigest()[:hash_len]


def hash_save_model(model: AbstractGNN, model_hyperparams: dict, optimizer_params: dict, args: Namespace, seed: int):
    model_hyperparams['model_cls'] = args.model_cls
    model_hyperparams['gcn_cls'] = args.gcn_cls
    hyperparam_hash = hash_dict(model_hyperparams)
    optparam_hash = hash_dict(optimizer_params)
    save_path = args.save_path.format(domain=args.domain,
                                      domain_params=f'n{args.problem_size}_d{args.node_degree}_{args.graph_type}',
                                      hyperparam_hash=hyperparam_hash, optparam_hash=optparam_hash,
                                      rnd_seed=seed)
    save_model_with_metadata(model, model_hyperparams, save_path)


def run_exp(args: Namespace, dataset: Dataset, model_cls: type[AbstractGNN], gcn_cls: type[MessagePassing], seed: int,
            should_save_model: bool = False) -> (float, [int]):
    dataset_size = len(dataset)
    dataloader = DataLoader(dataset, batch_size=dataset_size, shuffle=False)
    is_batch = dataset_size > 1

    model_hyperparams = {
        "n_nodes": args.problem_size,
        "in_feats": args.embedding_size,
        "hidden_channels": args.hidden_channels,
        "number_classes": dataset.domain.num_classes,
        "dropout": args.dropout
    }
    model: AbstractGNN = model_cls(gcn_cls, **model_hyperparams, device=args.device).type(args.data_type).to(args.device)
    optimizer_params = {
        "lr": args.lr,
        "weight_decay": args.weight_decay,
    }
    optimizer: torch.optim.Optimizer = torch.optim.Adam(model.parameters(), **optimizer_params)

    epoch = 0
    best_train_loss = float('inf')
    best_bit_prediction = torch.zeros((dataset[0].num_nodes,)).type(args.data_type).to(args.device)
    best_epoch = 0
    no_improv_counter = 0
    small_change_counter = 0
    last_loss = None

    for epoch in range(1, args.epochs + 1):
        full_batch = next(iter(dataloader))
        full_batch.to(args.device)
        train_loss = train_step(model, loss_func, optimizer, full_batch, is_batch=is_batch)
        prediction = predict(model, full_batch, args.assignment_threshold)

        if (epoch % min(1000, int(args.epochs // 10))) == 0:
            print(f'Epoch: {epoch}, Loss: {train_loss}')

        new_best_trigger = train_loss < best_train_loss
        if new_best_trigger:
            best_train_loss = train_loss
            best_epoch = epoch
            best_bit_prediction = prediction
            no_improv_counter = 0
        else:
            no_improv_counter += 1

        if last_loss is not None and abs(train_loss - last_loss) < args.early_stopping_small_diff:
            small_change_counter += 1
        else:
            small_change_counter = 0

        if no_improv_counter >= args.early_stopping_patience:
            print("Early stopping triggered due to no improvement")
            break
        if small_change_counter >= args.early_stopping_patience:
            print("Early stopping triggered due to small changes")
            break

        last_loss = train_loss

    if should_save_model:
        hash_save_model(model, model_hyperparams, optimizer_params, args, seed)

    print(f"Random seed {seed} | Epochs: {epoch} | Best epoch: {best_epoch}")
    print(f"Best loss: {best_train_loss:.4f}, last loss: {last_loss:.4f}")
    return best_train_loss, best_bit_prediction


def set_seed(seed):
    # Set the seed for everything
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


if __name__ == '__main__':
    parser = get_parser()
    parsed_args = parser.parse_args()

    try:
        model_class: type[AbstractGNN] = getattr(models, parsed_args.model_cls)
    except AttributeError:
        raise AttributeError('Unknown GNN model class')

    try:
        gcn_class: type[MessagePassing] = getattr(pyg_nn, parsed_args.gcn_cls)
    except AttributeError:
        raise AttributeError('Unknown GCN layer class')

    parsed_args.data_type = getattr(torch, parsed_args.data_type)
    exp_dataset: Dataset = get_dataset(parsed_args.domain, data_size=parsed_args.data_size,
                                       problem_size=parsed_args.problem_size, node_degree=parsed_args.node_degree,
                                       graph_type=parsed_args.graph_type,
                                       dtype=parsed_args.data_type, device=parsed_args.device)

    repo = git.Repo(search_parent_directories=True)
    sha = repo.head.object.hexsha
    parsed_args.sha = sha

    setting_msg = f'{parsed_args.model_cls} with {parsed_args.gcn_cls} on {parsed_args.domain} | SHA: {sha}'
    print(setting_msg)
    print('-' * len(setting_msg))

    if torch.cuda.is_available() and parsed_args.device == 'cuda':
        parsed_args.device = f'cuda:{parsed_args.cuda}'

    losses, predictions = [], []
    for rnd_seed in range(parsed_args.rnd_seeds):
        set_seed(rnd_seed)
        best_loss, best_prediction = run_exp(parsed_args, exp_dataset, model_class, gcn_class, rnd_seed,
                                             should_save_model=True)
        losses.append(best_loss)
        predictions.append(best_prediction)
        for datapoint, pred in zip(exp_dataset, best_prediction):
            # compute correctness of the neural prediction
            size_mis, ind_set, number_violations = exp_dataset.domain.postprocess_gnn([pred], datapoint.nx_graph)
            print(f'{exp_dataset.domain.criterion_name} found by GNN is {size_mis} with {number_violations} violations')

            # visualize the prediction
            visualize_graph(datapoint.nx_graph, [pred])

            # compute an approximate solution using a solver
            ind_set_bitstring_nx, ind_set_nx_size, nx_number_violations, t_solve = exp_dataset.domain.run_solver(datapoint.nx_graph)
            print(f'{exp_dataset.domain.criterion_name} found by solver is {size_mis} with {number_violations} violations')

    losses = np.array(losses)
    train_loss_mean = np.mean(losses, axis=0)
    train_loss_std = np.sqrt(np.var(losses, axis=0))

    print(f'Train loss: {train_loss_mean:.4f} +/- {train_loss_std:.4f}')
