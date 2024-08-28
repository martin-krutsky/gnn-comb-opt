import random
from argparse import Namespace
from typing import Callable

import git
import numpy as np

import torch
from torch.nn import Module
from torch.optim import Optimizer
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader
import torch_geometric.nn as pyg_nn
from torch_geometric.nn import MessagePassing
from tqdm import tqdm

import models
from models.abstract.abstract_gnn import AbstractGNN
from scripts.parser import get_parser
from utils.data import get_dataset
from utils.loss import loss_func


def train_step(model: Module, loss_fn: Callable, optimizer: Optimizer, data_batch: Data) -> float:
    model.train()
    optimizer.zero_grad()

    out = model(data_batch)
    loss = loss_fn(out, data_batch.edge_index)  # the edge index stores the Q matrix

    loss.backward()
    optimizer.step()
    del out

    return loss.detach().item()


def predict(model: Module, data: Data, prob_threshold: float) -> [int]:
    model.eval()
    with torch.no_grad():
        out = model(data)
        pred = (out >= prob_threshold).int()
        return pred.detach().cpu().numpy().tolist()


def run_exp(args: Namespace, dataset: Dataset, model_cls: AbstractGNN, gcn_cls: MessagePassing, seed: int) -> (float, [int]):
    dataset_size = len(dataset)
    dataloader = DataLoader(dataset, batch_size=dataset_size, shuffle=False)

    model = model_cls(
        gcn_cls, args.problem_size, args.embedding_size, args.hidden_channels, dataset.num_classes, args.dropout, args.device
    ).type(args.dtype).to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), weight_decay=args.weight_decay, lr=args.lr)

    epoch = 0
    best_train_loss = float('inf')
    best_bit_prediction = torch.zeros((dataset[0].num_nodes,)).type(args.dtype).to(args.device)
    best_epoch = 0
    no_improv_counter = 0
    small_change_counter = 0
    last_loss = None

    for epoch in range(args.epochs):
        full_batch = next(iter(dataloader))
        full_batch.to(args.device)
        train_loss = train_step(model, loss_func, optimizer, full_batch)
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

        if no_improv_counter == args.early_stopping_patience or small_change_counter == args.early_stopping_patience:
            break

        last_loss = train_loss

    print(f"Random seed {seed} | Epochs: {epoch} | Best epoch: {best_epoch}")
    print(f"Best loss: {best_train_loss:.4f}, last loss: {last_loss:.4f}")
    return best_train_loss, best_bit_prediction


def set_seed(seed):
    # Set the seed for everything
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()

    try:
        model_cls: AbstractGNN = getattr(models, args.model)
    except AttributeError:
        raise AttributeError('Unknown GNN model class')

    try:
        gcn_cls: MessagePassing = getattr(pyg_nn, args.gcn)
    except AttributeError:
        raise AttributeError('Unknown GCN layer class')

    dataset: Dataset = get_dataset(args.domain, data_size=args.data_size, problem_size=args.problem_size,
                                   node_degree=args.node_degree, graph_type=args.graph_type,
                                   dtype=args.data_type, device=args.device)

    repo = git.Repo(search_parent_directories=True)
    sha = repo.head.object.hexsha
    args.sha = sha

    print(f'{args.gcn} on {args.domain} | SHA: {sha}')

    if torch.cuda.is_available() and args.device == 'cuda':
        args.device = f'cuda:{args.cuda}'
    args.dtype = torch.float32

    results = []
    print(args)

    for rnd_seed in tqdm(range(args.rnd_seeds)):
        set_seed(rnd_seed)
        best_loss, best_prediction = run_exp(args, dataset, model_cls, gcn_cls, rnd_seed)
        results.append([best_loss, best_prediction])

    results = np.array(results)
    train_loss_mean = np.mean(results, axis=0)[0]
    train_loss_std = np.sqrt(np.var(results, axis=0)[0])

    # TODO: evaluate final predictions
    # print(f'Train loss: {train_loss_mean:.4f} +/- {train_loss_std:.4f}')
