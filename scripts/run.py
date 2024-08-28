import random
from argparse import Namespace
from typing import Callable

import git
import numpy as np

import torch
import torch.nn as nn
from torch.nn import Module
from torch.optim import Optimizer
from torch.xpu import device
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader
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
    loss = loss_fn(out, data_batch.y)

    loss.backward()
    optimizer.step()
    del out

    return loss.item()


def evaluate(model: Module, loss_fn: Callable, data: Data) -> ([int], [float], [float]):
    model.eval()
    with torch.no_grad():
        logits = model(data)
        pred = logits.max(1)[1]
        acc = pred.eq(data.y).sum().item() / data.y.size()

        loss = loss_fn(logits, data.y)
        return acc, pred.detach().cpu(), loss.detach().cpu()


def run_exp(args: Namespace, dataset: Dataset, model_cls: AbstractGNN, gcn_cls: MessagePassing, seed: int) -> (float, float):
    dataset_size = len(dataset)
    dataloader = DataLoader(dataset, batch_size=dataset_size, shuffle=False)

    model = model_cls(
        gcn_cls, args.problem_size, args.embedding_size, args.hidden_channels, dataset.num_classes, args.dropout, args.device
    ).type(args.dtype).to(args.device)
    optimizer = torch.optim.Adam(model.parameter(), weight_decay=args.weight_decay, lr=args.lr)

    epoch = 0
    best_train_acc = 0
    best_train_loss = float('inf')
    best_epoch = 0
    bad_counter = 0

    for epoch in range(args.epochs):
        full_batch = next(iter(dataloader))
        full_batch.to(args.device)
        train_step(model, loss_func, optimizer, full_batch)

        train_acc, preds, train_loss = evaluate(model, loss_func, full_batch)
        new_best_trigger = train_acc > best_train_acc if args.stop_strategy == 'acc' else train_loss < best_train_loss
        if new_best_trigger:
            best_train_acc = train_acc
            best_train_loss = train_loss
            best_epoch = epoch
            bad_counter = 0
        else:
            bad_counter += 1

        if bad_counter == args.early_stopping_patience:
            break

    print(f"Random seed {seed} | Epochs: {epoch} | Best epoch: {best_epoch}")
    print(f"Best train loss: {best_train_loss:.4f}")
    return best_train_acc, best_train_loss


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
        gcn_cls: MessagePassing = getattr(nn, args.gcn)
    except AttributeError:
        raise AttributeError('Unknown GCN layer class')

    dataset: Dataset = get_dataset(args.domain, data_size=args.data_size, problem_size=args.problem_size,
                                   node_degree=args.node_degree, graph_type=args.graph_type,
                                   dtype=args.dtype, device=args.device)

    repo = git.Repo(search_parent_directories=True)
    sha = repo.head.object.hexsha
    args.sha = sha

    if torch.cuda.is_available() and args.device == 'cuda':
        args.device = f'cuda:{args.cuda}'
    args.dtype = torch.float32

    results = []
    print(args)

    for rnd_seed in tqdm(range(args.rnd_seeds)):
        set_seed(rnd_seed)
        final_train_accuracy, final_train_loss = run_exp(args, dataset, model_cls, gcn_cls, rnd_seed)
        results.append([final_train_accuracy, final_train_loss])

    results = np.array(results)
    train_acc_mean, train_loss_mean = np.mean(results, axis=0)
    train_acc_mean *= 100
    train_acc_std, train_loss_std = np.sqrt(np.var(results, axis=0))
    train_acc_std *= 100


    print(f'{args.gcn} on {args.dataset} | SHA: {sha}')
    print(f'Train acc: {train_acc_mean:.4f} +/- {train_acc_std:.4f} | Train loss: {train_loss_mean:.4f} +/- {train_loss_std:.4f}')
