import random
from typing import Callable

import git
import numpy as np

import torch
import torch.nn as nn
from torch.nn import Module
from torch.optim import Optimizer
from torch_geometric.data import Data
from tqdm import tqdm

from scripts.parser import get_parser
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


def run_exp(args, dataset, model_cls, fold):
    data = get_fixed_splits(dataset, args['dataset'], fold)
    data = data.to(args['device'])

    model = model_cls(data.edge_index, args)
    model = model.to(args['device'])

    sheaf_learner_params, other_params = model.grouped_parameters()
    optimizer = torch.optim.Adam([
        {'params': sheaf_learner_params, 'weight_decay': args['sheaf_decay']},
        {'params': other_params, 'weight_decay': args['weight_decay']}
    ], lr=args['lr'])

    epoch = 0
    best_val_acc = test_acc = 0
    best_val_loss = float('inf')
    val_loss_history = []
    val_acc_history = []
    best_epoch = 0
    bad_counter = 0

    for epoch in range(args['epochs']):
        train_step(model, loss_func, optimizer, data)

        [train_acc, val_acc, tmp_test_acc], preds, [
            train_loss, val_loss, tmp_test_loss] = evaluate(model, loss_func, data)
        if fold == 0:
            res_dict = {
                f'fold{fold}_train_acc': train_acc,
                f'fold{fold}_train_loss': train_loss,
                f'fold{fold}_val_acc': val_acc,
                f'fold{fold}_val_loss': val_loss,
                f'fold{fold}_tmp_test_acc': tmp_test_acc,
                f'fold{fold}_tmp_test_loss': tmp_test_loss,
            }

        new_best_trigger = val_acc > best_val_acc if args['stop_strategy'] == 'acc' else val_loss < best_val_loss
        if new_best_trigger:
            best_val_acc = val_acc
            best_val_loss = val_loss
            test_acc = tmp_test_acc
            best_epoch = epoch
            bad_counter = 0
        else:
            bad_counter += 1

        if bad_counter == args['early_stopping']:
            break

    print(f"Fold {fold} | Epochs: {epoch} | Best epoch: {best_epoch}")
    print(f"Test acc: {test_acc:.4f}")
    print(f"Best val acc: {best_val_acc:.4f}")

    keep_running = False if test_acc < args['min_acc'] else True
    return test_acc, best_val_acc, keep_running


if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()

    try:
        model_cls = getattr(nn, args.gcn)
    except AttributeError:
        raise AttributeError('Unknown GCN layer')

    dataset = get_dataset(args.dataset)

    repo = git.Repo(search_parent_directories=True)
    sha = repo.head.object.hexsha
    args.sha = sha

    args.device = torch.device(f'cuda:{args.cuda}' if torch.cuda.is_available() else 'cpu')

    # Set the seed for everything
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    results = []
    print(args)

    for fold in tqdm(range(args.folds)):
        test_accuracy, best_val_accuracy, keep_running = run_exp(args, dataset, model_cls, fold)
        results.append([test_accuracy, best_val_accuracy])
        if not keep_running:
            break

    results = np.array(results)
    test_acc_mean, val_acc_mean = np.mean(results, axis=0) * 100
    test_acc_std = np.sqrt(np.var(results, axis=0)[0]) * 100


    print(f'{args.gcn} on {args.dataset} | SHA: {sha}')
    print(f'Test acc: {test_acc_mean:.4f} +/- {test_acc_std:.4f} | Val acc: {val_acc_mean:.4f}')
