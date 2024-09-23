from argparse import Namespace

import torch
from torch_geometric.data import Dataset
from torch_geometric.loader import DataLoader

from models.abstract.abstract_gnn import AbstractGNN
from runners.abstract.runner import Runner
from utils.loss import loss_qubo


class SimpleRunner(Runner):
    @classmethod
    def train(cls, args: Namespace, dataset: Dataset, seed: int, save_model: bool = False) -> (float, torch.Tensor):
        dataset_size = len(dataset)
        dataloader = DataLoader(dataset, batch_size=dataset_size, shuffle=False)
        is_batch = dataset_size > 1

        model_hyperparams = {
            "n_nodes": args.problem_size,
            "in_feats": args.embedding_size,
            "hidden_channels": args.hidden_channels,
            "number_classes": dataset.domain.num_classes,
            "dropout": args.dropout,
            "gcn_layer_kwargs": args.gcn_layer_kwargs,
        }
        model_cls, gcn_cls = SimpleRunner.get_torch_classes(args.model_cls, args.gcn_cls)

        model: AbstractGNN = model_cls(gcn_cls, **model_hyperparams, device=args.device).type(args.data_type).to(
            args.device)
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
            train_loss = cls.train_step(model, loss_qubo, optimizer, full_batch, is_batch=is_batch)
            prediction = cls.predict(model, full_batch, args.assignment_threshold)

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

        if save_model:
            cls.hash_save_model(model, model_hyperparams, optimizer_params, args, seed)

        print(f"Random seed {seed} | Epochs: {epoch} | Best epoch: {best_epoch}")
        print(f"Best loss: {best_train_loss:.4f}, last loss: {last_loss:.4f}")
        return best_train_loss, best_bit_prediction

    @classmethod
    def run(cls, args: Namespace, dataset: Dataset, seed: int, visualize: bool = False):
        cls.set_seed(seed)
        print(f'Training with random seed {seed}...')
        best_loss, best_pred = cls.train(args, dataset, seed, save_model=True)
        improvement = cls.postprocess(dataset, best_pred, visualize=visualize)
        return best_loss, best_pred, improvement
