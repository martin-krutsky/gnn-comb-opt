from argparse import Namespace
from typing import Callable

import torch
from torch.nn import Module
from torch_geometric.data import Dataset, Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn.conv import MessagePassing

from models.abstract.abstract_gnn import AbstractGNN
from runners.abstract.runner import Runner
import utils.loss as loss_module


class SimpleRunner(Runner):
    @classmethod
    def train_with_single_batch(
            cls, data: Data, args: Namespace, num_nodes: int, model_hyperparams: dict, has_multiple: bool, seed: int,
            save_model: bool = False, visualize: bool = False
        ) -> tuple[float, torch.Tensor, list[torch.Tensor]] | tuple[float, torch.Tensor]:
        cls.set_seed(seed)
        model_cls, gcn_cls, act_cls, reg_func, loss_cls = cls.get_torch_classes(
            args.model_cls, args.gcn_cls, args.activation, args.regularization, args.loss
        )

        model: AbstractGNN = model_cls(gcn_cls, act_cls, **model_hyperparams, device=args.device,
                                       temp_schedule=args.temp_schedule or None, inversed_temp=args.inversed_temp,
                                       nr_tr_epochs=args.epochs or None).type(
            args.data_type).to(args.device)
        optimizer_params = {
            "lr": args.lr,
            "weight_decay": args.weight_decay,
        }
        optimizer: torch.optim.Optimizer = torch.optim.Adam(model.parameters(), **optimizer_params)
        loss: loss_module.QUBOLoss = loss_cls(reg_func)

        best_train_loss = float('inf')
        best_bit_prediction = torch.zeros((num_nodes,)).type(args.data_type).to(args.device)
        best_epoch = 0
        no_improv_counter = 0
        small_change_counter = 0
        last_loss = None
        saved_predictions = []

        for epoch in range(1, args.epochs + 1):
            data.to(args.device)
            train_loss = cls.train_step(model, loss, optimizer, data, has_multiple=has_multiple,
                                        time_step=epoch-1 if args.activation in ['SigmoidTempAnnealing', 'SigmoidBackwardAnnealing'] else None)
            prediction = cls.predict(model, data, args.assignment_threshold,
                                     time_step=epoch-1 if args.activation in ['SigmoidTempAnnealing', 'SigmoidBackwardAnnealing'] else None)

            if (epoch % min(1000, int(args.epochs // 10))) == 0:
                print(f'Epoch: {epoch}, Loss: {train_loss}')
                if visualize:
                    saved_predictions.append(prediction)

            new_best_trigger = train_loss < best_train_loss
            if new_best_trigger:
                best_train_loss = train_loss
                best_epoch = epoch
                best_bit_prediction = prediction
                no_improv_counter = 0
            else:
                no_improv_counter += 1

            if last_loss is not None and abs(train_loss - last_loss) < args.early_stopping_tolerance:
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

        return best_train_loss, best_bit_prediction, saved_predictions

    @classmethod
    def train(cls, args: Namespace, dataset: Dataset, use_as_batch: bool, seed: int, save_model: bool = False,
              visualize: bool = False) -> tuple[list[float], list[torch.Tensor]]:
        cls.set_seed(seed)
        dataset_size = len(dataset)

        model_hyperparams = {
            "n_layers": args.n_layers,
            "n_nodes": args.problem_size,
            "in_feats": args.embedding_size,
            "hidden_channels": args.hidden_channels,
            "number_classes": dataset.domain.num_classes,
            "dropout": args.dropout,
            "gcn_layer_kwargs": args.gcn_layer_kwargs,
        }

        if use_as_batch or dataset_size == 1:
            dataloader = DataLoader(dataset, batch_size=dataset_size, shuffle=False)
        else:
            dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

        best_train_losses, best_bit_predictions = [], []
        for data in dataloader:
            best_train_loss, best_bit_prediction, predictions = cls.train_with_single_batch(
                data, args, dataset[0].num_nodes, model_hyperparams, use_as_batch and dataset_size > 1,
                seed, save_model, visualize
            )
            if visualize and len(predictions) > 0:
                cls.postprocess_animate(dataset, predictions)

            best_train_losses.append(best_train_loss)
            best_bit_predictions.append(best_bit_prediction)
        return best_train_losses, best_bit_predictions

    @classmethod
    def run(cls, args: Namespace, dataset: Dataset, seed: int, visualize: bool = False):
        print(f'Training with random seed {seed}...')
        best_losses, best_predictions = cls.train(args, dataset, args.use_as_batch, seed, save_model=True, visualize=visualize)
        improvements, pred_sizes, solver_sizes, violations = cls.postprocess(dataset, best_predictions, visualize=visualize)
        # avg_loss, std_loss = np.mean(best_losses), np.std(best_losses)
        # avg_improvement, std_improvement = np.mean(improvements), np.std(improvements)
        return best_losses, pred_sizes, solver_sizes, violations, improvements
