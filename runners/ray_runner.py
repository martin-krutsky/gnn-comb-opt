import os
import tempfile
from argparse import Namespace
from datetime import datetime
from functools import partialmethod

import mlflow
import ray
import torch
from ray.air.integrations.mlflow import MLflowLoggerCallback, setup_mlflow
from ray.train import Checkpoint
from ray.tune.schedulers import ASHAScheduler, TrialScheduler
from torch_geometric.data import Dataset
from torch_geometric.loader import DataLoader

from models.abstract.abstract_gnn import AbstractGNN
from runners.abstract.runner import Runner
from runners.config.ray_config import hyperparams_config
from utils.loss import loss_func


class RayRunner(Runner):
    @classmethod
    def train(cls, config: dict, tracking_uri: str, experiment_name: str, run_name: str, args: Namespace, dataset: Dataset,
              seed: int, retraining_model: bool = False) -> (float, torch.Tensor):
        dataset_size = len(dataset)
        dataloader = DataLoader(dataset, batch_size=dataset_size, shuffle=False)
        is_batch = dataset_size > 1

        model_hyperparams = {
            "n_nodes": args.problem_size,
            "in_feats": config["embedding_size"],
            "hidden_channels": config["hidden_channels"],
            "number_classes": dataset.domain.num_classes,
            "dropout": config["dropout"],
        }

        setup_mlflow(
            config,
            experiment_name=experiment_name,
            tracking_uri=tracking_uri,
            run_name=run_name
        )

        model_cls, gcn_cls = cls.get_torch_classes(args.model_cls, config["gcn_layer"]["layer_name"])
        model: AbstractGNN = model_cls(gcn_cls, **model_hyperparams, device=args.device).type(args.data_type).to(
            args.device)

        optimizer_params = {
            "lr": config["lr"],
            "weight_decay": config["weight_decay"],
        }
        optimizer: torch.optim.Optimizer = torch.optim.Adam(model.parameters(), **optimizer_params)

        if not retraining_model:
            # Load existing checkpoint through `get_checkpoint()` API.
            if ray.train.get_checkpoint():
                loaded_checkpoint = ray.train.get_checkpoint()
                with loaded_checkpoint.as_directory() as loaded_checkpoint_dir:
                    model_state, optimizer_state = torch.load(
                        os.path.join(loaded_checkpoint_dir, "checkpoint.pt")
                    )
                    model.load_state_dict(model_state)
                    optimizer.load_state_dict(optimizer_state)

        best_train_loss = float('inf')
        no_improv_counter = 0
        small_change_counter = 0
        last_loss = None
        prediction = None

        for epoch in range(1, args.epochs + 1):
            full_batch = next(iter(dataloader))
            full_batch.to(args.device)
            train_loss = cls.train_step(model, loss_func, optimizer, full_batch, is_batch=is_batch)
            prediction = cls.predict(model, full_batch, args.assignment_threshold)
            mlflow.log_metrics({
                "train_loss": train_loss,
                "prediction": prediction,
            }, step=epoch)

            if (epoch % min(1000, int(args.epochs // 10))) == 0:
                print(f'Epoch: {epoch}, Loss: {train_loss}')

            new_best_trigger = train_loss < best_train_loss
            if new_best_trigger:
                best_train_loss = train_loss
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

            if not retraining_model:
                with tempfile.TemporaryDirectory() as temp_checkpoint_dir:
                    path = os.path.join(temp_checkpoint_dir, "checkpoint.pt")
                    torch.save(
                        (model.state_dict(), optimizer.state_dict()), path
                    )
                    checkpoint = Checkpoint.from_directory(temp_checkpoint_dir)
                    ray.train.report(
                        {"loss": last_loss, "accuracy": prediction},
                        checkpoint=checkpoint,
                    )

        if retraining_model:
            cls.hash_save_model(model, model_hyperparams, optimizer_params, args, seed)

    @classmethod
    def run(cls, args: Namespace, dataset: Dataset, seed: int, ray_address: str, tracking_uri: str, experiment_name: str,
            visualize: bool = False):
        cls.set_seed(seed)
        time_str = datetime.now().strftime("%d-%m-%Y,%H:%M:%S")
        run_name = f"{args.domain}_{time_str}"
        log_dir = os.path.join(os.getcwd(), "logs")

        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment_name=experiment_name)

        ray.init(address=ray_address, ignore_reinit_error=True, log_to_driver=False)
        train_func = partialmethod(cls.train, cls=cls, tracking_uri=tracking_uri, experiment_name=experiment_name,
                                   run_name=run_name, args=args, dataset=dataset, seed=seed)

        scheduler: TrialScheduler = ASHAScheduler(
            max_t=args.epochs,
            grace_period=1,
            reduction_factor=2
        )
        tuner = ray.tune.Tuner(
            ray.tune.with_resources(
                ray.tune.with_parameters(train_func),
                resources={"cpu": 2}
            ),
            tune_config=ray.tune.TuneConfig(
                metric="loss",
                mode="min",
                scheduler=scheduler,
                num_samples=2,
            ),
            run_config=ray.train.RunConfig(
                name=experiment_name,
                storage_path=log_dir,
                # callbacks=[
                #     MLflowLoggerCallback(
                #         tracking_uri=tracking_uri,
                #         experiment_name=experiment_name,
                #         save_artifact=True,
                #     )
                # ],
            ),
            param_space=hyperparams_config,
        )
        results = tuner.fit()
        best_result = results.get_best_result("loss", "min")
        best_config = best_result.config

        print("Best trial config: {}".format(best_result.config))
        print("Best trial final loss: {}".format(best_result.metrics["loss"]))
        print("Best trial final accuracy: {}".format(best_result.metrics["accuracy"]))

        print("Retraining with best params...")
        best_loss, best_pred = cls.train(best_config, args, dataset, seed, save_model=True)
        improvement = cls.postprocess(dataset, best_pred)
        return best_loss, best_pred, improvement