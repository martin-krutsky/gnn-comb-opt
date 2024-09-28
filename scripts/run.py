import warnings

import git
import numpy as np

import torch
from torch_geometric.data import Dataset

from scripts.parser import get_parser
from utils.data import get_dataset
from runners import *


def evaluate_final_results(losses, improvements, nr_of_seeds):
    losses = np.array(losses)
    train_loss_mean = np.mean(losses, axis=0)
    train_loss_std = np.sqrt(np.var(losses, axis=0))
    print(f'Mean loss across {nr_of_seeds} seeds: {train_loss_mean:.4f} +/- {train_loss_std:.4f}')

    improvements = np.array(improvements)
    improved, worsened, equal = improvements[improvements > 0], improvements[improvements < 0], improvements[
        np.isclose(improvements, 0)]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        print(f'Neural network found '
              f'better solution in {len(improved)} case(s) (avg. {np.nan_to_num(improved.mean()):.4f} +- {np.nan_to_num(improved.std()):.4f}), '
              f'worse solution in {len(worsened)} case(s) (avg. {np.nan_to_num(worsened.mean()):.4f} +- {np.nan_to_num(worsened.std()):.4f}), '
              f'and equal solution in {len(equal)} case(s)')


if __name__ == '__main__':
    parser = get_parser()
    parsed_args = parser.parse_args()

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

    losses_ls, predictions_ls, improvements_ls = [], [], []

    if parsed_args.seed is None and parsed_args.rnd_seeds is not None:
        rnd_seeds = list(range(parsed_args.rnd_seeds))
    else:
        rnd_seeds = [parsed_args.seed]

    for rnd_seed in rnd_seeds:
        if parsed_args.use_ray_tune:
            best_training_loss, best_prediction, improvement_to_solver = RayRunner.run(
                parsed_args, exp_dataset, rnd_seed,
                ray_address=parsed_args.ray_address, tracking_uri=parsed_args.tracking_uri, experiment_name="krutsma1-gnn-comb-opt",
                num_raytune_samples=parsed_args.num_raytune_samples, visualize=False
            )
        else:
            best_training_loss, best_prediction, improvement_to_solver = SimpleRunner.run(
                parsed_args, exp_dataset, rnd_seed, visualize=parsed_args.visualize
            )

        losses_ls.append(best_training_loss)
        predictions_ls.append(best_prediction)
        improvements_ls.append(improvement_to_solver)

    print('\n')
    evaluate_final_results(losses_ls, improvements_ls, len(rnd_seeds))
