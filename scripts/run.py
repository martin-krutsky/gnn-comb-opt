import os.path
import warnings

import git
import numpy as np
import pandas as pd

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
    if parsed_args.regularization != '' and parsed_args.activation != 'Sigmoid':
        parser.error(f'Choosing soft discretization via regularization: ({parsed_args.regularization}), '
                     f'is incompatible with binarized activation ({parsed_args.activation})')

    if parsed_args.activation == 'SigmoidTempAnnealing' and parsed_args.temp_schedule == '':
        parser.error(f'For temperature annealed sigmoid, specify a non-empty temperature schedule parameter.')
    elif parsed_args.temp_schedule != '' and parsed_args.activation != 'SigmoidTempAnnealing':
        parser.error(f'Non-empty temperature schedule is only compatible with temperature annealed sigmoid.')

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

    result_dict = {key: [] for key in [
        'qubo_loss_avg', 'qubo_loss_std', 'qubo_loss_best',
        'pred_size_avg', 'pred_size_std', 'pred_size_best',
        'solver_size_avg', 'solver_size_std', 'solver_size_best',
        'violation_avg', 'violation_std', 'violation_best',
        'improvement_avg', 'improvement_std', 'improvement_best'
    ]}

    if parsed_args.seed is None and parsed_args.rnd_seeds is not None:
        rnd_seeds = list(range(parsed_args.rnd_seeds))
    else:
        rnd_seeds = [parsed_args.seed]

    data_folder = f'{parsed_args.domain}/{parsed_args.node_degree}/{parsed_args.problem_size}'
    os.makedirs(os.path.join(parsed_args.result_path, data_folder), exist_ok=True)
    for rnd_seed in rnd_seeds:
        if parsed_args.use_ray_tune:
            losses, pred_sizes, solver_sizes, violations, improvements = RayRunner.run(  # TODO adjust raytune
                parsed_args, exp_dataset, rnd_seed,
                ray_address=parsed_args.ray_address, tracking_uri=parsed_args.tracking_uri, experiment_name="krutsma1-gnn-comb-opt",
                num_raytune_samples=parsed_args.num_raytune_samples, visualize=False
            )
        else:
            # TODO: add solver solutions
            losses, pred_sizes, solver_sizes, violations, improvements = SimpleRunner.run(
                parsed_args, exp_dataset, rnd_seed, visualize=parsed_args.visualize
            )

        single_result_df = pd.DataFrame({'qubo_loss': losses, 'pred_size': pred_sizes, 'solver_size': solver_sizes,
                                         'violation': violations, 'improvement': improvements})
        result_file = os.path.join(parsed_args.result_path, data_folder, f'{rnd_seed}.csv')
        single_result_df.to_csv(result_file, index=False)

        for key, value in zip(result_dict.keys(), [
            np.mean(losses), np.std(losses), np.min(losses),
            np.mean(pred_sizes), np.std(pred_sizes), np.max(pred_sizes),
            np.mean(solver_sizes), np.std(solver_sizes), np.max(solver_sizes),
            np.mean(violations), np.std(violations), np.min(violations),
            np.mean(improvements), np.std(improvements), np.max(improvements),
        ]):
            result_dict[key].append(value)

    # if len(avg_losses_ls) > 1:
    result_df = pd.DataFrame(result_dict)
    print(result_df)
    result_file = os.path.join(parsed_args.result_path, data_folder, f'complete_results.csv')
    result_df.to_csv(result_file, index=False)
    # evaluate_final_results(losses_ls, improvements_ls, len(rnd_seeds))
