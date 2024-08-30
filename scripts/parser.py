import argparse

import numpy as np


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()

    # Optimization params
    parser.add_argument('--epochs', type=int, default=int(1e5))
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=0)
    parser.add_argument('--early_stopping_patience', type=int, default=100)
    parser.add_argument('--early_stopping_small_diff', type=float, default=1e-4)
    parser.add_argument('--stop_strategy', type=str, choices=['loss'], default='loss')
    parser.add_argument('--assignment_threshold', type=float, default=0.5)

    DEFAULT_PROBLEM_SIZE = 100
    # Domain params
    parser.add_argument('--domain', choices=['MIS'], default='MIS')
    parser.add_argument('--problem_size', type=int, default=DEFAULT_PROBLEM_SIZE)
    parser.add_argument('--node_degree', type=int, default=3)
    parser.add_argument('--graph_type', type=str, choices=['reg', 'prob', 'erdos'], default='reg')

    DEFAULT_EMBEDDING_SIZE = int(np.sqrt(DEFAULT_PROBLEM_SIZE))
    DEFAULT_HIDDEN_SIZE = int(DEFAULT_EMBEDDING_SIZE / 2)
    # Model hyperparams
    parser.add_argument('--embedding_size', type=int, default=DEFAULT_EMBEDDING_SIZE)
    parser.add_argument('--hidden_channels', type=int, default=DEFAULT_HIDDEN_SIZE)
    parser.add_argument('--dropout', type=float, default=0.0)
    # parser.add_argument('--layers', type=int, default=2)
    parser.add_argument('--model_cls', type=str, choices=['GNN'], default='GNN')
    parser.add_argument('--gcn_cls', type=str, choices=[
        'GCNConv', 'ChebConv', 'SAGEConv', 'GraphConv', 'GatedGraphConv', 'ResGatedGraphConv', 'GATConv', 'GATv2Conv',
        'TransformerConv', 'AGNNConv', 'TAGConv', 'GINConv', 'GINEConv', 'ARMAConv', 'SGConv', 'APPNP', 'MFConv',
        'RGCNConv', 'DNAConv', 'GMMConv', 'SplineConv', 'NNConv', 'CGConv', 'EdgeConv', 'FeaStConv', 'LEConv',
        'PNAConv', 'ClusterGCNConv', 'GENConv', 'GCN2Conv', 'PANConv', 'WLConv', 'FiLMConv', 'SuperGATConv', 'FAConv',
        'EGConv', 'PDNConv', 'GeneralConv'
    ], default='GCNConv')

    # Experiment params
    parser.add_argument('--data_size', type=int, default=1)
    parser.add_argument('--save_path', type=str,
                        default='bin/{domain}/{domain_params}/{hyperparam_hash}_{optparam_hash}/model{rnd_seed}.onnx')
    parser.add_argument('--device', type=str, choices=['cpu', 'cuda'], default='cpu')
    parser.add_argument('--cuda', type=int, default=0)
    parser.add_argument('--data_type', type=str, choices=['float32', 'float64'], default='float32')
    parser.add_argument('--rnd_seeds', type=int, default=10)
    # parser.add_argument('--seed', type=int, default=42)
    return parser
