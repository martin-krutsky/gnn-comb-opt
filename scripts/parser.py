import argparse


def get_parser():
    parser = argparse.ArgumentParser()
    # Optimisation params
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--weight_decay', type=float, default=0.0005)
    parser.add_argument('--early_stopping', type=int, default=100)
    # Model configuration
    parser.add_argument('--layers', type=int, default=2)
    parser.add_argument('--hidden_channels', type=int, default=20)
    parser.add_argument('--dropout', type=float, default=0.0)

    # Experiment parameters
    parser.add_argument('--dataset', default='texas')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--cuda', type=int, default=0)
    parser.add_argument('--folds', type=int, default=10)
    parser.add_argument('--gcn', type=str, choices=[
        'GCNConv', 'ChebConv', 'SAGEConv', 'GraphConv', 'GatedGraphConv', 'ResGatedGraphConv', 'GATConv', 'GATv2Conv',
        'TransformerConv', 'AGNNConv', 'TAGConv', 'GINConv', 'GINEConv', 'ARMAConv', 'SGConv', 'APPNP', 'MFConv',
        'RGCNConv', 'DNAConv', 'GMMConv', 'SplineConv', 'NNConv', 'CGConv', 'EdgeConv', 'FeaStConv', 'LEConv',
        'PNAConv', 'ClusterGCNConv', 'GENConv', 'GCN2Conv', 'PANConv', 'WLConv', 'FiLMConv', 'SuperGATConv', 'FAConv',
        'EGConv', 'PDNConv', 'GeneralConv'
    ], default='GCNConv')
    return parser
