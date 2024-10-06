from ray import tune


gcn_hyperparams_mapper = {
    'GCNConv': {
        "improved": tune.choice([True, False]),
        "normalize": tune.choice([True, False]),
        "add_self_loops": tune.choice([True, False]),
    },
    'SAGEConv': {
        "aggr": tune.choice(['mean', 'max', 'lstm']),
        "normalize": tune.choice([True, False]),
        "project": tune.choice([True, False]),
    },
    'GraphConv': {
        "aggr": tune.choice(['add', 'mean', 'max']),
    },
    'GATv2Conv': {
        "heads": tune.choice([1, 2, 4]),
        "concat": False,
        "add_self_loops": tune.choice([True, False]),
        "share_weights": tune.choice([True, False]),
        "residual": tune.choice([True, False]),
    },
    'TransformerConv': {
        "heads": tune.choice([1, 2, 4]),
        "concat": False,
    },
    'GENConv': {
        "aggr": tune.choice(['softmax', 'powermean', 'add', 'mean', 'max']),
        "learn_t": tune.choice([True, False]),
        "learn_p": tune.choice([True, False]),
        "num_layers": tune.choice([2, 4, 8]),
    },
    'GeneralConv': {
        "aggr": tune.choice(['add', 'mean', 'max']),
        "skip_linear": tune.choice([True, False]),
        "heads": tune.choice([1, 2, 4]),
        "attention": tune.choice([True, False]),
    }
}

hyperparams_config = {
    "lr": tune.loguniform(0.00005, 0.01),
    "weight_decay": tune.choice([0.0, 1e-5, 1e-4, 1e-3]),
    "embedding_size": tune.choice([5, 10, 20, 50, 100, 200]),
    "n_layers": tune.choice([2, 3, 4]),
    "hidden_channels": tune.choice([5, 10, 20, 50, 100, 200]),
    "dropout": tune.choice([0.0, 0.1, 0.2, 0.4, 0.3, 0.5]),
    "gcn_layer": tune.choice([{"layer_name": key, "hyperparams": values} for key, values in gcn_hyperparams_mapper.items()]),
}
