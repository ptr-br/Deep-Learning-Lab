
sweep_params_configuration = {
    "name": "har-gru-sweep-server01",
    "metric": {"name": "max_acc", "goal": "maximize"},
    "method": "bayes",
    "parameters": {
        "rnn.rnn_units ": {
            'distribution': 'int_uniform',
            'min': 8,
            'max': 64
        },
        # "rnn.rnn_type ": {
        #     'values': ['lstm', 'gru', 'rnn'],
        # },
        "rnn.num_rnn ": {
            'distribution': 'int_uniform',
            'min': 1,
            'max': 10
        },
        "rnn.dense_units ": {
            'distribution': 'int_uniform',
            'min': 32,
            'max': 256
        },
        "rnn.num_dense ": {
            'distribution': 'int_uniform',
            'min': 1,
            'max': 10
        },
        "rnn.dropout_rnn ": {
            'distribution': 'uniform',
            'min': 0,
            'max': 0.5
        },
        "rnn.dropout_dense ": {
            'distribution': 'uniform',
            'min': 0,
            'max': 0.5
        },
        "create_tfrecords.window_length_and_shift":{
            'values': [(250,125),(250,75),
                        (500,100),(500,250),
                        (100,50), (100,75)]
            }
    }
}


sweep_loss_configuration = {
            "name": "loss-optimization-sweep-01",
            "metric": {"name": "max_acc_balanced", "goal": "maximize"},
             "method": "bayes",
    "parameters": {
        'Trainer.loss_weight':{
            'distribution': 'uniform',
            'min': 1,
            'max': 20
        }
    }
}

sweep_tcn= {
    "name": "har-tcn-sweep-server01",
    "metric": {"name": "max_acc", "goal": "maximize"},
    "method": "bayes",
    "parameters": {
        "model_tcn.nb_filters ": {
            'distribution': 'int_uniform',
            'min': 32,
            'max': 64
        },
        "model_tcn.kernel_size ": {
            'distribution': 'int_uniform',
            'min': 2,
            'max': 4
        },
        "model_tcn.nb_stacks ": {
            'distribution': 'int_uniform',
            'min': 1,
            'max': 2
        },
        "model_tcn.dropout_rate ": {
            'distribution': 'uniform',
            'min': 0,
            'max': 0.15
        },
        "create_tfrecords.window_length_and_shift":{
            'values': [(250,125),(250,75),
                        (500,100),(500,250),
                        (100,50), (100,75)]
            }
    }
}
