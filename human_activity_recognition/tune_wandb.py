import logging
import gin
import wandb
import sys
import argparse
import os
from input_pipeline.datasets import load
from models.model import rnn
from models.tcn_model import model_tcn
from train import Trainer
from utils import utils_params, utils_misc
from evaluation.eval import Evaluator
from input_pipeline import make_tfrecords
from configs.sweep_configs import sweep_params_configuration, sweep_loss_configuration, sweep_tcn
from main import check_for_tfrecords

argparser = argparse.ArgumentParser(description='Specify type of sweep')
argparser.add_argument('-m', '--mode', choices=['params', 'loss', "tcn"],
                    default='params',
                    help='Type of sweep to execute (params or loss)')

arg = argparser.parse_args()




def train_func():
    
    # search for best balanced acc instead of total acc
    if arg.mode == 'loss':
        optimze_balanced_acc = True
    else:
        optimze_balanced_acc = False
        
    with wandb.init() as run:

        gin.clear_config()

        # Hyperparameters
        bindings = []
        for key, value in run.config.items():
            bindings.append(f'{key}={value}')

        # generate folder structures
        run_paths = utils_params.gen_run_folder()
        run.name = run_paths['path_model_id'].split(os.sep)[-1]

        # set loggers
        utils_misc.set_loggers(run_paths['path_logs_train'], logging.INFO)

        # gin-config
        # change path to absolute path of config file
        gin.parse_config_files_and_bindings(['configs/config.gin'], bindings)
        utils_params.save_config(run_paths['path_gin'], gin.config_str())
        
        check_for_tfrecords()

        # setup pipeline
        window_length, window_shift = gin.query_parameter('create_tfrecords.window_length_and_shift')
        ds_train, ds_val, ds_test = load(
            data_dir=gin.query_parameter('create_tfrecords.records_dir'),
            window_length=window_length,
            window_shift=window_shift)

        # model
        if arg.mode == "tcn":
            model = model_tcn(window_length)
        else:
            model = rnn(window_length=window_length)
        model.summary()


        trainer = Trainer(model, ds_train, ds_val, run_paths,
                          lr_rate=1e-2, wandb=True)
        for _ in trainer.train(optimze_balanced_acc=optimze_balanced_acc):
            continue

        # Evaluate
        utils_misc.set_loggers(
            run_paths['path_logs_eval'], logging.INFO, del_prev_handler=True)
        evaluator = Evaluator(model, ds_test, ds_val, run_paths)
        evaluator.evaluate()
        # Do not vis on server sweep since matplotlib is not thread safe....
        #evaluator.vis_file(gin.query_parameter('create_tfrecords.data_dir'))

        # reset logger
        utils_misc.remove_handler()


# run the sweep
if arg.mode == 'params':
    sweep_id = wandb.sweep(sweep_params_configuration)
    wandb.agent(sweep_id, function=train_func, count=20)

elif arg.mode == 'loss': 
    sweep_id = wandb.sweep(sweep_loss_configuration)
    wandb.agent(sweep_id, function=train_func, count=20)

elif arg.mode == 'tcn':
    sweep_id = wandb.sweep(sweep_tcn)
    wandb.agent(sweep_id, function=train_func, count=20)
