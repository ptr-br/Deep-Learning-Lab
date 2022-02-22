import logging
import gin
import wandb
import sys

from input_pipeline.datasets import load

from models.architectures import vgg_like, cnn_blueprint, inception_blueprint, team20_cnn_01, transfer_model, resnet

from train import Trainer
from utils import utils_params, utils_misc
from input_pipeline import make_tfrecords
from deepvis.visualization import visual
from evaluation.eval import Evaluator

import os



def train_func():
    
    with wandb.init() as run:     
        
        gin.clear_config()

        # Hyperparameters
        bindings = []
        for key, value in run.config.items():
            bindings.append(f'{key}={value}')
            
        # generate folder structures
        run_paths = utils_params.gen_run_folder()

        # set loggers
        utils_misc.set_loggers(run_paths['path_logs_train'], logging.INFO)

        # gin-config
        # change path to absolute path of config file
        gin.parse_config_files_and_bindings(['configs/config.gin'], bindings)
        utils_params.save_config(run_paths['path_gin'], gin.config_str())
        
        run.name = run_paths['path_model_id'].split(os.sep)[-1]

        # setup pipeline
        ds_train, ds_val, ds_test, ds_info = load(data_dir=gin.query_parameter('create_tfrecords.records_dir'))


        # model 
        # model = resnet()
        model = team20_cnn_01()
        # model = cnn_blueprint(input_shape=(256, 256, 3), n_classes=2)
        # model = inception_blueprint(input_shape=(256, 256, 3), n_classes=2)
        # model = transfer_model(basemodel_name='Xception')
        
        model.summary()

        # Train
        trainer = Trainer(model, ds_train, ds_val, ds_info, run_paths, wandb=True)
        for _ in trainer.train():
            continue

         # Evaluate
        utils_misc.set_loggers(run_paths['path_logs_eval'], logging.INFO, del_prev_handler=True)
        evaluator = Evaluator(model, ds_test, ds_val, run_paths)
        evaluator.evaluate()
        
        # Visualize 
        visual(model, run_paths, images_folder=gin.query_parameter('create_tfrecords.data_dir'))
        

train_func()
