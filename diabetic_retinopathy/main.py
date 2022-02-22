import gin
import logging
import sys
import wandb
from absl import app, flags
import os

from train import Trainer
from evaluation.eval import Evaluator
from input_pipeline import datasets
from utils import utils_params, utils_misc
from models.architectures import team20_cnn_02, vgg_like, team20_cnn_01, cnn_blueprint, inception_blueprint, resnet, transfer_model
from input_pipeline import make_tfrecords
from deepvis.visualization import visual




FLAGS = flags.FLAGS
flags.DEFINE_string(
    'run', 'all', 'Specify whether to train, evaluate or visualize a model')
flags.DEFINE_boolean(
    'tf_records', False, 'Specify if script should be exited after tf records creation.')
flags.DEFINE_string('model_name', 'cnn_01', 'choose model')
flags.DEFINE_string(
    "model_id", "", "Specify path for evaluaion of a specific model (make sure model_name matches)")
flags.DEFINE_boolean('wandb', False, "Specify if the run should be logged and sent to weights & biases")

# tf.config.run_functions_eagerly(True) 


def main(argv):

    # generate folder structures
    run_paths = utils_params.gen_run_folder(FLAGS.model_id)

    # gin-config
    gin.parse_config_files_and_bindings(['configs/config.gin'], [])
    utils_params.save_config(run_paths['path_gin'], gin.config_str())

    # init weights & biases
    if FLAGS.wandb:
        wandb.init(project="Diabetic_Retinopathy", name=run_paths['path_model_id'].split(os.sep)[-1],
        config=utils_params.gin_config_to_readable_dictionary(gin.config._CONFIG))
    
    # create tf-records folder and files if they do not exist yet
    if make_tfrecords.create_tfrecords():
        logging.info("Created TFRecords files at path specified in gin file")
    else:
        logging.info("TFRecords files already exist. Proceed with the execution")

    if FLAGS.tf_records:
        logging.info("Exiting script since no trinaing wanted!")
        logging.info("Change FLAGS.tf_records in main.py to False to enable training...")
        sys.exit()

    # setup pipeline
    ds_train, ds_val, ds_test, ds_info = datasets.load(data_dir=gin.query_parameter('create_tfrecords.records_dir'))

    if FLAGS.model_name == 'cnn_01':
            model = team20_cnn_01()

    elif FLAGS.model_name == 'cnn_02':
        model = team20_cnn_02()
        
    elif FLAGS.model_name == 'cnn_blueprint':
        model = cnn_blueprint()

    elif FLAGS.model_name == 'vgg':
        model = vgg_like()
        
    # elif FLAGS.model_name == "inception":
    #   model = inception_blueprint()
        
    elif FLAGS.model_name == 'resnet':
        model = resnet()
        
    elif FLAGS.model_name.lower() == 'effinet':
        model = transfer_model(basemodel_name='EfficientNetB0')

    elif FLAGS.model_name.lower() == 'inceptionresnet':
        model = transfer_model(basemodel_name='InceptionResNet')

    elif FLAGS.model_name.lower() == 'xception':
        model = transfer_model(basemodel_name='Xception')
    else:
        logging.error('Please specify correct model name! Exiting now, Bye.')
        sys.exit()

    if FLAGS.run.lower() == 'train':

        utils_misc.set_loggers(run_paths['path_logs_train'], logging.INFO)
        logging.info(f'Training model: {FLAGS.model_name}')
        model.summary()
        
        trainer = Trainer(model, ds_train, ds_val, ds_info, run_paths, lr_rate=1e-2, wandb=FLAGS.wandb)
        for _ in trainer.train():
            continue
    
    elif FLAGS.run.lower() == 'eval':   
        
        utils_misc.set_loggers(run_paths['path_logs_eval'], logging.INFO)
        logging.info(f'Evaluating model: {FLAGS.model_name}')
        evaluator = Evaluator(model,ds_test, ds_val, run_paths)
        evaluator.evaluate()
        
    elif FLAGS.run.lower() == 'visual':
        
        utils_misc.set_loggers(run_paths['path_logs_eval'], logging.INFO)
        logging.info(f'Visualizing model: {FLAGS.model_name}')
        visual(model, run_paths, images_folder=gin.query_parameter('create_tfrecords.data_dir'))
        
    elif FLAGS.run.lower() ==  'all':
 
        # Training
        utils_misc.set_loggers(run_paths['path_logs_train'], logging.INFO)
        logging.info(f'Running model: {FLAGS.model_name}')
        model.summary()
        
        trainer = Trainer(model, ds_train, ds_val, ds_info, run_paths, lr_rate=1e-2, wandb=FLAGS.wandb)
        for _ in trainer.train():
            continue
        
        # Evaluate
        utils_misc.set_loggers(run_paths['path_logs_eval'], logging.INFO, del_prev_handler=True)
        evaluator = Evaluator(model,ds_test, ds_val, run_paths)
        evaluator.evaluate()
        
        # Visualize 
        visual(model, run_paths, images_folder=gin.query_parameter('create_tfrecords.data_dir'))
        

if __name__ == "__main__":
    app.run(main)
