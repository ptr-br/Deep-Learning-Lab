import gin
import logging
import sys
import wandb
from absl import app, flags
import os
from train import Trainer
from input_pipeline import datasets
from utils import utils_params, utils_misc
from input_pipeline.make_tfrecords import create_tfrecords
from models.model import rnn
from evaluation.eval import Evaluator
from models.tcn_model import model_tcn

FLAGS = flags.FLAGS
flags.DEFINE_string(
    'model_name', 'gru', 'Specify to use which model (default: gru)')
flags.DEFINE_boolean(
    'wandb', False, "Specify if the run should be logged and sent to weights & biases")
flags.DEFINE_string(
    'run', 'all', 'Specify whether to train or evaluate a model (default: all -> do both)')
flags.DEFINE_boolean(
    'tf_records', False, 'Specify if script should be exited after tf records creation.')
flags.DEFINE_string(
    "model_id", "", "Specify path for evaluaion of a specific model")
flags.DEFINE_string(
    "initialization", "glorot_uniform", "Specify path for evaluaion of a specific model")
flags.DEFINE_boolean(
    "bidirectional", "False", "Specify path for evaluaion of a specific model")
flags.DEFINE_integer(
    "lowpass_filter", "-1", "Specify cutoff frequency to lowpassfilter the dataset")
flags.DEFINE_integer(
    "loss_weight", "-1", "Specify loss_weight"
)

def setup():
    """
    Setup folder structure, loggers and run_paths.

    Returns:
        [dict]: folders to store data from current run
    """
    # generate folder structures
    run_paths = utils_params.gen_run_folder(FLAGS.model_id)

    # set loggers
    utils_misc.set_loggers(run_paths['path_logs_train'], logging.INFO)

    # gin-config
    gin.parse_config_files_and_bindings(['configs/config.gin'], [])
    utils_params.save_config(run_paths['path_gin'], gin.config_str())
    
    return run_paths

def check_for_tfrecords(fc=False, stop=False):
    """
    Check if tfrecords exist, otherwise create it.

    Args:
        fc (bool, optional): cutoff frequency. Defaults to False.
        stop (bool, optional): stop after creation of files. Defaults to False.
    """
    
    # create tf-records folder and files if they do not exist yet
    if create_tfrecords(fc=fc):
        logging.info(
            f"Created TFRecords files at: {gin.query_parameter('create_tfrecords.records_dir')}")
    else:
        logging.info(
            "TFRecords files already exist. Proceed with the execution")

    if stop:
        logging.info("Exiting script since no training wanted!")
        logging.info(
            "Change FLAGS.tf_records in main.py to False to enable training...")
        sys.exit()
    
def main(argv):

    run_paths = setup()

    # init weights & biases
    if FLAGS.wandb:
        wandb.init(project="Human_Activity_Recognition", name=run_paths['path_model_id'].split(os.sep)[-1],
                   config=utils_params.gin_config_to_readable_dictionary(gin.config._CONFIG))

    check_for_tfrecords(fc=FLAGS.lowpass_filter, stop=FLAGS.tf_records)

    # setup pipeline
    window_length, window_shift = gin.query_parameter('create_tfrecords.window_length_and_shift')
   
    ds_train, ds_val, ds_test = datasets.load(
        data_dir=gin.query_parameter('create_tfrecords.records_dir'),
        window_length=window_length,
        window_shift=window_shift,
        fc=FLAGS.lowpass_filter
    )

    if FLAGS.model_name == 'lstm':
        model = rnn(rnn_type ="lstm", window_length=window_length,
                    kernel_initializer=FLAGS.initialization, bi_direction=FLAGS.bidirectional)
    if FLAGS.model_name == 'gru':
        model = rnn(rnn_type ="gru", window_length=window_length,
                    kernel_initializer=FLAGS.initialization, bi_direction=FLAGS.bidirectional)
    if FLAGS.model_name == 'rnn':
        model = rnn(rnn_type ="rnn", window_length=window_length,
                    kernel_initializer=FLAGS.initialization, bi_direction=FLAGS.bidirectional)
    if FLAGS.model_name == 'tcn':
        model = model_tcn(window_length)


    if FLAGS.run.lower() == 'train':
        
        model.summary()

        trainer = Trainer(model, ds_train, ds_val, run_paths, lr_rate=1e-2,
                          wandb=FLAGS.wandb, loss_weight=FLAGS.loss_weight)
        for _ in trainer.train():
            continue

    elif FLAGS.run.lower() == 'eval':

        utils_misc.set_loggers(run_paths['path_logs_eval'], logging.INFO)
        evaluator = Evaluator(model, ds_test, ds_val, run_paths)
        evaluator.evaluate()
        evaluator.vis_file(gin.query_parameter('create_tfrecords.data_dir'), fc=FLAGS.lowpass_filter)


    elif FLAGS.run.lower() == 'all':
    
        model.summary()
        
        # Training
        trainer = Trainer(model, ds_train, ds_val, run_paths, lr_rate=1e-2, wandb=FLAGS.wandb)
        for _ in trainer.train():
            continue

        # Evaluate
        utils_misc.set_loggers(run_paths['path_logs_eval'], logging.INFO, del_prev_handler=True)
        evaluator = Evaluator(model, ds_test, ds_val, run_paths)
        evaluator.evaluate()
        evaluator.vis_file(gin.query_parameter('create_tfrecords.data_dir'), fc=FLAGS.lowpass_filter)


if __name__ == '__main__':
    app.run(main)
