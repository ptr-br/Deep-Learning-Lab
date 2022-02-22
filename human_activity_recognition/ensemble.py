import tensorflow as tf
from evaluation.metrics import ConfusionMatrix
import gin
import logging

from absl import app
from train import Trainer
from evaluation.eval import Evaluator
from input_pipeline import datasets
from utils import utils_params, utils_misc
from models.model import rnn
from models.tcn_model import model_tcn
from input_pipeline import make_tfrecords


def avg_ensemble(model_list, ds_test, n_classes, run_paths):
    """

    Args:
        model_list: a list of restored models
        ds_test: test dataset
        n_classes: number of classes, same as definition in the model
        run_paths: run_path of these file

    Returns: logging infos about ensemble result, and save the cm_plot png

    """


    logging.info(f'======== Starting Ensemble Evaluation ========')
    cm = ConfusionMatrix(n_classes=n_classes)
    template_acc = 'unbalanced_acc: {:.2f}%, balanced_acc: {:.2f}%'
    template_total = 'precision: {:.2f}%, recall: {:.2f}%, F1-score: {:.2f}%'

    for features, labels in ds_test:
        predictions = [model(features, training=False) for model in model_list]
        y_pred = tf.squeeze(tf.reduce_mean(tf.convert_to_tensor(predictions), axis=0))
        cm.update_state(labels, y_pred)

    total_acc = cm.acc()
    logging.info(
        "Total acc 12 class problem: " + template_acc.format(total_acc["unbalanced_acc"] * 100,
                                                             total_acc["balanced_acc"] * 100)
    )
    logging.info(
        "Other total metrics: " + template_total.format(total_acc["precision"] * 100,
                                                        total_acc["recall"] * 100,
                                                        total_acc["f1_score"] * 100)
    )

    logging.info('Confusion matrix:\n{}'.format(cm.result().numpy()))

    logging.info('======== Finished Ensemble Evaluation ========')


def main(argv):
    run_paths = utils_params.gen_run_folder("ensemble")
    utils_misc.set_loggers(run_paths['path_logs_train'], logging.DEBUG)

    # gin-config
    gin.parse_config_files_and_bindings(['configs/config.gin'], [])
    utils_params.save_config(run_paths['path_gin'], gin.config_str())
    
    # setup pipeline
    window_length, window_shift = gin.query_parameter('create_tfrecords.window_length_and_shift')
    ds_train, ds_val, ds_test = datasets.load(
        data_dir=gin.query_parameter('create_tfrecords.records_dir'),
        window_length=window_length,
        window_shift=window_shift,
    )



    ckpt = {"lstm": "./best_runs/lstm/ckpts",
            "gru": "./best_runs/gru_glorot_uniform/ckpts",
            "tcn": "./best_runs/tcn/ckpts"}

    lstm = rnn(rnn_type="lstm", dense_units=179, num_dense=2, num_rnn=4, rnn_units=31, window_length=window_length)
    checkpoint_1 = tf.train.Checkpoint(model=lstm)
    checkpoint_1.restore(tf.train.latest_checkpoint(ckpt["lstm"])).expect_partial()

    gru = rnn(rnn_type="gru", window_length=window_length)
    checkpoint_2 = tf.train.Checkpoint(model=gru)
    checkpoint_2.restore(tf.train.latest_checkpoint(ckpt["gru"])).expect_partial()

    tcn = model_tcn(window_length)
    checkpoint_3 = tf.train.Checkpoint(model=tcn)
    checkpoint_3.restore(tf.train.latest_checkpoint(ckpt["tcn"])).expect_partial()
    model_list = [lstm, gru, tcn]
    avg_ensemble(model_list, ds_test, n_classes=12, run_paths=run_paths)


if __name__ == "__main__":
    app.run(main)


