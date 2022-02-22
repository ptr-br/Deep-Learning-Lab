from genericpath import exists
import shutil
import gin
import logging
import os
import numpy as np
from absl import app, flags
from train import Trainer
import matplotlib.pyplot as plt
from input_pipeline import datasets
from models.model import rnn
from evaluation.eval import Evaluator
from models.tcn_model import model_tcn
from input_pipeline.make_tfrecords import create_tfrecords
from main import setup


FLAGS = flags.FLAGS
flags.DEFINE_integer("start", "5", "Specify the shortest window length (default: 25)")
flags.DEFINE_integer("end", "305", "Specify the longest window length (default: 300)")
flags.DEFINE_integer("step", "20", "Specify the longest window length (default: 5)")
flags.DEFINE_integer(
    "num", "5", "Number of runs that should be used to average the accuracy of one specification (defautl: 5)"
)


def main(argv):
    values = []
    windows = []
    run_paths = setup()
    for i in range(FLAGS.start, FLAGS.end + FLAGS.step, FLAGS.step):

        window_length = i
        if i <= 100:
            window_shift = window_length
        else:
            window_shift = int(window_length / 2)

        if create_tfrecords(fc=FLAGS.lowpass_filter, window_length_and_shift=(window_length, window_shift)):
            logging.info(f"Created TFRecords files at: {gin.query_parameter('create_tfrecords.records_dir')}")
        else:
            logging.info("TFRecords files already exist. Proceed with the execution")

        # setup pipeline
        data_dir = gin.query_parameter("create_tfrecords.records_dir")
        ds_train, ds_val, ds_test = datasets.load(
            data_dir=data_dir,
            window_length=window_length,
            window_shift=window_shift,
            fc=FLAGS.lowpass_filter,
        )

        # run model 5 times and average the results
        scores = []
        for i in range(FLAGS.num):

            if FLAGS.model_name == "lstm":
                model = rnn(
                    rnn_type="lstm",
                    window_length=window_length,
                    kernel_initializer=FLAGS.initialization,
                    bi_direction=FLAGS.bidirectional,
                )
            if FLAGS.model_name == "gru":
                model = rnn(
                    rnn_type="gru",
                    window_length=window_length,
                    kernel_initializer=FLAGS.initialization,
                    bi_direction=FLAGS.bidirectional,
                )
            if FLAGS.model_name == "rnn":
                model = rnn(
                    rnn_type="rnn",
                    window_length=window_length,
                    kernel_initializer=FLAGS.initialization,
                    bi_direction=FLAGS.bidirectional,
                )
            if FLAGS.model_name == "tcn":
                model = model_tcn(window_length)

            logging.info(f"Starting 5 runs for how_small.py plots with window_length:{window_length}")

            model.summary()

            # train
            trainer = Trainer(model, ds_train, ds_val, run_paths, lr_rate=1e-2, wandb=FLAGS.wandb)
            for _ in trainer.train():
                continue

            # eval
            evaluator = Evaluator(model, ds_test, ds_val, run_paths)
            total_acc = evaluator.evaluate(yield_acc=True)

            scores.append(total_acc["unbalanced_acc"])

        scores = np.array(scores)

        values.append(scores.mean())
        windows.append(window_length)
    
        # delete records dir to not have to many stored files
        if FLAGS.lowpass_filter > 0 and FLAGS.lowpass_filter < 25:
            records_dir = data_dir + f"wl{str(window_length)}_ws{str(window_shift)}_fc{str(FLAGS.lowpass_filter)}"
        else:
            records_dir = data_dir + f"wl{str(window_length)}_ws{str(window_shift)}"
        try:
            shutil.rmtree(records_dir)
        except OSError as e:
            logging.info(f"Could not delete folder at: {records_dir} ")
        

    # plot results
    fig = plt.figure(figsize=(5, 5))

    plt.plot(windows, values)

    plt.title("How small can the window size get?")
    plt.xlabel("Window size")
    plt.ylabel("Accuracy")

    if FLAGS.lowpass_filter > 0 and FLAGS.lowpass_filter < 25:
        template = f"start{FLAGS.start}_end{FLAGS.end}_step{FLAGS.step}_fc{FLAGS.lowpass_filter}/"
        folder = "./how_small/" + template
    else:
        folder = "./how_small/" + f"start{FLAGS.start}_end{FLAGS.end}_step{FLAGS.step}/"

    if not os.path.isdir(folder):
        os.mkdir(folder)

    # save figure
    plot_path = folder + "figure.png"
    logging.info(f"Save figure of how_small.py to {plot_path}")
    plt.savefig(plot_path)

    # save values
    np.savetxt(folder + "values.csv", values, delimiter=",")
    np.savetxt(folder + "windows.csv", windows, delimiter=",")


if __name__ == "__main__":
    app.run(main)
