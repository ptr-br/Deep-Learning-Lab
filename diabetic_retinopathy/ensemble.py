import tensorflow as tf
import tensorflow.keras as keras
from evaluation.metrics import ConfusionMatrix
import gin
import logging

from absl import app
from train import Trainer
from evaluation.eval import Evaluator
from input_pipeline import datasets
from utils import utils_params, utils_misc
from models.vit import vit
from models.architectures import team20_cnn_02, vgg_like, team20_cnn_01, cnn_blueprint, inception_blueprint, resnet
from input_pipeline import make_tfrecords
from deepvis.visualization import visual

ENSEMBLE_METHOD = "voting"
# ENSEMBLE_METHOD = "avg"


def get_stack_ensamble_model(model_list, n_classes=2):
    inputs = keras.Input(shape=(256, 256, 3))
    out = []
    for model in model_list:
        model.trainable = False
        out.append(model(inputs, training=False))

    out = tf.concat(out, axis=-1)
    outputs = keras.layers.Dense(n_classes)(out)
    return keras.Model(inputs, outputs)



def ensamble(model_list, ds_val, ds_test, ensamble_type="voting"):
    logging.info(f'======== Starting {ensamble_type} Ensamble Evaluation ========')
    cm = ConfusionMatrix()
    for name, ds in [('val', ds_val), ('test', ds_test)]:
        if ensamble_type == "voting":
            for images, labels in ds:
                predictions = [model(images, training=False) for model in model_list]
                y_labels = [tf.argmax(pred, axis=-1) for pred in predictions]
                labels_sum = tf.reduce_sum(tf.convert_to_tensor(y_labels), axis=0)
                boundary = len(model_list) // 2
                voted_label = tf.where(labels_sum > boundary, 1, 0)
                cm.update_state(labels, voted_label)

        if ensamble_type == "avg":
            for images, labels in ds:
                predictions = [model(images, training=False) for model in model_list]
                predictions = tf.squeeze(tf.reduce_mean(tf.convert_to_tensor(predictions), axis=0))
                y_pred = tf.argmax(predictions, axis=1)
                cm.update_state(labels, y_pred)

        cm_result = cm.result()

        template = '\n' \
                   f'Evaluating on {name} dataset\n' + \
                   'Unbalanced_Acc:       {}\n' + \
                   'Balanced_Acc:         {}\n' + \
                   'Sensitivity:          {}\n' + \
                   'Specificity:          {}\n' + \
                   'F1_Score:             {}\n' + \
                   'Confusion_matrix:   \n' + \
                   '                   {}\n' + \
                   '                   {}\n' + \
                   '\n'

        prob = cm.probability()

        logging.info(
            template.format(prob['unbalanced_acc'] * 100, prob['balanced_acc'] * 100,
                            prob['sensitivity'] * 100, prob['specificity'] * 100, prob['f1_score'] * 100,
                            cm_result.numpy()[0], cm_result.numpy()[1]))
        cm.reset_states()

    logging.info('======== Finished Ensamble Evaluation ========')

def main(argv):
        run_paths = utils_params.gen_run_folder("ensamble")
        utils_misc.set_loggers(run_paths['path_logs_train'], logging.DEBUG)

        # gin-config
        gin.parse_config_files_and_bindings(['configs/config.gin'], [])
        utils_params.save_config(run_paths['path_gin'], gin.config_str())
        ds_train, ds_val, ds_test, ds_info = datasets.load(data_dir=gin.query_parameter('create_tfrecords.records_dir'))


        ckpt = {"resnet": "./best_runs/resnet/ckpts",
                "vgg": "./best_runs/vgg_like/ckpts",
                "cnn01": "./best_runs/team20_cnn_01/ckpts"}

        cnn = team20_cnn_01()
        checkpoint_1 = tf.train.Checkpoint(model=cnn)
        checkpoint_1.restore(tf.train.latest_checkpoint(ckpt['cnn01'])).expect_partial()

        vgg_trained = vgg_like()
        checkpoint_2 = tf.train.Checkpoint(model=vgg_trained)
        checkpoint_2.restore(tf.train.latest_checkpoint(ckpt['vgg'])).expect_partial()

        resnet_trained = resnet()
        checkpoint_3 = tf.train.Checkpoint(model=resnet_trained)
        checkpoint_3.restore(tf.train.latest_checkpoint(ckpt['resnet'])).expect_partial()
        model_list = [cnn, vgg_trained, resnet_trained]
        ensamble(model_list, ds_val, ds_test, ENSEMBLE_METHOD)

if __name__ == "__main__":
    app.run(main)
