import tensorflow as tf
import logging


from evaluation.metrics import ConfusionMatrix
from input_pipeline.preprocessing import random_brightness, random_flip_up_down, random_flip_left_right, random_rotate

class Evaluator():

    def __init__(self, model, ds_test, ds_val, run_paths, ensamble=False):

        self.model = model
        self.run_paths = run_paths
        self.ds_test = ds_test
        self.ds_val = ds_val
        self.ensamble = ensamble
        self.eval_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(
            name='train_accuracy')

    def evaluate(self):
        
        logging.info('\n======== Starting Evaluation ========')

        checkpoint = tf.train.Checkpoint(model=self.model)
        checkpoint.restore(tf.train.latest_checkpoint(
            self.run_paths['path_ckpts_train'])).expect_partial()
        logging.info(f'Checkpoint restored from: {self.run_paths["path_ckpts_train"]}')

        cm = ConfusionMatrix()

        for name, ds in [('val',self.ds_val), ('test',self.ds_test)]:
        
            if self.ensamble == True:
                for images, labels in ds:
                    predictions = self.multi_predict(images)
                    y_pred = tf.argmax(predictions, axis=1)
                    cm.update_state(labels, y_pred)
            else:
                for images, labels in ds:
                    predictions = self.model(images, training=False)
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
                                prob['sensitivity'] * 100, prob['specificity'] * 100, prob['f1_score'] * 100, cm_result.numpy()[0], cm_result.numpy()[1]))
            
            cm.reset_states()
        
        logging.info('======== Finished Evaluation ========')

    def multi_predict(self, images):
        predictions = []
        for i in [images, random_brightness(images), random_flip_left_right(images), random_flip_up_down(images), random_rotate(images)]:
            i = tf.clip_by_value(i, 0, 1)
            predictions.append(tf.nn.softmax(self.model(i, training=False)))
        pred = tf.convert_to_tensor(predictions)
        avg_pred = tf.reduce_mean(pred, axis=0)
        return avg_pred
