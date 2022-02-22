import gin
import tensorflow as tf
import logging
import datetime
import os
import wandb
import sys
from evaluation import metrics

@gin.configurable
class Trainer(object):

    def __init__(self, model, ds_train, ds_val, run_paths,
                 total_steps, log_interval, ckpt_interval,
                 lr_rate=0.01, wandb=False, ckpt=False, loss_weight=1, acc_weight=1):

        logging.info(f'All relevant data from this run is stored in {run_paths["path_model_id"]}')

        # Init summary Writer
        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        current_dir = os.path.dirname(__file__)
        tensorboard_log_dir = os.path.join(current_dir, 'logs')
        log_dir = os.path.join(tensorboard_log_dir, current_time)
        logging.info(f"Tensorboard output will be stored in: {log_dir}")
        self.train_log_dir = os.path.join(log_dir, 'train')
        self.test_log_dir = os.path.join(log_dir, 'validation')
        self.train_summary_writer = tf.summary.create_file_writer(self.train_log_dir)
        self.test_summary_writer = tf.summary.create_file_writer(self.test_log_dir)

        # Loss objective
        self.loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)
        # self.optimizer = GCAdam(lr_rate)
        lr_scheduler = tf.keras.optimizers.schedules.CosineDecay(initial_learning_rate=lr_rate,
                                                                 decay_steps=1000,
                                                                 alpha=0.1)
        self.optimizer = tf.keras.optimizers.Adam(lr_scheduler)
        # Metrics
        self.train_loss = tf.keras.metrics.Mean(name='train_loss')
        self.train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

        self.test_loss = tf.keras.metrics.Mean(name='test_loss')
        self.test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')

        self.model = model
        self.ds_train = ds_train
        self.ds_val = ds_val
        self.run_paths = run_paths
        self.total_steps = total_steps
        self.log_interval = log_interval
        self.ckpt_interval = ckpt_interval
        self.loss_weight = loss_weight
        self.acc_weight = acc_weight
        self.max_acc = 0.0
        self.max_acc_balanced = 0.0
        self.test_cm = metrics.ConfusionMatrix(n_classes=self.model.output.shape[1])

        # Checkpoint Manager
        self.ckpt = tf.train.Checkpoint(model=self.model)
        # if path is given load old model form checkpoint path
        if ckpt:
            self.manager = tf.train.CheckpointManager(self.ckpt, ckpt, max_to_keep=4)
            if self.manager.latest_checkpoint:
                try:
                    self.ckpt.restore(self.manager.latest_checkpoint)
                    logging.info(f"Restored checkpoint from {self.manager.latest_checkpoint}")
                    logging.info("Training model from checkpoint ...")
                except Exception as e:
                    logging.error(str(e))
                    logging.error('Your defined model and the parameters that should be loaded do not fit. ' \
                                  'Please make sure that the checkpoint is from the same architcture as the model you want to train further!')
                    sys.exit(1)
            else:
                logging.error(
                    f'Could not find any checkpoints at path specified! Please make sure chekpoints exist at {ckpt}.')
        else:
            self.manager = tf.train.CheckpointManager(self.ckpt, run_paths["path_ckpts_train"], max_to_keep=3)
            logging.info("Training will be from scratch since no valid checkpoint was specified.")
            logging.info(f"All checkpoints will be stored in: {run_paths['path_ckpts_train']}")

        # Turn on weights and biases logging if wanted
        self.wandb = wandb

    @tf.function
    def train_step(self, features, labels):
        # give the samples of class>5 a different weight
        loss_weight_vector = tf.squeeze(tf.where(labels > 5, self.loss_weight, 1))
        acc_weight_vector = tf.squeeze(tf.where(labels > 5, self.acc_weight, 1))
        with tf.GradientTape() as tape:
            # training=True is only needed if there are layers with different
            # behavior during training versus inference (e.g. Dropout).
            predictions = self.model(features, training=True)
            loss = self.loss_object(labels, predictions, sample_weight=loss_weight_vector)
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(
            zip(gradients, self.model.trainable_variables))
        self.train_loss(loss)
        self.train_accuracy.update_state(labels, predictions, sample_weight=acc_weight_vector)

    @tf.function
    def test_step(self, features, labels):
        # give the samples of class>5 a different weight
        loss_weight_vector = tf.squeeze(tf.where(labels > 5, self.loss_weight, 1))
        acc_weight_vector = tf.squeeze(tf.where(labels > 5, self.acc_weight, 1))
        # training=False is only needed if there are layers with different
        # behavior during training versus inference (e.g. Dropout).
        predictions = self.model(features, training=False)
        self.test_cm.update_state(labels, predictions)
        t_loss = self.loss_object(labels, predictions, sample_weight=loss_weight_vector)
        self.test_loss(t_loss)
        self.test_accuracy.update_state(labels, predictions, sample_weight=acc_weight_vector)


    # def write_scalar_summary(self, other_metrices_train, other_metrices_test, step):
    def write_scalar_summary(self, step, other_metrices_test_12,
                            other_metrices_test_6_normal, other_metrices_test_6_transition):
        """ Write scalar summary to tensorboard """

        with self.train_summary_writer.as_default():
            tf.summary.scalar('loss', self.train_loss.result(), step=step)
            tf.summary.scalar('accuracy', self.train_accuracy.result(), step=step)
        
        with self.test_summary_writer.as_default():
            tf.summary.scalar('loss', self.test_loss.result(), step=step)
            tf.summary.scalar('accuracy', self.test_accuracy.result(), step=step)
            tf.summary.scalar('val_accuracy_6class_transition', other_metrices_test_6_transition['unbalanced_acc'], step=step)
            tf.summary.scalar('val_accuracy_6class_normal',  other_metrices_test_6_normal['unbalanced_acc'], step=step)

            tf.summary.scalar('val_precision_12class',  other_metrices_test_12['precision'], step=step)
            tf.summary.scalar('val_recall_12class',  other_metrices_test_12['recall'], step=step)
            tf.summary.scalar('val_f1_score_12class',  other_metrices_test_12['f1_score'], step=step)
            tf.summary.scalar('val_balanced_acc_12class',  other_metrices_test_12['balanced_acc'], step=step)




    # def write_wandb(self, other_metrices_train, other_metrices_test, step):
    def write_wandb(self, step, optimze_balanced_acc, other_metrices_test_12,
                    other_metrices_test_6_normal ,other_metrices_test_6_transition):
        """ Write summary to wandb """
        logs_dict = {
            'train_loss': self.train_loss.result(),
            'train_accuracy': self.train_accuracy.result(),

            'val_loss': self.test_loss.result(),
            'val_accuracy_12class': self.test_accuracy.result(),

            'val_accuracy_6class_transition': other_metrices_test_6_transition['unbalanced_acc'],
            'val_accuracy_6class_normal': other_metrices_test_6_normal['unbalanced_acc'],

            'val_precision_12class': other_metrices_test_12['precision'],
            'val_recall_12class': other_metrices_test_12['recall'],
            'val_f1_score_12class': other_metrices_test_12['f1_score'],
            'val_balanced_acc_12class': other_metrices_test_12['balanced_acc'],

            'step': step
        }
        
        if optimze_balanced_acc:
            logs_dict['max_acc_balanced'] = self.max_acc_balanced
        else:
            logs_dict['max_acc'] = self.max_acc
        
        wandb.log(logs_dict)

    def train(self, optimze_balanced_acc=False):

        logging.info('\n======== Starting Training ========')

        for idx, (features, labels) in enumerate(self.ds_train):
            step = idx + 1
            self.train_step(features, labels)

            if step % self.log_interval == 0:
                # Reset test metrics
                self.test_loss.reset_states()
                self.test_accuracy.reset_states()
                self.test_cm.reset_states()

                for test_features, test_labels in self.ds_val:
                    self.test_step(test_features, test_labels)

                template = 'Step {}, Loss: {}, Accuracy: {:.2f}%, Test Loss: {}, Validation Accuracy: {:.2f}%'
                logging.info(template.format(step,
                                             self.train_loss.result(),
                                             self.train_accuracy.result() * 100,
                                             self.test_loss.result(),
                                             self.test_accuracy.result() * 100))

                other_metrices_test_12 = self.test_cm.acc()
                other_metrices_test_6_normal = self.test_cm.acc(six_class=True)
                other_metrices_test_6_transition = self.test_cm.acc(six_class=True, transition=True)

                # write summary to tensorboard
                self.write_scalar_summary(step, other_metrices_test_12, other_metrices_test_6_normal, other_metrices_test_6_transition  )
                # write summary to wandb
                if self.wandb:
                    self.write_wandb(step, optimze_balanced_acc, other_metrices_test_12,
                                     other_metrices_test_6_normal, other_metrices_test_6_transition)
                # Reset train metrics
                self.train_loss.reset_states()
                self.train_accuracy.reset_states()

                yield self.test_accuracy.result().numpy()

            if step % self.ckpt_interval == 0:
                if optimze_balanced_acc:
                    if self.max_acc_balanced <= other_metrices_test_12['balanced_acc']:
                        self.max_acc_balanced = other_metrices_test_12['balanced_acc']
                        logging.info(f'Saving better balanced val_acc checkpoint to {self.run_paths["path_ckpts_train"]}.')
                        self.manager.save()
                else:
                    if self.max_acc <= self.test_accuracy.result():
                        self.max_acc = self.test_accuracy.result()
                        logging.info(f'Saving better val_acc checkpoint to {self.run_paths["path_ckpts_train"]}.')
                        self.manager.save()
                    

            if step % self.total_steps == 0:
                logging.info(f'Finished training after {step} steps.')

                return self.test_accuracy.result().numpy()

        logging.info('======== Finished Training ========')

