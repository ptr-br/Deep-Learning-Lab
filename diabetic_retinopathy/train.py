import gin
import tensorflow as tf
import logging
import datetime
import os
import wandb
import sys
from evaluation import metrics
from utils.gc_optimizer import GCAdam


@gin.configurable
class Trainer(object):
    
    
    def __init__(self, model, ds_train, ds_val, ds_info, run_paths,
                 total_steps, log_interval, ckpt_interval,
                 lr_rate=0.005, wandb=False, ckpt=False):
        
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
        self.loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        # self.loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=False)
        self.optimizer = tf.keras.optimizers.Adam()
        # self.optimizer = GCAdam(lr_rate)
        # lr_scheduler = tf.keras.optimizers.schedules.CosineDecay(initial_learning_rate=lr_rate,
        #                                                        decay_steps=1000,
        #                                                        alpha=0.1)
        # self.optimizer = tf.keras.optimizers.Adam(lr_scheduler)
        # Metrics
        self.train_loss = tf.keras.metrics.Mean(name='train_loss')
        # self.train_accuracy = tf.keras.metrics.BinaryAccuracy(name='train_accuracy')
        self.train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')
        self.train_auc = tf.keras.metrics.AUC(name='train_auc')
        self.train_cm = metrics.ConfusionMatrix()

        self.test_loss = tf.keras.metrics.Mean(name='test_loss')
        self.test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')
        # self.test_accuracy = tf.keras.metrics.BinaryAccuracy(name='test_accuracy')
        self.test_auc = tf.keras.metrics.AUC(name='test_auc')
        self.test_cm = metrics.ConfusionMatrix()

        self.model = model
        self.ds_train = ds_train
        self.ds_val = ds_val
        self.ds_info = ds_info
        self.run_paths = run_paths
        self.total_steps = total_steps
        self.log_interval = log_interval
        self.ckpt_interval = ckpt_interval
        self.max_acc = 0.0

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
                    logging.error('Your defined model and the parameters that should be loaded do not fit. '\
                                  'Please make sure that the checkpoint is from the same architcture as the model you want to train further!')
                    sys.exit(1)
            else:
                logging.error(f'Could not find any checkpoints at path specified! Please make sure chekpoints exist at {ckpt}.')
        else:
            self.manager = tf.train.CheckpointManager(self.ckpt, run_paths["path_ckpts_train"], max_to_keep=3)
            logging.info("Training will be from scratch since no valid checkpoint was specified.")
            logging.info(f"All checkpoints will be stored in: {run_paths['path_ckpts_train']}")
        
        # Turn on weights and biases logging if wanted
        self.wandb = wandb

    @tf.function
    def train_step(self, images, labels):
        with tf.GradientTape() as tape:
            # training=True is only needed if there are layers with different
            # behavior during training versus inference (e.g. Dropout).
            predictions = self.model(images, training=True)
            loss = self.loss_object(labels, predictions)
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(
            zip(gradients, self.model.trainable_variables))
        self.train_loss(loss)
        self.train_accuracy(labels, predictions)
        y_pred = tf.argmax(predictions, axis=1)
        self.train_cm.update_state(labels, y_pred)
        pred = tf.nn.softmax(predictions)
        pred = pred[:, 1]
        self.train_auc(labels, pred)

    @tf.function
    def test_step(self, images, labels):
        # training=False is only needed if there are layers with different
        # behavior during training versus inference (e.g. Dropout).
        predictions = self.model(images, training=False)
        t_loss = self.loss_object(labels, predictions)

        self.test_loss(t_loss)
        self.test_accuracy(labels, predictions)
        y_pred = tf.argmax(predictions, axis=1)
        self.test_cm.update_state(labels, y_pred)
        pred = tf.nn.softmax(predictions)
        pred = pred[:, 1]
        self.test_auc(labels, pred)

    def write_scalar_summary(self, other_metrices_train, other_metrices_test, step):
        """ Write scalar summary to tensorboard """

        with self.train_summary_writer.as_default():
            tf.summary.scalar('loss', self.train_loss.result(), step=step)
            tf.summary.scalar('accuracy', self.train_accuracy.result(), step=step)
            tf.summary.scalar('AUC', self.train_auc.result(), step=step)
            tf.summary.scalar('sensitivity',      other_metrices_train['sensitivity'], step=step)
            tf.summary.scalar('specificity', other_metrices_train['specificity'], step=step)
            tf.summary.scalar('f1_score',                other_metrices_train['f1_score'], step=step)
            tf.summary.scalar('balanced_acc',  other_metrices_train['balanced_acc'], step=step)

        with self.test_summary_writer.as_default():
            tf.summary.scalar('loss', self.test_loss.result(), step=step)
            tf.summary.scalar('accuracy', self.test_accuracy.result(), step=step)
            tf.summary.scalar('AUC', self.test_auc.result(), step=step)
            tf.summary.scalar('sensitivity',      other_metrices_test['sensitivity'], step=step)
            tf.summary.scalar('specificity', other_metrices_test['specificity'], step=step)
            tf.summary.scalar('f1_score',                other_metrices_test['f1_score'], step=step)
            tf.summary.scalar('balanced_acc',  other_metrices_test['balanced_acc'], step=step)

    def write_wandb(self, other_metrices_train, other_metrices_test, step):
        """ Write summary to wandb """

        wandb.log({
                'train_loss': self.train_loss.result(),
                'train_accuracy': self.train_accuracy.result(),
                'train_AUC': self.train_auc.result(),
                'train_sensitivity': other_metrices_train['sensitivity'],
                'train_specificity': other_metrices_train['specificity'],
                'train_f1_score':    other_metrices_train['f1_score'],
                'train_balanced_acc':  other_metrices_train['balanced_acc'],

                'val_loss': self.test_loss.result(),
                'val_accuracy': self.test_accuracy.result(),
                'val_AUC': self.test_auc.result(),
                'val_sensitivity': other_metrices_test['sensitivity'],
                'val_specificity': other_metrices_test['specificity'],
                'val_f1_score':    other_metrices_test['f1_score'],
                'val_balanced_acc':  other_metrices_test['balanced_acc'],
                'max_val_acc': self.max_acc,

                'step':step
            })


    def train(self):
        
        logging.info('\n======== Starting Training ========')
        
        for idx, (images, labels) in enumerate(self.ds_train):
            step = idx + 1
            self.train_step(images, labels)

            if step % self.log_interval == 0:
                # Reset test metrics
                self.test_loss.reset_states()
                self.test_accuracy.reset_states()
                self.test_auc.reset_states()
                self.test_cm.reset_states()

                for test_images, test_labels in self.ds_val:
                    self.test_step(test_images, test_labels)

                template = 'Step {}, Loss: {}, Accuracy: {:.2f}%, Validation Loss: {}, Validation Accuracy: {:.2f}%'
                logging.info(template.format(step,
                                             self.train_loss.result(),
                                             self.train_accuracy.result() * 100,
                                             self.test_loss.result(),
                                             self.test_accuracy.result() * 100))

                other_metrices_test = self.test_cm.probability(return_dict=True)
                other_metrices_train = self.train_cm.probability(return_dict=True)

                # write summary to tensorboard
                self.write_scalar_summary(other_metrices_train, other_metrices_test, step)

                # write summary to wandb
                if self.wandb:
                    self.write_wandb(other_metrices_train, other_metrices_test, step)
                
                
                # Reset train metrics
                self.train_loss.reset_states()
                self.train_accuracy.reset_states()
                self.train_auc.reset_states()
                self.train_cm.reset_states()

                yield self.test_accuracy.result().numpy()

            if step % self.ckpt_interval == 0:
                if  self.max_acc < self.test_accuracy.result():
                    self.max_acc = self.test_accuracy.result()
                    logging.info(f'Saving better val_acc checkpoint to {self.run_paths["path_ckpts_train"]}.')
                    self.manager.save()
                else:
                    pass

            if step % self.total_steps == 0:
                logging.info(f'Finished training after {step} steps.')
                

                return self.test_accuracy.result().numpy()

        logging.info('======== Finished Training ========')
