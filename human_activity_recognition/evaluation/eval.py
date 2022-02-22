import tensorflow as tf
import logging
import gin
import os
import sys
import pandas as pd
import numpy as np
from evaluation.metrics import ConfusionMatrix
from scipy.stats import zscore, mode
from matplotlib import pyplot as plt

sys.path.insert(0, '../')
from input_pipeline.make_tfrecords import low_pass



@gin.configurable
class Evaluator():

    def __init__(self, model, ds_test, ds_val, run_paths, vis_exp_user):

        self.model = model
        self.run_paths = run_paths
        self.ds_test = ds_test
        self.ds_val = ds_val
        self.n_classes = model.output_shape[1]
        self.eval_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(
            name='train_accuracy')
        self.vis_exp_user = vis_exp_user
        self.window_length, _ = gin.query_parameter(
            'create_tfrecords.window_length_and_shift')
        self.color_dict = {1: 'lightcoral', 2: 'moccasin', 3: 'yellow', 4: 'yellowgreen', 5: 'lightgreen',
                           6: 'mediumaquamarine',
                           7: 'paleturquoise', 8: 'slateblue',
                           9: 'mediumpurple', 10: 'darkorchid', 11: 'plum', 12: 'lightpink', 0: 'white'}
        self.activity_labels = ['WALKING', 'WALKING_UPSTAIRS', 'WALKING_DOWNSTAIRS', 'SITTING', 'STANDING',
                                'LAYING', 'STAND_TO_SIT', 'SIT_TO_STAND', 'SIT_TO_LIE', 'LIE_TO_SIT', 'STAND_TO_LIE',
                                'LIE_TO_STAND']

    def plot_file(self, values, x, y, z, legend_x, legend_y, legend_z, title, run_paths):

        fig = plt.figure(figsize=(20, 4))
        for index, color in enumerate(values):
            plt.axvspan(index, index+1,
                        facecolor=self.color_dict[color], alpha=0.6)

        plt.plot(x, color='r', label=legend_x)
        plt.plot(y, color='b', label=legend_y)
        plt.plot(z, color='g', label=legend_z)

        plt.title(title)

        plt.legend(loc="upper left")
        plot_path = os.path.join(
            run_paths['path_plot'], title + '_visualization.png')

        plt.savefig(plot_path)
        logging.info(f'Saving "{title}" plot to: {plot_path}')

    def plot_legend(self):
        plt.figure(figsize=(25, 5))
        x = np.arange(0, 12, 1)
        plt.bar(x, height=1, width=1, align='center',
                color=list(self.color_dict.values()))
        plt.xticks(x, self.activity_labels, rotation=60)
        plt.yticks([])
        plt.title('Colormap')
        plt.margins(0)
        ax = plt.gca()
        ax.spines['left'].set_color('black')
        ax.spines['right'].set_color('black')
        ax.spines['top'].set_color('black')
        ax.spines['bottom'].set_color('black')
        # change figure size
        fig = plt.gcf()
        fig.set_size_inches(8, 4)
        plt.tight_layout()
        plot_path = os.path.join(self.run_paths['path_plot'], 'colormap.png')
        plt.savefig(plot_path)
        
    def vis_fc_plots(self, x_acc, x_fc, phi_gyro, phi_fc, fc):
        """
        Create plot if lowpass is used (fc)

        Args:
            fc (int): cutoff frequency
            x_acc (np.array): normal data of acc
            x_fc (np.array): lowpass data of acc
            phi_gyro (np.array): normal data of gyro
            phi_fc (np.array): lowpass data of gyro
        """
        # create plots for acc
        fig = plt.figure(figsize=(20, 4))
        plt.plot(x_acc, color='r', label='normal data')
        plt.plot(x_fc, color='b', label=f'fc={fc}')
        
        plt.legend(loc="upper right")
        plot_path = os.path.join(self.run_paths['path_plot'], 'frequency_comparison_acc.png')

        plt.title("Acc lowpass comparison (first-axis)")
        plt.savefig(plot_path)
        logging.info(f'Saving frequency comparison acc plot to: {plot_path}')
        plt.close()
        
        # create plots for gyro
        fig = plt.figure(figsize=(20, 4))
        plt.plot(phi_gyro, color='g', label='normal data')
        plt.plot(phi_fc, color='b', label=f'fc={fc}')
        
        plt.legend(loc="upper right")
        plot_path = os.path.join(self.run_paths['path_plot'], 'frequency_comparison_gyro.png')

        plt.title("Gyro lowpass comparison (first-axis)")
        plt.savefig(plot_path)
        logging.info(f'Saving frequency comparison gyro plot to: {plot_path}')
        plt.close()
            
            

    def vis_file(self, data_dir, fc):
        """
        Visualize acc and gyro data

        Args:
            data_dir (str): path to the directory where data is stored
            fc (int): cutoff frequency
        """

        logging.info(
            f'\n======== Starting Visualization of complete file  ========')

        # load gyro and acc data from file
        labels = pd.read_csv(os.path.join(
            data_dir, "labels.txt"), sep=" ", header=None)
        acc_file = data_dir + f"acc_exp" + \
            str(self.vis_exp_user[0]).zfill(2) + "_user" + \
            str(self.vis_exp_user[1]).zfill(2) + ".txt"
        gyro_file = data_dir + f"gyro_exp" + \
            str(self.vis_exp_user[0]).zfill(2) + "_user" + \
            str(self.vis_exp_user[1]).zfill(2) + ".txt"

        acc_data = pd.read_csv(acc_file, sep=" ", header=None)
        gyro_data = pd.read_csv(gyro_file, sep=" ", header=None)

        sensor_data = pd.concat([acc_data, gyro_data], axis=1)
        
        # low pass filter the data and store visual comparison 
        if fc > 0 and fc < 25:
            # get one sequence in the file sta
            x_acc = sensor_data[0].values[5000:5000+self.window_length][:,0]
            phi_gyro  = sensor_data[0].values[5000:5000+self.window_length][:,1]
            sensor_data = low_pass(sensor_data, fc=fc)
            # make new data frame since lowpass filtering removes it 
            x_fc = sensor_data[:,0][5000:5000+self.window_length]
            phi_fc  = sensor_data[:,3][5000:5000+self.window_length]
            
            sensor_data = pd.DataFrame(sensor_data)
              
            self.vis_fc_plots(x_acc, x_fc, phi_gyro, phi_fc, fc)
        
        sensor_data.columns = ['acc1', 'acc2',
                               'acc3', 'gyro1', 'gyro2', 'gyro3']

        norm_sensor_data = zscore(sensor_data, axis=0)

        file_length = sensor_data.shape[0]

        gt_color_values = []
        pred_color_values = []

        # set initial label
        norm_sensor_data['label'] = 0

        # get gt data
        for index, (exp, usr, act, sco, eco) in labels.iterrows():
            if exp == self.vis_exp_user[0] and usr == self.vis_exp_user[1]:
                norm_sensor_data.loc[sco:eco, 'label'] = act

        # get gt and prediction labels for sequence (batched into window_length specified)
        for i in range(0, file_length, self.window_length):

            # get gt label for sequence
            gt_seq_labels =  norm_sensor_data.loc[i:i + self.window_length-1, 'label'].to_numpy()
            gt_color_values.append(gt_seq_labels)
            # do not use unlabeled data
            if mode(gt_seq_labels)[0][0] == 0:
                pred_color_values.append(gt_seq_labels)

            else:
                # pred_seq_label
                features = norm_sensor_data.values[i:i +
                                                   self.window_length, :-1]
                features = np.expand_dims(features, 0)

                preds = self.model(features, training=False)
                # add 1 to get back range from 1-12
                predicted_label = np.argmax(preds) + 1
                predicted_labels = np.full((self.window_length), predicted_label)

                pred_color_values.append(predicted_labels)
                
        pred_color_values = np.concatenate(pred_color_values).ravel()
        gt_color_values = np.concatenate(gt_color_values).ravel()
                
                            
        # Plot results
        self.plot_file(values=gt_color_values, x=norm_sensor_data['acc1'].values, y=norm_sensor_data['acc2'].values,
                       z=norm_sensor_data['acc3'].values,
                       legend_x='acc_X', legend_y='acc_Y', legend_z='acc_Z', title="ground_truth_acc",
                       run_paths=self.run_paths)

        self.plot_file(values=pred_color_values, x=norm_sensor_data['acc1'].values, y=norm_sensor_data['acc2'].values,
                       z=norm_sensor_data['acc3'].values,
                       legend_x='acc_X', legend_y='acc_Y', legend_z='acc_Z', title="predictions_acc",
                       run_paths=self.run_paths)

        self.plot_file(values=gt_color_values, x=norm_sensor_data['gyro1'].values, y=norm_sensor_data['gyro2'].values,
                       z=norm_sensor_data['gyro3'].values,
                       legend_x='gyro_X', legend_y='gyro_Y', legend_z='gyro_Z', title="ground_truth_gyro",
                       run_paths=self.run_paths)

        self.plot_file(values=pred_color_values, x=norm_sensor_data['gyro1'].values, y=norm_sensor_data['gyro2'].values,
                       z=norm_sensor_data['gyro3'].values,
                       legend_x='gyro_X', legend_y='gyro_Y', legend_z='gyro_Z', title="predictions_gyro",
                       run_paths=self.run_paths)

        self.plot_legend()

        logging.info(
            f'======== Finished Visualization of complete file  ========')

    def evaluate(self, yield_acc=False):
        """
        Evaluete on validation and teat dataset

        Args:
            yield_acc (bool, optional): If True return the accuracy scores for 
                                        the 12 class problem. Defaults to False.
        """

        logging.info('\n======== Starting Evaluation ========')

        checkpoint = tf.train.Checkpoint(model=self.model)
        checkpoint.restore(tf.train.latest_checkpoint(
            self.run_paths['path_ckpts_train'])).expect_partial()
        logging.info(
            f'Checkpoint restored from: {self.run_paths["path_ckpts_train"]}')

        cm = ConfusionMatrix(n_classes=self.n_classes)

        template_each_class = 'sensitivity: {:.2f}%, specificity: {:.2f}%, F1-score: {:.2f}%'
        template_acc = 'unbalanced_acc: {:.2f}%, balanced_acc: {:.2f}%'
        template_total = 'precision: {:.2f}%, recall: {:.2f}%, F1-score: {:.2f}%'

        for name, ds in [('val', self.ds_val), ('test', self.ds_test)]:

            logging.info(f'\n======== Evaluating on {name} dataset ========')

            for features, labels in ds:
                predictions = self.model(features, training=False)
                cm.update_state(labels, predictions)

                
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

            total_acc_six_normal = cm.acc(six_class=True)
            logging.info(
                "Total acc 6 class problem (normal): " + template_acc.format(total_acc_six_normal["unbalanced_acc"] * 100,
                                                                             total_acc_six_normal["balanced_acc"] * 100)
            )
            logging.info(
                "Other total metrics (normal): " + template_total.format(total_acc_six_normal["precision"] * 100,
                                                                            total_acc_six_normal["recall"] * 100,
                                                                            total_acc_six_normal["f1_score"] * 100)
            )

            total_acc_six_trans = cm.acc(six_class=True, transition=True)
            logging.info(
                "Total acc 6 class problem (transition): " + template_acc.format(total_acc_six_trans["unbalanced_acc"] * 100,
                                                                                 total_acc_six_trans["balanced_acc"] * 100)
            )
            logging.info(
                "Other total metrics (transition): " + template_total.format(total_acc_six_trans["precision"] * 100,
                                                                            total_acc_six_trans["recall"] * 100,
                                                                            total_acc_six_trans["f1_score"] * 100)
            )


            logging.info('Confusion matrix:\n{}'.format(
                cm.result()[:int(self.n_classes), :int(self.n_classes)].numpy()))
            
            cm_against = cm.result()

            for i in range(self.n_classes):
                prob = cm.probability(class_index=i, cm=cm_against)
                logging.info(
                    f'Class {i + 1} against all: ' +
                    template_each_class.format(prob['sensitivity'] * 100,
                                               prob['specificity'] * 100, prob['f1_score'] * 100))

            if name == "test":
                cm.plot_heatmap(
                    self.run_paths, total_acc["unbalanced_acc"] * 100, total_acc["balanced_acc"] * 100)
                logging.info(
                    f"Save confusion matrix plot to the path: {self.run_paths['path_plot']}/confusion_matrix.png")
                if yield_acc:
                    return total_acc

            cm.reset_states()

        logging.info('======== Finished Evaluation ========')
