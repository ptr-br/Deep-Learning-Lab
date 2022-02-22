import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
import numpy as np
import tensorflow.keras as keras
import os


class ConfusionMatrix(keras.metrics.Metric):
    def __init__(self, n_classes, name="confusion_matrix", **kwargs):
        super(ConfusionMatrix, self).__init__(name=name, **kwargs)
        self.cm = self.add_weight(name="cm", shape=(n_classes, n_classes),
                                  dtype=tf.float32, initializer="zeros")
        self.n_classes = n_classes

    def update_state(self, y_true, y_pred):
        y_true = tf.squeeze(y_true, axis=-1)
        y_pred = tf.argmax(y_pred, axis=1)
        matrix = tf.math.confusion_matrix(
            y_true, y_pred, num_classes=self.n_classes, dtype=tf.float32)
        matrix = tf.transpose(matrix)

        self.cm.assign_add(matrix)

    def reset_states(self):
        for variable in self.variables:
            variable.assign(tf.zeros(shape=variable.shape, dtype=tf.float32))

    def result(self):
        return self.cm

    def probability(self, class_index, cm):
        tp = cm[class_index, class_index]
        fn = tf.reduce_sum(cm[:, class_index]) - tp
        fp = tf.reduce_sum(cm[class_index, :]) - tp
        tn = tf.reduce_sum(cm) - tp - fp - fn

        unbalanced_acc = (tp + tn) / (tp + fn + fp + tn)

        sensitivity = tp / (tp + fn + keras.backend.epsilon())
        specificity = tn / (tn + fp + keras.backend.epsilon())
        balanced_acc = (sensitivity + specificity) / 2

        precision = tp / (tp + fp + keras.backend.epsilon())
        recall = tp / (tp + fn + keras.backend.epsilon())
        f1_score = 2 * (recall * precision) / (recall +
                        precision + keras.backend.epsilon())

        return {"class_index": class_index, "sensitivity": sensitivity.numpy(),
                "specificity": specificity.numpy(), "f1_score": f1_score.numpy(),
                "unbalanced_acc": unbalanced_acc.numpy(), "balanced_acc": balanced_acc.numpy(),
                "precision": precision.numpy(), "recall": recall.numpy()}

    def acc(self, six_class=False, transition=False):
        if six_class:
            split = int(self.n_classes / 2)
            if transition:
                # only get tranaition activities
                 cm=self.cm[split:, split:]
               
            else:
                # only get normal activities
                cm = self.cm[:split, :split]
        else:
            cm=self.cm
        unbalanced_acc=tf.reduce_sum(
            tf.linalg.diag_part(cm)) / tf.reduce_sum(cm)
        balanced_acc=0.0
        precision=0.0
        recall=0.0
        f1_score=0.0

        for i in range(cm.shape[0]):
            tp=cm[i, i]
            fn=tf.reduce_sum(cm[:, i]) - tp
            accuracy=tp / (tp + fn + keras.backend.epsilon())
            balanced_acc += accuracy
            prob=self.probability(i, cm)
            precision += prob["precision"]
            recall += prob["recall"]
            f1_score += prob["f1_score"]

        balanced_acc /= cm.shape[0]
        precision /= cm.shape[0]
        recall /= cm.shape[0]
        f1_score /= cm.shape[0]

        return {"unbalanced_acc": unbalanced_acc.numpy(), "balanced_acc": balanced_acc.numpy(),
                "precision": precision, "recall": recall, "f1_score": f1_score}

    def plot_heatmap(self, run_paths, unbalanced_acc, balanced_acc):

        plot_path = os.path.join(
            run_paths['path_plot'], 'confusion_matrix.png')

        categories=["WALKING", "WALKING_UPSTAIRS", "WALKING_DOWNSTAIRS", "SITTING", "STANDING", "LAYING",
                      "STAND_TO_SIT", "SIT_TO_STAND", "SIT_TO_LIE", "LIE_TO_SIT", "STAND_TO_LIE", "LIE_TO_STAND"]

        plt.figure(figsize=(20, 20))

        # normalize
        normalized_cm= self.cm / np.sum(self.cm)
        columnwise_normalized_cm= tf.divide(self.cm, np.sum(self.cm, axis=0))

        # draw heatmap for results
        sns.heatmap(columnwise_normalized_cm, annot_kws={'va': 'bottom'}, annot=columnwise_normalized_cm,
                    fmt='.1%', cmap='Blues', xticklabels=categories, yticklabels=categories, cbar=False)
        sns.heatmap(columnwise_normalized_cm, annot_kws={'va': 'top'}, annot=self.cm,
                    fmt='g', cmap='Blues', xticklabels=categories, yticklabels=categories, cbar=False)

        plt.title('Prediction Heatmap\n\nUnbalanced_acc: {:.2f}%\nBalanced_acc: {:.2f}%'. \
                  format(unbalanced_acc, balanced_acc))
        plt.xlabel('True-Class', fontsize=12)
        plt.ylabel('Prediction', fontsize=12)
        # abuse legend property to explain numbers
        plt.legend(title='Explanation of values:\n 1: % (normalized per column)\n 2: total number of occurrences',
                   loc='upper right', labels=[])

        plt.savefig(plot_path)

