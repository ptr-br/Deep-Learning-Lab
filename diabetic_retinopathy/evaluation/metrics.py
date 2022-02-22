import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
import numpy as np


class ConfusionMatrix(tf.keras.metrics.Metric):

    def __init__(self, n_classes=2, boundary=0.5, name="confusion_matrix", **kwargs):
        super(ConfusionMatrix, self).__init__(name=name, **kwargs)
        self.cm = self.add_weight(name="cm", shape=(n_classes, n_classes),
                                  dtype=tf.int32, initializer="zeros")
        self.boundary = boundary

    def update_state(self, y_true, y_pred):
        y_true = tf.squeeze(y_true, axis=-1)
        # y_pred = tf.argmax(y_pred, axis=1)
        matrix = tf.math.confusion_matrix(y_true, y_pred, num_classes=2)
        self.cm.assign_add(matrix)
    
    ## DEPRTECATED: USED FOR BINARY CROSS-ENTROPY LOSS
    # def update_state(self, y_true, y_pred):
    #     y_true = tf.squeeze(y_true, axis=-1)
    #
    #     # convert y_pred (sigmoid outputs probability) to labels 0 and 1
    #     y_pred = tf.squeeze(y_pred)
    #     one = tf.ones_like(y_pred)
    #     zero = tf.zeros_like(y_pred)
    #     y_pred = tf.where(y_pred > self.boundary, one, zero)
    #
    #     matrix = tf.math.confusion_matrix(y_true, y_pred, )
    #     self.cm.assign_add(matrix)

    def reset_states(self):
        for variable in self.variables:
            variable.assign(tf.zeros(shape=variable.shape, dtype=tf.dtypes.int32))

    def result(self):
        return self.cm

    def plot_cm(self):
        plt.figure(figsize=(5, 5))
        sns.heatmap(self.cm, annot=True, fmt="d")
        plt.title('Confusion matrix')
        plt.ylabel('Actual label')
        plt.xlabel('Predicted label')
        plt.show()

    def probability(self, return_dict=False):
        
        tp = self.cm[0,0]
        tn = self.cm[1,1]
        fp = self.cm[0,1]
        fn = self.cm[1,0]
       
        
        unbalanced_acc = (tp + tn) / (tp + fn + fp + tn)
        
        sensitivity = tp / (tp + fn) 
        specificity   = tn / (tn + fp) 
        balanced_acc = (sensitivity + specificity)/2
        
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        f1_score = 2 * (recall * precision)/(recall + precision)
        
        return {"sensitivity": sensitivity.numpy(), "specificity": specificity.numpy(),
                "f1_score": f1_score.numpy(), "unbalanced_acc": unbalanced_acc.numpy(),
                "balanced_acc": balanced_acc.numpy()}


if __name__ == "__main__":
    a = np.random.randint(0, 2, (1000, 1))
    b = tf.random.uniform((1000, 1))
    metrics_cm = ConfusionMatrix()
    metrics_cm.update_state(a, b)
    cm = metrics_cm.result()
    metrics_cm.plot_cm()
    template = 'sensitivity: {:.2f}%, specificity: {:.2f}%, f1_score: {:.2f}%, unbalanced_acc: {:.2f}%, balanced_acc: {:.2f}%'
    prob = metrics_cm.probability()
    print(template.format(prob[0]*100, prob[1]*100,
          prob[2]*100, prob[3]*100, prob[4]*100))
