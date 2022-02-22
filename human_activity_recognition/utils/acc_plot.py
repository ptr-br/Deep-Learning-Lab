
import matplotlib.pyplot as plt
import numpy as np

def bar_plot(labels, acc, balanced_acc, title, save_name):
    x = np.arange(len(labels))  # the label locations
    width = 0.35  # the width of the bars

    fig, ax = plt.subplots()
    plt.ylim((85, 100))
    rects1 = ax.bar(x - width / 2, acc, width, label='Acc')
    rects2 = ax.bar(x + width / 2, balanced_acc, width, label='Balanced_acc')

    # Add some text for labels, title and custom x-axis tick labels, etc.

    ax.set_ylabel('Accuracy / %')
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    plt.savefig(save_name)

labels = ["no filter", 'fc=3', 'fc=10', 'fc=15', 'fc=20']
Acc = [95.20, 96.67, 95.20, 94.32, 94.61]
Balanced_acc = [89.21, 92.73, 91.79, 88.46, 87.32]
title = 'Mean acc of different lowpassfilter'
save_name = "ablation_fc.jpg"
bar_plot(labels, Acc, Balanced_acc, title, save_name)

labels = ["1", '2', '5', '10', '20']
Acc = [95.20, 95.31, 95.20, 95.88, 95.76]
Balanced_acc = [89.21, 88.93, 89.11, 90.93, 91.47]
title = 'Mean acc of different weighted loss'
save_name = "ablation_weighted_loss.jpg"
bar_plot(labels, Acc, Balanced_acc, title, save_name)