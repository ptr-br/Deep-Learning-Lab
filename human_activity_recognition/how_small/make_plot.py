from cProfile import label
import matplotlib
from matplotlib import markers
import numpy as np
import pandas as pd
import  matplotlib.pyplot as plt

fc_no ="./start5_end300_step10"
fc_3 = "./start5_end300_step10_fc3"

windows_list = []
values_list = []
legend_list = ["normal data", "fc=3"]

for file_path in (fc_no, fc_3):
    values = pd.read_csv(file_path + "/values.csv", header=None)
    windows = pd.read_csv(file_path + "/windows.csv",  header=None) 
    values_list.append(values)
    windows_list.append(windows)


fig = plt.figure(figsize=(7, 5))

for i in range(len(values_list)):
    plt.plot(windows_list[i], values_list[i], linestyle="--", marker="o", label=legend_list[i])

plt.title("How small can the window size get?")
plt.xlabel("Window size")
plt.ylabel("Accuracy")
plt.legend(loc="upper left")

plt.savefig("combined_plot.png")

