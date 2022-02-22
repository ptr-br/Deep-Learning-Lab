# Human Activity Recognition

## Setup
There is not much to do from your side. We used relative paths, so make sure to execute everything from this directory on the server. Most commands we mention in this description are also in the `run.sh`. You can simply uncomment the related lines and run it.

## Preprocessing
#### Notes on the dataset
We encountered that there are much less transition activites then normal ones (~1:10). Therfore, the dataset get resampled into two differnet groups. Transition activitis get resampled to the count majority of transition activites and normal activities get resampled to the count majority of normal activities.

#### Label assignment strategy
We used a hard coded label assignment strategy. At first, the mode of a sequence is selected. If the mode belongs to a normal activity, the sequence has to contain more than 85% of that label, otherwise the sequence is removed from the dataset. For sequences with a transition activity as mode, we assign the label if more then 40% of the sequence contains the label.
We found this strategy to work quite well, especially because it emphasizes on the transition sequences that are often very short. 

## How to run the code 
Executing the script, the default behavior is to first train the model for the steps specified in your *config.gin*. After training is finished, the model will be evaluated on the test set. Furthermore, the file choosen per configurtion(user and experiment numbers) will be visulaized and stored during the evaluation.
All three substeps (train | evaluate | visualize) can also be executed on their own.

## Getting started 
To train your first model you can simply run `python3 main.py`.
```python
# This is running the script in default mode (GRU model - train, eval and visualize)
python3 main.py
```
## Traning
Only training a model can be done by passing the *train* flag.
```python
# Training example 
python3 main.py --run train --model_name gru
```
## Evaluation 
Evaluation is performed by default. You can have a look at eval.log to see the results. If by accident the file get's damaged or is lost you can rerun evaluation by setting the flag and giving the model path.

```python
# Evaluation example (lstm - already trained)
python3 main.py --run eval --model_name lstm --model_id  /home/RUS_CIP/st<YOUR_ST_NUMBER_HERE>/dl-lab-21w-team20/runs/run_<date>
```
>- make sure model_name and model_id are from the same type (here lstm)
 
Evaluation outputs a eval.log file, which contains follows:
- The class i against all: sensitivity, specificity, F1-score
- Total acc 6 class problem (normal): unbalanced_acc, balanced_acc
Other total metrics (normal): precision, recall, F1-score
- Total acc 6 class problem (transition): unbalanced_acc, balanced_acc
Other total metrics (transition): precision, recall, F1-score
- Total acc 12 class problem: unbalanced_acc, balanced_acc
Other total metrics: precision, recall, F1-score
- Total 12 class Confusion matrix 

### Visualization
After evaluation, a choosen file will be visualized. The file to be viualized can be set in *config.gin* by editing the parameter `Evaluator.vis_exp_user = (60, 30)`. This creates the visualization for acc_exp60_user30.txt & gyro_exp60_user30.txt (make sure to use a tuple that belongs to an existing file). The output will be stored in the output folder (in the plot subdirectory). One can also add new files to the output of an already trained model.
The visualization contains the groud truth and prediction of the accelerometer and gyroscope signals and their labels. The following is an example from a trained GRU model.
#### Visulaization Example 
![Alt text](https://github.tik.uni-stuttgart.de/iss/dl-lab-21w-team20/blob/master/human_activity_recognition/best_runs/gru_he_normal/plot/ground_truth_acc_visualization.png)
![Alt text](https://github.tik.uni-stuttgart.de/iss/dl-lab-21w-team20/blob/master/human_activity_recognition/best_runs/gru_he_normal/plot/predictions_acc_visualization.png)
![Alt text](https://github.tik.uni-stuttgart.de/iss/dl-lab-21w-team20/blob/master/human_activity_recognition/best_runs/gru_he_normal/plot/ground_truth_gyro_visualization.png)
![Alt text](https://github.tik.uni-stuttgart.de/iss/dl-lab-21w-team20/blob/master/human_activity_recognition/best_runs/gru_he_normal/plot/predictions_gyro_visualization.png)
<p align="center">
  <img src="https://github.tik.uni-stuttgart.de/iss/dl-lab-21w-team20/blob/master/human_activity_recognition/best_runs/gru_he_normal/plot/colormap.png" />
</p>


## Performance
Please note that the results may be hard to reproduce since the model performance varies between different runs. We ran each model five times for 1500 steps and report the best (and average) result here. Best validation checkpoints are stored and later on used for evaluation. You can have a look at our log files in  `bet_runs/` to see the detailed results of each run made for the chart below.

| Model | Accuracy [%] | Balanced Accuracy [%] | Precision [%]  | Recall [%]| F1-Score [%] |
| :---: | :---: | :---: | :---: | :---: | :---: |
| LSTM | 96.10 (92.78) |  90.26 (84.72) |   91.75 (85.23) | 90.26 (84.72)| 90.50 (84.39) |
| GRU |  **96.87** (95.20) | 92.24 (89.21) | 91.87 (90.40)  | 92.24 (89.21) | 91.61 (88.90)  |
| TCN | 96.15 (95.17) | 92.45 (85.78)| 92.91 (89.91)| 92.45 (85.78)| 92.49 (87.31)|
| Ensemble | 96.46 | **93.04**| **94.56**| **93.04**| **93.50**|



### GRU
This is our default model. The script can be simply executed and the model will be trained.
To do so, execute `python3 main.py`.
The default architecture has following parameters:
```python
rnn.n_classes = 12
rnn.rnn_units = 61
rnn.rnn_type = "gru"
rnn.num_rnn = 3
rnn.dense_units = 89
rnn.num_dense = 1
rnn.dropout_dense = 0.3304
rnn.dropout_rnn = 0.062
rnn.kernel_initializer = "glorot_uniform"
rnn.bi_direction = False
```
### LSTM
Architecture like GRU model but using lstm layer as rnn layer. Run it with `python3 main.py --model_name lstm`.
Default parameters obtained from hyperparameter optimization are commented in the config-file.

### TCN
Temporal Convolutional Network (TCN) exhibits longer memory than recurrent architectures with the same capacity. It has parallelism (convolutional layers), flexible receptive field size (possible to specify how far the model can see), stable gradients (backpropagation through time, vanishing gradients). 
We just use the default model from [[Repo](https://github.com/philipperemy/keras-tcn)].

Before runing this model, first `pip3 install keras-tcn --no-dependencies`, and then run `python3 main.py --model_name tcn`

### Ensemble
Ensemble learning uses the best three model from our different architectures and averages the outputs of these. Run it with `python3 ensemble.py`

### Ablation 
#### glorot_uniform
`python3 main.py --model_name gru --model_id glorot_uniform  --initialization glorot_uniform`for glorot_uniform kernel_initializer of rnn (default).
#### he_uniform
`python3 main.py --model_name gru --model_id he_normal  --initialization he_normal`for he_normal kernel_initializer of rnn.
#### lecun_uniform
`python3 main.py --model_name gru --model_id lecun_normal  --initialization lecun_normal`for lecun_normal kernel_initializer of rnn.
#### bidirectional layer
`python3 main.py --model_name gru --model_id bidirectional  --bidirectional=True` for using bidrectional layer in model.
#### weight loss
`python3 main.py --model_name gru --model_id loss_weight=2 --loss_weight 2` for choosing different weight_loss during training.(interger 2 can be replaced by other, the bigger the interger, the bigger attenion model will focus on the transtion labels)
#### lowpassfilter
`python3 main.py --model_name gru --model_id fc_3 --lowpass_filter 3`


## Ablation study result
We ran the GRU model of each intialization configuration five times for 1500 steps, all other configurations in the *config.gin* are the same. For ablation study of bidrectional layers we use glorot_uniform as initialzation. The best result of 5 runs (total 12 classes performance) is reported here. Best validation checkpoints are stored and later on used for evaluation. You can have a look at our log files in `best_runs/` to see the detailed results of each run made for the chart below. Please note that the results may be hard to reproduce since the transition samples in dataset are too less and therefore the model performance of balanced accuracy varies greatly between different runs (we encounterd up to 5% difference of balanced accuracy for two runs to be quite normal). 

>- name with * is baseline model, which uses glorot_uniform, doesn't  bidrectional and lowpass filter.
>- in the cell of each table, there are two values, left value is maximum of 5 runs, right value with bracket is mean of 5 runs.

#### Initialization
| Initialization| Accuracy [%] | Balanced Accuracy [%] | Precision [%]  | Recall [%]| F1-Score [%] |
| :---: | :---: | :---: | :---: | :---: | :---: |
| he_normal | **97.33** (96.56)|  **92.24**  (90.07) |  **93.62** (91.32) | **92.24** (90.07)| **92.59** (90.42) |
| lecun_normal | 96.56 (93.11)| 91.79 (90.45)|  93.16 (89.11)|  91.79 (90.45)|  91.76 (88.60)|
| glorot_uniform<sup>*</sup> | 96.87 (95.20) | 92.22 (89.21) | 91.87 (90.40)  | 92.22 (89.21) | 91.61 (88.90) |

#### Bidrectional
| Bidrectional | Accuracy [%] | Balanced Accuracy [%] | Precision [%]  | Recall [%]| F1-Score [%] |
| :---: | :---: | :---: | :---: | :---: | :---: |
| True | 96.20 (95.77) | 92.10 (90.61)|  **93.84** (91.77) | 92.10 (90.61) | **92.81** (90.78)|
| False<sup>*</sup> | **96.87** (95.20) | **92.24** (89.21) | 91.87 (90.40)  | **92.24** (89.21) | 91.61 (88.90) |

#### Weight loss
| Weight loss| Accuracy [%] | Balanced Accuracy [%] | Precision [%]  | Recall [%]| F1-Score [%] |
| :---: | :---: | :---: | :---: | :---: | :---: |
| 1<sup>*</sup> |  96.87 (95.20) | 92.24 (89.21) | 91.87 (90.40)  | 92.24 (89.21) | 91.61 (88.90) |
| 2 | 97.02 (95.31)| 92.61 (88.93)|  **94.04** (91.11)|  92.61 (88.93)|  92.70 (89.60)|
| 5 | 96.71 (95.20) | 89.89 (89.11) | 91.45 (90.16)  | 89.89 (89.11) | 89.74 (88.30) |
| 10 | **97.38** (95.88) | 92.68 (90.93) | 93.88 (92.06)  | 92.68 (90.93) | **92.80** (91.37) |
| 20 | 97.33 (95.76) | **92.80** (91.47) | 93.84 (92.26)  | **92.80** (91.47) | **93.10** (91.67) |

#### Lowpassfilter
| F<sub>c</sub> | Accuracy [%] | Balanced Accuracy [%] | Precision [%]  | Recall [%]| F1-Score [%] |
| :---: | :---: | :---: | :---: | :---: | :---: |
| no filter<sup>*</sup> | 96.87 (95.20) | 92.24 (89.21) | 91.87 (90.40)  | 92.24 (89.21) | 91.61 (88.90) |
| 3 | **97.07** (96.67) | **94.65** (92.73) |  **95.73** (92.89)| **94.65** (92.73) | **94.81** (92.58)|
| 10 | 96.97 (95.20) | 93.71 (91.79) |  93.04 (91.49)| 93.71 (91.79) | 93.24 (90.52)|
| 15 | 95.74  (94.32)| 89.69 (88.46)| 90.18 (89.49)| 89.69 (88.46)| 89.87 (87.21)|
| 20 | 95.89 (94.61)| 90.30 (87.32)| 90.29 (86.23)| 90.30 (87.32)| 89.49 (84.62)|

#### Ablation visualization
![fc](https://github.tik.uni-stuttgart.de/iss/dl-lab-21w-team20/blob/master/human_activity_recognition/best_runs/ablation_fc.jpg)![weighted_loss](https://github.tik.uni-stuttgart.de/iss/dl-lab-21w-team20/blob/master/human_activity_recognition/best_runs/ablation_weighted_loss.jpg)

### Best model confusion matrix
<p align="center">
  <img src="https://github.tik.uni-stuttgart.de/iss/dl-lab-21w-team20/blob/master/human_activity_recognition/best_runs/lowpassfilter_fc_3/plot/confusion_matrix.png" />
</p>

## How small can the Window size become ?
We made several experiments on the influence of the window size. Our results are all stored in the `how_small` directory.
For our experiments we ran each configuration 15 times with our best GRU-based model and averaged the performance.
You can make your own experiences using:

```python
python3 how_small.py --start 5 --end 300 --step 10 --num 5
python3 how_small.py --start 5 --end 300 --step 10 --lowpass_filter 3 --num 5 
python3 how_small.py --start 5 --end 300 --step 10 --lowpass_filter 10 --num 5
python3 how_small.py --start 5 --end 300 --step 10 --lowpass_filter 15 --num 5
```
Our experimental results show that the **performance of the classifier can still be good for small window sizes** (especially if the data is smoothed beforehand).
<p align="center">
  <img src="https://github.tik.uni-stuttgart.de/iss/dl-lab-21w-team20/blob/master/human_activity_recognition/how_small/combined_plot.png" />
</p>

# Logging
Relevant data is loggend into the run directory. The logging directory is created at `runs/run_<date>`.
```
runs
└───run_<date>
    │└─>config_operative.gin    
    │───ckpts
    │   └─>*      
    │───logs
    │   |─>eval.log
    │   └─>train.log
    └───plot
        |─>colormap.png
        |─>confusion_matrix.png
        |─>frequency_comparison_acc.png (optional)
        |─>frequency_comparison_gyro.png (optional)
        |─>ground_truth_acc_visualization.png
        |─>ground_truth_gyro_visualization.png
        |─>predictions_acc_visualization.png
        |->ground_truth_acc_visualization.png
        └─>predictions_gyro_visualization.png 

```


## Weights & Biases
We mainly used wandb for **logging** and **hyperparameter optimization**.
```
diabetic_retinopathy
└───wandb
    |───>*
    └───*
```

>- Make sure you logged into your wandb-account on the device you are running.

To enable simple **logging** on wandb, pass the wandb flag.
```python
# Enable wandb logging for a run
python3 main.py --wandb 
```

For **hyperparameter optimization** you can use the `tune_wandb.py` file. All configurations for the sweep are stored in the `configs/sweep_configs.py` file and can easily be adapted.

```python
# Start a wandb sweep (login to your wandb account before)
# normal mode optimizees GRU/LSTM based models
python3 tune_wandb.py
# loss mode only optimizes the loss for transition actiities (GRU/LSTM)
python3 tune_wandb.py --mode loss
# tcn-mode sweep to optimize configuration of convolutional approach
python3 tune_wandb.py --mode tcn
```


## Tensorboard
To run Tensorboard  use `tensorboard --logdir logs/`. If you running locally the output can be watched in a browser.
```
diabetic_retinopathy
└───logs
    └───runs_<date>
        |───train
        |   └─>*  
        └───validation
            └─>*
```



