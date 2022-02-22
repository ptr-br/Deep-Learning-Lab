# Diabetic Retinopathy

## Setup
There is not much to do from your side. We used relatives paths, so make sure to execute everything from this directory on the server. Most commands we mention in this description are also in the `run.sh`. You can simply uncomment the related lines and run it.
```bash
# On the server
git clone https://github.tik.uni-stuttgart.de/iss/dl-lab-21w-team20.git
cd dl-lab-21w-team20/diabetic_retinopathy
```

# How to run the code 
Executing the script, the default behavior is to first train the model for the steps specified in your *config.gin*. After training is finished, the model will be evaluated on the test set. Furthermore, the images choosen per configurtion will be visulaized and stored.
All three substeps (train | evaluate | visualize) can also be executed on their own.
## Getting started 
If the TFRecords folder does not exist yet, it will be automatically created and the data will be stored there.
To train your first model, you can simply run
```python
# This is running the script in default mode (team20_cnn_01 model - train, eval and visualize)
python3 main.py
```
## Training
By default, no training from checkpoint is performed. However, uncommnet the `Trainer.ckpt` in the gin file and specify the checkpoint you want to load. Make sure to watch the log output to see if model loading has worked (if model and weigts do not fit or there is no checkpoint available, training will start from scratch).

```python
# Training example (resnet model - no deepvis or eval performed)
python3 main.py --run train --model_name resnet

# Train one of our uploaded models from checkpoint 
# - remember to first uncomment "Trainer.ckpt" in config.gin file
python3 main.py --run train --model_name cnn_02

```

## Evaluation 
Evaluation is performed by default. You can have a look at eval.log to see the results. If by accident the file get's damaged or is lost you can re-run evaluation by setting the flag and giving the model path.

```python
# Evaluation example (cnn_blueprint - already trained)
python3 main.py --run eval --model_name cnn_blueprint --model_id  /home/RUS_CIP/st<YOUR_ST_NUMBER_HERE>/dl-lab-21w-team20/runs/run_<date>
```
>- make sure model_name and model_id are from the same type (here cnn_blueprint) 

## Deep Visualization
The images to be visualized can be set in *config.gin* by editing the parameter `visual.image_numbers = [1,12,13]`. This setup will create visualizations for the images *IDRID_001.jpg*, *IDRID_012.jpg*, *IDRID_013.jpg* of both test and training set. The output will be stored in the output folder (in the plot subdirectory).
One can also add new images to the output of an already trained model.

```python
# Visualization example (team20_cnn_02 model - already trained)
python3 main.py --run visual --model_name cnn_01 --model_id ./best_runs/cnn_01
```

>- make sure *model_name* and *model_id* are from the same type (here *cnn_2*) and have the same parameters in the config-file, otherwise loading will fail.
>- Deep Visualization only works properly with our 'normal' CNN archirtectures (team20_cnn_0x and cnn_blueprint)

#### Deep Visulaization Example 
![Alt text](https://github.tik.uni-stuttgart.de/iss/dl-lab-21w-team20/blob/master/diabetic_retinopathy/best_runs/team20_cnn_01/plot/train/deepvis_IDRiD_001.jpg?raw=true?raw=true)

# Performance
Please note that the results may be hard to reproduce since the dataset is small  and therefore the model performance varies greatly between different runs (we encounterd up to 15% difference for some models). We ran each model five times for 5000 steps (transfer models 700) and report the best result here. Best validation checkpoints are stored and later on used for evaluation. You can have a look at our log files in `best_runs/performance_logs/` to see the detailed results of each run made for the chart below.

|  | Accuracy [%] | Balanced Accuracy [%] | Sensitivity [%]  | Specificity [%]| F1-Score [%] |
| :---: | :---: | :---: | :---: | :---: | :---: |
| team20_cnn_01 | 81.6 |  80.3 |  73.8 | 86.9 | **86.8** |
| team20_cnn_02 | 80.6 | 79.4 |  74.3 |  84.4|  74.3 |
| cnn_blueprint | 78.6 | 77.3 | 71.8 | 82.8| 71.8 |
| resnet | 84.5 | 84.3 |  72.5 | 96.2 | 82.2 |
| vgg_like | 85.4 | 85.1 | 74.0 | 96.2| 83.1 |
| InceptionResnet | 81.6 | 80.4 | 72.7 | 88.1| 77.1 |
| Xception | 81.5 | 81.2 | 70.0 | 92.4| 78.7 |
| Voting (ensemble) | **87.4** | 86.7 | **77.1** | 96.4| 85.1 |
| Average (ensemble) | **87.4** | **87.1** | 76.0 | **98.1** | 85.4 |


## Team20_cnn_01
This is our default model. The script can be simply executed and the model will be trained.
To do so simply execute `python3 main.py`.
 
Since this is one of our static models, the architecure is fixed. This model has three convolutional layers, always followed by maxpooling and batch-normalization. To flatten the image, global average pooling is performed. Finally ther are two more dense layers with dopout in between.

## Team20_cnn_02
This model is quite similar to Team20_CNN_01, but a little bit larger. To run it on the server execute `python3 main.py --model_name cnn_02`.


## CNN_Blueprint
The cnn_blueprint mode is a scalable architecture. It first uses `n` cnn layers followed by `m` cnn_blocks (cnn_block is a self defined block consisting of a cnn layer followed by max-pooling and batch normalization). To run it on the server execute `python3 main.py --model_name cnn_blueprint`.


## Resnet
This model combines the known resnet blocks in a scalable architecture. To run it use `python3 main.py --model_name resnet`.


## VGG_Like
VGG-Like model from the MNIST-example. To run it use `python3 main.py --model_name vgg`


## Transfer Learning
All transfer models are connected to a dense layer, and a dropout layer is set before the output layer. Three hyperparameters were tuned, the num of trainable layers of transfer model, dense_units of the dense layer, and dropout rate. We used three differnet transfer models (InceptionResnet, Xception, EfficientNetB0), however, only the first two showed reasonable results.

```python

# Xception
python3 main.py --model_name xception

# InceptionResnet
python3 main.py --model_name inceptionresnet

```

## Ensemble Learning 
We employed voting and average methods for ensamble leaning respectively, voting method counts the classification of each model directly, while average method first gets the average prediction of applied models and then puts it into the metric. Both ensemble learning methods use our resnet, vgg_like and team20_cnn_01 models stored in `/best_runs`. By default, voting will be perfoemd. To get the average prediction, open the `ensemble.py` file and uncomment the `ENSEMBLE_METHOD` variable at the top of the file. To run our ensemble approach execute

```python
# Ensemble learnign of already trained models
python3 ensemble.py
```


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
        │
        └───test
        │   └─>*.png
        └───train
            └─>*.png
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


## Weights & Biases
We mainly used wandb for **logging** and **hyperparameter optimization**.
```
diabetic_retinopathy
└───wandb
    |───>*
    └───*
```


>- We are not sure if the following commnads will work out if you are not added to our weights and biases project. 
>- Make sure you logged into your wandb-account on the device you are running.

To enable simple **logging** on wandb, pass the wandb flag.
```python
# Enable wandb logging for a run
python3 main.py --wandb 
```

For **hyperparameter optimization** you can use the `tune_wandb.py` file. However, we did not implement a routine to handle different inputs. This means you need to set the model inside the file (e.g. `model = team20_cnn_01()`). Then you can create an agent for a sweep at the website of wandb and run it on the server. 
If you want to see some reports from our sweeps, you can contact us.







