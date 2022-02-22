#!/bin/bash -l
 
# Slurm parameters
#SBATCH --job-name=Group20
#SBATCH --output=Test20-%j.%N.out
#SBATCH --time=1-00:00:00
#SBATCH --gpus=1
 
# Activate everything you need
module load cuda/11.2

## Run your python code

## Train, eval and visualize on our default model (team20_cnn_01)
python3 main.py
# use wandb login
# python3 main.py --wandb

# Differnet models to run
# python3 main.py --model_name cnn_01 --model_id cnn_01
# python3 main.py --model_name cnn_02 --model_id cnn_02
# python3 main.py --model_name resnet --model_id resnet
# python3 main.py --model_name cnn_blueprint --model_id cnn_blueprint
# python3 main.py --model_name vgg --model_id vgg
# python3 main.py --model_name xception --model_id xception
# python3 main.py --model_name inceptionresnets --model_id inceptionresnets

# run our ensemble method
# python3 ensemble.py

# Only train model
# python3 main.py --run train --model_name resnet

## Run from a checkpoint - first uncomment Trainer.ckpt in config.gin
# python3 main.py --run train --model_name cnn_02

# Only evaluate model
# python3 main.py --run eval --model_name cnn_blueprint  --model_id ./best_runs/cnn_bp

# create new visualization (make sure to add the waned images to the config.gin)
# python3 main.py --run visual --model_name cnn_01 --model_id ./best_runs/cnn_01


