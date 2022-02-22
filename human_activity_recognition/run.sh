#!/bin/bash -l
 
# Slurm parameters
#SBATCH --job-name=Group20
#SBATCH --output=Group20_sweep_all-%j.%N.out
#SBATCH --time=1-00:00:00
#SBATCH --gpus=1
 
# Activate everything you need
module load cuda/11.2

# Run our python code

# train, eval and visualize on our default model (gru)
python3 main.py

# train, eval and visualize on lstm
# python3 main.py --model_name lstm --model_id lstm

# train, eval and visualize on TCN
# python3 main.py --model_name tcn --model_id tcn

# ensemble
# python3 ensemble.py

# ablation for different initialization 
# python3 main.py --model_name gru --model_id glorot_uniform  --initialization glorot_uniform 
# python3 main.py --model_name gru --model_id he_normal  --initialization he_normal
# python3 main.py --model_name gru --model_id lecun_normal  --initialization lecun_normal

# ablation for bidirectional or not
# python3 main.py --model_name gru --model_id bidirectional  --bidirectional=True

# ablation for different lowpassfilter
# python3 main.py --model_name gru --model_id fc_3 --lowpass_filter 3
# python3 main.py --model_name gru --model_id fc_15 --lowpass_filter 15
# python3 main.py --model_name gru --model_id fc_20 --lowpass_filter 20

# ablation for different weight loss
# python3 main.py --model_name gru --model_id loss_weight_2 --loss_weight 2
# python3 main.py --model_name gru --model_id loss_weight_10 --loss_weight 10

# tune using wandb (login before) 
# python3 tune_wandb.py
# python3 tune_wandb.py --mode loss
# python3 tune_wandb.py --mode tcn

# Run how small for different cutoff frequencies 
# python3 how_small.py --start 5 --end 300 --step 10 --model_id how_small
# python3 how_small.py --start 5 --end 300 --step 10 --lowpass_filter 3 --model_id how_small_fc_3 
# python3 how_small.py --start 5 --end 300 --step 10 --lowpass_filter 10 --model_id how_small_fc_10
# python3 how_small.py --start 5 --end 300 --step 10 --lowpass_filter 15 --model_id how_small_fc_15
