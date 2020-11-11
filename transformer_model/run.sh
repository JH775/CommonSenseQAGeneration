#!/bin/sh

#SBATCH -n 1                  # number of cores
#SBATCH -t 0-24:00                  # wall time (D-HH:MM)
#SBATCH -p gpu 
#SBATCH -q wildfire                 # Run job under wildfire QOS queue

#SBATCH --gres=gpu:2                # Request two GPUs
 
#SBATCH -A jrosenke             # Account hours will be pulled from (commented out with double # in front)
#SBATCH -o slurm.%j.out             # STDOUT (%j = JobId)
#SBATCH -e slurm.%j.err             # STDERR (%j = JobId)
#SBATCH --mail-type=ALL             # Send a notification when the job starts, stops, or fails
#SBATCH --mail-user=jrosenke@asu.edu # send-to address

#module load anaconda/py3
#source activate transformers2

python ./run_multiple_choice.py --task_name swag --model_name_or_path roberta-base --do_train --do_eval --data_dir ./swag --learning_rate 5e-5 --num_train_epochs 3 --max_seq_length 80 --output_dir models_bert/swag_base --per_gpu_eval_batch_size=16 --per_device_train_batch_size=16 --gradient_accumulation_steps 2 --overwrite_output
