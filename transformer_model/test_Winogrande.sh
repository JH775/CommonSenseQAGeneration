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
#models_bert/customMC_base
python ./run_multiple_choice.py --task_name  Winogrande --model_name_or_path models_bert/CustomMC_protoqa_base/albert  --do_predict  --data_dir ./data/Winogrande_MC --max_seq_length 100 --output_dir  models_bert/Winogrande_base/CustomMC_protoqa/albert --overwrite_output --tokenizer_name elgeish/cs224n-squad2.0-albert-base-v2


