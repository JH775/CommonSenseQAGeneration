#!/bin/bash
 
#SBATCH -n 4                        # number of cores
#SBATCH -t 0-24:00                  # wall time (D-HH:MM)
#SBATCH -A <ASURITE>             # Account hours will be pulled from (commented out with double # in front)
#SBATCH -o slurm.%j.out             # STDOUT (%j = JobId)
#SBATCH -e slurm.%j.err             # STDERR (%j = JobId)
#SBATCH --mail-type=ALL             # Send a notification when the job starts, stops, or fails
#SBATCH --mail-user=<ASURITE>@asu.edu # send-to address


python -m allennlp.run predict ./data/qasrl_parser_elmo.tar.gz ./input/input_nrlQA_CONTEXT.txt --include-package nrl --predictor qasrl_parser --output-file ./out_nrlQA.txt
