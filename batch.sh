#!/bin/bash
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=2048
#SBATCH --output=slurm/%j.out

module add cuda/8.0
module add cudnn/7-cuda-8.0

./run.sh

