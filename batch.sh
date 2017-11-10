#!/bin/bash
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=4096

module add cuda/8.0
module add cudnn/7-cuda-8.0

./run.sh

