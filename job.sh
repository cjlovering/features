#!/bin/sh

# Request half an hour of runtime:
#SBATCH --time=05:00:00

# Ask for the GPU partition and 1 GPU
#SBATCH -p gpu --gres=gpu:1

# Use more memory (4GB) (CPU RAM):
#SBATCH --mem=16G
#SBATCH --partition=gpu-he

# Specify a job name:
#SBATCH -J job

# Specify an output file
#SBATCH -o ./out/%j-0.out
#SBATCH -e ./err/%j-0.out

module load python/3.7.4 gcc/8.3 cuda/10.2 cudnn/7.6.5
. /gpfs/runtime/opt/anaconda/3-5.2.0/etc/profile.d/conda.sh
conda activate features

mkdir -p ./out/
mkdir -p ./err/
mkdir -p ./results/

echo "job started."
python main.py
echo "job finished."
