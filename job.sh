#!/bin/sh

# Request half an hour of runtime:
#SBATCH --time=01:00:00

# Ask for the GPU partition and 1 GPU
#SBATCH -p gpu --gres=gpu:1

# Use more memory (4GB) (CPU RAM):
#SBATCH --mem=8G
#SBATCH --partition=gpu-he

# Specify a job name:
#SBATCH -J job

# Specify an output file
#SBATCH -o ./out/%j-0.out
#SBATCH -e ./err/%j-0.out
#### BATCH -a 2-2%10
module load python/3.7.4 gcc/8.3 cuda/10.0.130 cudnn/7.4
. /gpfs/runtime/opt/anaconda/3-5.2.0/etc/profile.d/conda.sh
conda activate feature_001

mkdir -p ./out/
mkdir -p ./err/
mkdir -p ./results/

nvidia-smi

python trash.py
# python main_bug.py 
# --rate 0 --prop gap --task finetune --model en_trf_bertbaseuncased_lg
echo "job finished."
