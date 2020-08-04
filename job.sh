#!/bin/sh

# Request half an hour of runtime:
#SBATCH --time=00:30:00

# Ask for the GPU partition and 1 GPU
#SBATCH -p gpu --gres=gpu:1

# Use more memory (4GB) (CPU RAM):
#SBATCH --mem=8G
#### BATCH --partition=gpu-he

# Specify a job name:
#SBATCH -J job

# Specify an output file
#SBATCH -o ./out/%j-0.out
#SBATCH -e ./err/%j-0.out
#SBATCH -a 0-0%10
module load python/3.7.4 gcc/8.3 cuda/10.2 cudnn/7.6.5
. /gpfs/runtime/opt/anaconda/3-5.2.0/etc/profile.d/conda.sh
conda activate features

mkdir -p ./out/
mkdir -p ./err/
mkdir -p ./results/

echo "job started."
if [ "$SLURM_ARRAY_TASK_ID" -eq 0 ]
then
python main.py --rate 0 --prop gap --task finetune
fi
if [ "$SLURM_ARRAY_TASK_ID" -eq 1 ]
then
python main.py --rate 0 --prop isl --task finetune
fi
if [ "$SLURM_ARRAY_TASK_ID" -eq 2 ]
then
python main.py --rate 1 --prop isl --task finetune
fi
if [ "$SLURM_ARRAY_TASK_ID" -eq 3 ]
then
python main.py --rate 5 --prop isl --task finetune
fi
if [ "$SLURM_ARRAY_TASK_ID" -eq 4 ]
then
python main.py --rate weak --prop isl --task probing
fi
if [ "$SLURM_ARRAY_TASK_ID" -eq 5 ]
then
python main.py --rate strong --prop isl --task probing
fi

echo "job finished."
