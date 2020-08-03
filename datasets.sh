#!/bin/sh

# Request half an hour of runtime:
#SBATCH --time=00:20:00

# Use more memory (4GB) (CPU RAM):
#SBATCH --mem=16G

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

echo "job started."
python isl.py
python gap.py
echo "job finished."
