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
#SBATCH -a 0-1

module load python/3.7.4 gcc/8.3
. /gpfs/runtime/opt/anaconda/3-5.2.0/etc/profile.d/conda.sh
conda activate features

mkdir -p ./out/
mkdir -p ./err/
mkdir -p ./jobs/

echo "job started."
if [ "$SLURM_ARRAY_TASK_ID" -eq 0 ];
then
python isl.py
fi

if [ "$SLURM_ARRAY_TASK_ID" -eq 1 ];
then
python gap.py
fi
echo "job finished."
