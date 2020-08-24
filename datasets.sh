#!/bin/sh

#SBATCH --time=01:00:00
#SBATCH --mem=16G
#SBATCH -J job
#SBATCH -o ./out/%j-0.out
#SBATCH -e ./err/%j-0.out
#SBATCH -a 0-4%3

module load python/3.7.4 gcc/8.3
. /gpfs/runtime/opt/anaconda/3-5.2.0/etc/profile.d/conda.sh
conda activate features

echo "job started."
SECONDS=0;
if [ "$SLURM_ARRAY_TASK_ID" -eq 0 ];
then
python gap_lexical.py
fi
if [ "$SLURM_ARRAY_TASK_ID" -eq 1 ];
then
python gap_length.py
fi
if [ "$SLURM_ARRAY_TASK_ID" -eq 2 ];
then
python gap_isl.py
fi
if [ "$SLURM_ARRAY_TASK_ID" -eq 3 ];
then
python npi.py
fi
if [ "$SLURM_ARRAY_TASK_ID" -eq 4 ];
then
python sva.py
fi
echo "job finished in ${SECONDS}"
