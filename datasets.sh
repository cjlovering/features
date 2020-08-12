#!/bin/sh

#SBATCH --time=00:45:00
#SBATCH --mem=16G
#SBATCH -J job
#SBATCH -o ./out/%j-0.out
#SBATCH -e ./err/%j-0.out
#SBATCH -a 0-15

module load python/3.7.4 gcc/8.3
. /gpfs/runtime/opt/anaconda/3-5.2.0/etc/profile.d/conda.sh
conda activate features

echo "job started."
if [ "$SLURM_ARRAY_TASK_ID" -eq 0 ];
then
python gap.py --rate 0.1 --prop gap_lexical
fi
if [ "$SLURM_ARRAY_TASK_ID" -eq 1 ];
then
python gap.py --rate 0.1 --prop gap_flexible
fi
if [ "$SLURM_ARRAY_TASK_ID" -eq 2 ];
then
python gap.py --rate 0.1 --prop gap_scoping
fi
if [ "$SLURM_ARRAY_TASK_ID" -eq 3 ];
then
python gap.py --rate 0.1 --prop gap_isl
fi
if [ "$SLURM_ARRAY_TASK_ID" -eq 4 ];
then
python gap.py --rate 0.01 --prop gap_lexical
fi
if [ "$SLURM_ARRAY_TASK_ID" -eq 5 ];
then
python gap.py --rate 0.01 --prop gap_flexible
fi
if [ "$SLURM_ARRAY_TASK_ID" -eq 6 ];
then
python gap.py --rate 0.01 --prop gap_scoping
fi
if [ "$SLURM_ARRAY_TASK_ID" -eq 7 ];
then
python gap.py --rate 0.01 --prop gap_isl
fi
if [ "$SLURM_ARRAY_TASK_ID" -eq 8 ];
then
python gap.py --rate 0.001 --prop gap_lexical
fi
if [ "$SLURM_ARRAY_TASK_ID" -eq 9 ];
then
python gap.py --rate 0.001 --prop gap_flexible
fi
if [ "$SLURM_ARRAY_TASK_ID" -eq 10 ];
then
python gap.py --rate 0.001 --prop gap_scoping
fi
if [ "$SLURM_ARRAY_TASK_ID" -eq 11 ];
then
python gap.py --rate 0.001 --prop gap_isl
fi
if [ "$SLURM_ARRAY_TASK_ID" -eq 12 ];
then
python gap.py --rate 0.0 --prop gap_lexical
fi
if [ "$SLURM_ARRAY_TASK_ID" -eq 13 ];
then
python gap.py --rate 0.0 --prop gap_flexible
fi
if [ "$SLURM_ARRAY_TASK_ID" -eq 14 ];
then
python gap.py --rate 0.0 --prop gap_scoping
fi
if [ "$SLURM_ARRAY_TASK_ID" -eq 15 ];
then
python gap.py --rate 0.0 --prop gap_isl
fi
echo "job finished."
