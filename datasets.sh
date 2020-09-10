#!/bin/sh

#SBATCH --time=01:00:00
#SBATCH --mem=16G
#SBATCH -J job
#SBATCH -o ./out/%j-0.out
#SBATCH -e ./err/%j-0.out
#SBATCH -a 0-9%4

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
python sva.py --template base --weak lexical
fi
if [ "$SLURM_ARRAY_TASK_ID" -eq 5 ];
then
python sva.py --template base --weak agreement
fi
if [ "$SLURM_ARRAY_TASK_ID" -eq 6 ];
then
<<<<<<< HEAD
python sva.py --template hard --weak lexical
fi
if [ "$SLURM_ARRAY_TASK_ID" -eq 7 ];
then
python sva.py --template hard --weak agreement
=======
python sva.py --template base --weak plural
fi
if [ "$SLURM_ARRAY_TASK_ID" -eq 7 ];
then
python sva.py --template hard --weak lexical
>>>>>>> da227d27c0313dac79521ea2b5ec53acf91764db
fi
if [ "$SLURM_ARRAY_TASK_ID" -eq 8 ];
then
python sva.py --template hard --weak agreement
fi
if [ "$SLURM_ARRAY_TASK_ID" -eq 9 ];
then
python sva.py --template hard --weak plural
fi
if [ "$SLURM_ARRAY_TASK_ID" -eq 10 ];
then
python sva.py --template hard --weak length
fi
if [ "$SLURM_ARRAY_TASK_ID" -eq 11 ];
then
python gap_plural.py
fi
if [ "$SLURM_ARRAY_TASK_ID" -eq 12 ];
then
python gap_tense.py
fi
echo "job finished in ${SECONDS}"
