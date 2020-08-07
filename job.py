import datetime
import itertools
import json
import os

import plac


@plac.opt("experiment", "experiment name", choices=["probing", "finetune"])
def main(experiment="finetune"):
    output_dir = datetime.datetime.now().strftime(f"./output/{experiment}-%Y-%m-%d")
    jobs_dir = datetime.datetime.now().strftime(f"./jobs/{experiment}-%Y-%m-%d")

    if not os.path.exists("./output"):
        os.mkdir("./output")
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    if not os.path.exists("./jobs"):
        os.mkdir("./jobs")
    if not os.path.exists(jobs_dir):
        os.mkdir(jobs_dir)

    with open(f"./{experiment}.json", "r") as f:
        settings = json.load(f)
    options = list(itertools.product(*settings.values()))

    jobs = []
    for idx, option in enumerate(options):
        job_text = template_option(*option)
        job = setup(job_text, idx)
        jobs.append(job)

    jobs_file = template_file(jobs)
    with open(f"{jobs_dir}/jobs.sh", "w") as f:
        f.write(jobs_file)


def template_file(texts):
    text = "".join(texts)
    out = f"""#!/bin/sh

# Request half an hour of runtime:
#SBATCH --time=01:00:00

# Ask for the GPU partition and 1 GPU
#SBATCH -p gpu --gres=gpu:1

# Use more memory (8GB) and correct partition.
#SBATCH --mem=8G
#SBATCH --partition=gpu-he

# Specify a job name:
#SBATCH -J job

# Specify an output file
#SBATCH -o ./out/%j-0.out
#SBATCH -e ./err/%j-0.out
#SBATCH -a 0-{len(texts)}%10

module load python/3.7.4 gcc/8.3 cuda/10.2 cudnn/7.6.5
. /gpfs/runtime/opt/anaconda/3-5.2.0/etc/profile.d/conda.sh
conda activate features

mkdir -p ./out/
mkdir -p ./err/
mkdir -p ./results/

nvidia-smi

{text}
"""
    return out


def setup(text, index):
    return f"""
if [ "$SLURM_ARRAY_TASK_ID" -eq {index} ];
then
{text}
fi
"""


def template_option(
    prop, rate, task, model,
):
    """Generates the template for an a call to train.
    """

    out = f"""python main.py \
        --prop {prop} \
        --rate {rate} \
        --task {task} \
        --model {model}
"""
    return out


if __name__ == "__main__":
    plac.call(main)
