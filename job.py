import datetime
import itertools
import json
import os

import plac


@plac.opt(
    "experiment",
    "experiment name",
    choices=[
        "probing",
        "finetune",
        "npi_finetune",
        "npi_probing",
        "sva_finetune",
        "sva_probing",
        "arg_probing",
    ],
)
def main(experiment="finetune"):
    if not os.path.exists("./jobs"):
        os.mkdir("./jobs")

    with open(f"./{experiment}.json", "r") as f:
        settings = json.load(f)
    options = list(itertools.product(*settings.values()))

    jobs = []
    for idx, option in enumerate(options):
        job_text = template_option(*option)
        job = setup(job_text, idx)
        jobs.append(job)

    jobs_file = template_file(jobs, experiment)
    jobs_name = datetime.datetime.now().strftime(f"{experiment}-%Y-%m-%d")
    with open(f"./jobs/{jobs_name}.sh", "w") as f:
        f.write(jobs_file)


def template_file(texts, experiment):
    text = "".join(texts)
    out = f"""#!/bin/sh

# Request half an hour of runtime:
#SBATCH --time=03:00:00
#SBATCH -p gpu-he --gres=gpu:1
#SBATCH --mem=32G
#SBATCH -J {experiment}

# Specify an output file
#SBATCH -o ./out/{experiment}-%j.out
#SBATCH -e ./err/{experiment}-%j.out
#SBATCH -a 0-{len(texts)}

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
{text}fi
"""


def template_option(
    prop, rate, probe, task, model,
):
    """Generates the template for an a call to train.
    """

    out = f"""python main.py \
        --prop {prop} \
        --rate {rate} \
        --probe {probe} \
        --task {task} \
        --model {model}
"""
    return out


if __name__ == "__main__":
    plac.call(main)
