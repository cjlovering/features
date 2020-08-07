# features

Install requirements.

```bash
# Create new env.
conda create --name features python=3.8
conda activate features

# Install pytorch.
conda install pytorch torchvision cudatoolkit=10.2 -c pytorch

# Install further reqs.
pip install tqdm pandas gputil spacy[cuda102] transformers plac pyinflect
python -m spacy download en_core_web_lg
```

Set `wandb` subscription key in your `.bash_profile`.

```bash
export WANDB_API_KEY=628318530717958647692528
```

Generate experiments & run!

```bash
# run!
python jobs.py --experiment finetune
python jobs.py --experiment finetune
sbatch jobs/[DATE]/jobs.sh
```

## Troubleshooting

If you have issues with `plac` (e.g. `plac.opt` is not defined) reinstall it.