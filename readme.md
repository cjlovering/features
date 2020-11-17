# features

Install requirements.

```bash
# Create new env.
conda create --name features python=3.8
conda activate features

# Install pytorch.
conda install pytorch torchvision cudatoolkit=10.2 -c pytorch

# Install further reqs.
pip install pytest  tqdm pandas gputil spacy[cuda102] transformers pytorch_lightning  pyinflect sklearn wandb nltk
pip install plac --upgrade
python -m spacy download en_core_web_lg
```

Set `wandb` subscription key in your `~/.bash_profile`.

```bash
# This is not the real key.
export WANDB_API_KEY=628318530717958647692528
```

Generate experiments & run!

```bash
# generate datasets
./setup.sh
# approx <30 min
sbatch datasets.sh
pytest test.py

# generate jobs
python job.py --experiment finetune
python job.py --experiment probing

# run jobs
sbatch jobs/[DATE]/jobs.sh
```

## Troubleshooting

If you have issues with `plac` (e.g. `plac.opt` is not defined) reinstall it with `pip install plac --upgrade`.

If you have issues with `cupy` uninstall (`pip uninstall cupy-cuda102`) and then re-install (`pip install cupy-cuda102`). 


# Stashed Settings

```
"lstm-toy",
"lstm-glove",
"t5-base",
"bert-base-uncased",
"gpt2",
"rebert"
```

```
[
        "gap-base-length",
        "gap-base-plural",
        "gap-hard-length",
        "gap-hard-none",
        "gap-hard-tense",
        "gap-base-lexical",
        "gap-base-tense",
        "gap-hard-lexical",
        "gap-hard-plural",
        "npi_lexical",
        "npi_plural",
        "npi_tense",
        "npi_length",
        "sva_base_agreement",
        "sva_base_lexical",
        "sva_base_plural",
        "sva_hard_agreement",
        "sva_hard_lexical",
        "sva_hard_length",
        "sva_hard_plural"
    ],
```

```
        "nyu-mll/roberta-base-1B-1",
        "nyu-mll/roberta-base-1B-2",
        "nyu-mll/roberta-base-1B-3",
        "nyu-mll/roberta-base-100M-1",
        "nyu-mll/roberta-base-100M-2",
        "nyu-mll/roberta-base-100M-3",
        "nyu-mll/roberta-base-10M-1",
        "nyu-mll/roberta-base-10M-2",
        "nyu-mll/roberta-base-10M-3",
        "nyu-mll/roberta-med-small-1M-1",
        "nyu-mll/roberta-med-small-1M-2",
        "nyu-mll/roberta-med-small-1M-3"
```

```
        "msgs-title_case_control",
        "msgs-syntactic_category_control",
        "msgs-absolute_token_position_control",
        "msgs-control_raising_control",
        "msgs-irregular_form_control",
        "msgs-lexical_content_the_control",
        "msgs-relative_position_control",
        "msgs-main_verb_control",
        "msgs-length_control"
```