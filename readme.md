# features

Setup torch, pandas, numpy:

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

## Troubleshooting

If you have issues with `plac` (e.g. `plac.opt` is not defined) or you don't see GPU utilization in the logs, uninstall and reinstall the libraries.