# features

Setup torch, pandas, numpy:

```bash
# Create new env.
conda create --name features python=3.8
conda activate features

# Install pytorch.
conda install pytorch torchvision cudatoolkit=10.2 -c pytorch

# Install further reqs.
pip install tqdm pandas gputil spacy spacy-transformers[cuda102] plac pyinflect
python -m spacy download en_trf_bertbaseuncased_lg
python -m spacy download en_core_web_lg
```