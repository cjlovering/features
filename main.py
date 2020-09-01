import itertools
import random

import numpy as np
import pandas as pd
import plac
import pytorch_lightning as pl
import sklearn.metrics as metrics
import spacy
import torch
import torch.nn as nn
import tqdm
import transformers
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
from spacy.util import minibatch
from torch.utils.data import DataLoader, random_split
from transformers import BertModel, BertTokenizer
from pytorch_lightning.callbacks.base import Callback

import wandb
from models import bert, t5


@plac.opt(
    "prop",
    "property name",
    choices=[
        "gap_length",
        "gap_lexical",
        "gap_isl",
        "gap_plural",
        "gap_tense",
        "npi",
        "sva",
        "sva_easy",
        "sva_hard",
        "sva_diff",
        "arg"
        # "_gap_lexical",
        # "gap_flexible",
        # "gap_scoping",
    ],
)
@plac.opt(
    "rate",
    type=float,
    help=(
        "This is the co-occurence rate between the counter examples and the labels"
        "We generate data for rates {0., 0.001, 0.01, 0.1, 0.5, 0.9, 0.99, 0.999, 1.0}."
        "We use a rate=-1. when the task is `probing` as a filler value"
        "but its not used or checked, so anything is fine."
    ),
)
@plac.opt("probe", "probing feature", choices=["strong", "weak", "n/a"], abbrev="prb")
@plac.opt("task", "which mode/task we're doing", choices=["probing", "finetune"])
@plac.opt(
    "model", "which model to use",
)
@plac.opt(
    "wandb_entity", "wandb entity. set WANDB_API_KEY (in script or bashrc) to use."
)
def main(
    prop="sva",
    rate=0,
    probe="strong",
    task="finetune",
    model="bert-base-uncased",
    wandb_entity="bert-syntax",
):
    """Trains and evaluates model.

    NOTE:
    * If `task` = finetune, then `probe` is ignored.
    * If `task` = probe, then `rate` is ignored.

    NOTE: Use the `properties.py` file to generate your data.
    """
    ## static hp
    batch_size = 64
    num_epochs = 1

    # Check 10% of the validation data every 1/10 epoch.
    # We shuffle the validation data so we get new examples.
    limit_val_batches = 0.1
    val_check_interval = 0.1

    ## constants
    if task == "finetune":
        # TODO: Fix elsewhere.
        if rate == 0:
            rate = int(0)
        title = f"{prop}_{task}_{rate}_{model}"
        path = f"{task}_{rate}"
    else:
        title = f"{prop}_{task}_{probe}_{model}"
        path = f"{task}_{probe}"

    # We use huggingface for transformer-based models and spacy for baseline models.
    # The models/pipelines use slightly different APIs.
    using_huggingface = "bert" in model or "t5" in model
    if using_huggingface:
        negative_label = 0
        positive_label = 1
    else:
        # NOTE: If you need more than two classes or use different positive labels
        # make sure to also update `prepare_labels_spacy`.
        negative_label = "0"
        positive_label = "1"

    if "t5" in model:
        # use "yes" / "no"
        label_col = "label_str"
    else:
        # use 0, 1
        label_col = "label"

    # NOTE: Set `entity` to your wandb username, and add a line
    # to your `.bashrc` (or whatever) exporting your wandb key.
    # `export WANDB_API_KEY=62831853071795864769252867665590057683943`.
    config = dict(prop=prop, rate=rate, probe=probe, task=task, model=model)
    wandb_logger = WandbLogger(entity=wandb_entity, project="features")
    wandb_logger.log_hyperparams(config)
    train_data, eval_data, test_data = load_data(
        prop, path, label_col, [positive_label, negative_label], using_huggingface
    )
    num_steps = (len(train_data) // batch_size) * num_epochs
    datamodule = DataModule(batch_size, train_data, eval_data, test_data)
    classifier = load_model(model, num_steps)
    lossauc = LossAuc()
    trainer = Trainer(
        gpus=1 if spacy.prefer_gpu() else 0,
        logger=wandb_logger,
        limit_train_batches=0.1,
        limit_val_batches=limit_val_batches,
        limit_test_batches=0.1,
        val_check_interval=val_check_interval,
        min_epochs=num_epochs,
        max_epochs=num_epochs,
        callbacks=[lossauc],
    )
    trainer.fit(classifier, datamodule)

    # Test
    test_result = trainer.test(datamodule=datamodule)[0]
    classifier.freeze()
    classifier.eval()
    if "bert" in model:
        # *bert produces logits.
        test_pred = classifier(test_data).argmax(1).cpu().numpy()
    else:
        # t5 produces words
        test_pred = classifier(test_data)

    test_df = pd.read_table(f"./properties/{prop}/test.tsv")
    test_df["pred"] = test_pred
    test_df.to_csv(
        f"results/raw/{title}.tsv", sep="\t", index=False,
    )

    # Additional evaluation.
    if task == "finetune":
        additional_results = finetune_evaluation(test_df, label_col)
    elif task == "probing":
        additional_results = compute_mdl(train_data, model, batch_size, num_epochs)

    # Save summary results.
    wandb_logger.log_metrics(
        {
            # NOTE: `loss_auc` is not tracked when finetuning.
            "val_loss_auc": lossauc.get(),
            **test_result,
            **additional_results,
        }
    )
    pd.DataFrame(
        [
            {
                # NOTE: `loss_auc` is not tracked when finetuning.
                "val_loss_auc": lossauc.get(),
                **test_result,
                **additional_results,
                **config,  # log results for easy post processing in pandas, etc.
            }
        ]
    ).to_csv(
        f"./results/stats/{title}.tsv", sep="\t", index=False,
    )


def prepare_labels_pytorch(labels):
    # Currently a no-op.
    return labels


def prepare_labels_spacy(labels, categories):
    """spacy uses a strange label format -- here we set up the labels,
    for that format: [{cats: {yes: bool, no: bool}} ...]

    Expected usage:
    > categories = ["0", "1"] # neg, pos labels.
    > labels = [0, 1...]
    > prepare_labels_spacy(labels, categories)
    [{"cats": {"0": True, "1": False}}, {"cats": {"0": False, "1": True}}...]
    """
    # NOTE: This is really awkward but itll work for now.
    # The labels will always come as binary labels for now (in the labels column)
    # but for spacy we have to map it to strings.
    return [{"cats": {c: str(y) == c for c in categories}} for y in labels]


def load_data(prop, path, label_col, categories, using_huggingface):
    """Load data from the IMDB dataset, splitting off a held-out set."""
    # SHUFFLE
    trn = (
        pd.read_table(f"./properties/{prop}/{path}_train.tsv")
        .sample(frac=1)
        .reset_index(drop=True)
    )
    val = (
        pd.read_table(f"./properties/{prop}/{path}_val.tsv")
        .sample(frac=1)
        .reset_index(drop=True)
    )
    # NO SHUFFLE (so we can re-align the results with the input data.)
    tst = pd.read_table(f"./properties/{prop}/test.tsv")

    # SPLIT & PREPARE
    if using_huggingface:
        trn_txt, trn_lbl = (
            trn.sentence.tolist(),
            prepare_labels_pytorch(trn[label_col].tolist()),
        )
        val_txt, val_lbl = (
            val.sentence.tolist(),
            prepare_labels_pytorch(val[label_col].tolist()),
        )
        tst_txt, tst_lbl = (
            tst.sentence.tolist(),
            prepare_labels_pytorch(tst[label_col].tolist()),
        )
    else:
        # using spacy
        trn_txt, trn_lbl = (
            trn.sentence.tolist(),
            prepare_labels_spacy(trn[label_col].tolist(), categories),
        )
        val_txt, val_lbl = (
            val.sentence.tolist(),
            prepare_labels_spacy(val[label_col].tolist(), categories),
        )
        tst_txt, tst_lbl = (
            tst.sentence.tolist(),
            prepare_labels_spacy(tst[label_col].tolist(), categories),
        )
    train_data = list(zip(trn_txt, trn_lbl))
    eval_data = list(zip(val_txt, val_lbl))
    test_data = list(zip(tst_txt, tst_lbl))
    print("train", len(train_data))
    print("val", len(eval_data))
    print("test", len(test_data))

    return train_data, eval_data, test_data


def evaluate(nlp, data, batch_size):
    """Evaluate model `nlp` over `data` with `batch_size`.
    """
    nlp.eval()
    with torch.no_grad():
        true = []
        pred = []
        logits = []
        for batch in minibatch(data, size=batch_size):
            texts, labels = zip(*batch)
            _logits = nlp(texts)
            pred.extend(_logits.argmax(1).cpu().tolist())
            true.extend(labels)
            logits.append(_logits)

        f_score = metrics.f1_score(pred, true)
        accuracy = metrics.accuracy_score(pred, true)
        precision = metrics.precision_score(pred, true)
        recall = metrics.recall_score(pred, true)
        loss = nn.functional.cross_entropy(torch.cat(logits), torch.tensor(true)).item()
    nlp.train()
    return (
        {
            "precision": precision,
            "recall": recall,
            "f_score": f_score,
            "accuracy": accuracy,
            "loss": loss,
        },
        pred,
    )


def evaluate_spacy(nlp, data, negative_label, positive_label, batch_size):
    """Evaluates a spacy textcat pipeline.
    """
    pred = []
    logits = []
    texts, labels = zip(*data)
    true = []
    for i, doc in enumerate(nlp.pipe(texts, batch_size=batch_size)):
        pred_yes = doc.cats[positive_label] > 0.5
        logits.append([doc.cats[negative_label], doc.cats[positive_label]])
        pred.append(1 if pred_yes else 0)
        true.append(1 if labels[i]["cats"][positive_label] else 0)
    f_score = metrics.f1_score(pred, true)
    accuracy = metrics.accuracy_score(pred, true)
    precision = metrics.precision_score(pred, true)
    recall = metrics.recall_score(pred, true)
    loss = nn.functional.cross_entropy(torch.tensor(logits), torch.tensor(true)).item()
    return (
        {
            "precision": precision,
            "recall": recall,
            "f_score": f_score,
            "accuracy": accuracy,
            "loss": loss,
        },
        pred,
    )


def load_model(model, num_steps):
    """Loads appropriate model & optimizer (& optionally lr scheduler.)
    """
    if "bert" in model:
        return bert.BertClassifier(model, num_steps)
    elif "t5" in model:
        return t5.T5Classifier(model, num_steps)


def finetune_evaluation(df, label_col):
    """Compute additional evaluation.

    1. Use `label` for the label.
    2. Use `section` and denote which of `{weak, strong, both, neither} hold.
    """
    df["error"] = df["pred"] != df[label_col]
    # For "weak_feature", we mean the `weak_feature` is present in the example.
    df["weak_feature"] = ((df.section == "both") | (df.section == "weak")).astype(int)
    both = df[df.section == "both"]
    neither = df[df.section == "neither"]
    strong = df[df.section == "strong"]
    weak = df[df.section == "weak"]

    # Here we use `label` as 1:1 map for the strong feature. This might not hold up
    # if we move to using composite strong features.
    I_pred_true = metrics.mutual_info_score(df[label_col], df["pred"])
    I_pred_weak = metrics.mutual_info_score(df["weak_feature"], df["pred"])
    error = lambda x: x["error"].mean()
    score = lambda x: 1 - x["error"].mean()
    return {
        "test-error": error(df),
        "both-error": error(both),
        "neither-error": error(neither),
        "strong-error": error(strong),
        "weak-error": error(weak),
        "test-accuracy": score(df),
        "both-accuracy": score(both),
        "neither-accuracy": score(neither),
        "strong-accuracy": score(strong),
        "weak-accuracy": score(weak),
        "I-pred-true": I_pred_true,
        "I-pred-weak": I_pred_weak,
    }


def random_split_partition(zipped_list, sizes):
    # NOTE: I'm getting some strange issues where the 0.1% doesn't have
    # two labels, thus it gets some bad errors. 0.1% = 0.001, for 2000 * 0.001 = 2,
    # so fair enough.
    # SOLUTION: The training data is shuffled and contains equal counts (or close enough)
    # of labels.
    random.shuffle(zipped_list)
    pos = [z for z in zipped_list if z[1] in {1, "1", "yes"}]
    neg = [z for z in zipped_list if z[1] not in {1, "1", "yes"}]
    interleaved_list = list(itertools.chain(*zip(pos, neg)))
    return [
        interleaved_list[end - length : end]
        for end, length in zip(itertools.accumulate(sizes), sizes)
    ]


def compute_mdl(train_data, model, batch_size, num_epochs):
    """Computes the Minimum Description Length (MDL) over the training data given the model.

    We use the prequential MDL.

    Voita, Elena, and Ivan Titov. "Information-Theoretic Probing with Minimum Description Length." 
    arXiv preprint arXiv:2003.12298 (2020). `https://arxiv.org/pdf/2003.12298`

    Parameters
    ----------
    ``train_data``: list of tuples of examples and labels.
    ``model``: A model string.
    """
    # NOTE: These aren't the split sizes, exactly; the first training size will be the first split size,
    # the second will be the concatenation of the first two, and so on. This is to take advantage
    # of the random_split function.
    split_proportions = np.array(
        [0.1, 0.1, 0.2, 0.4, 0.8, 1.6, 3.05, 6.25, 12.5, 25, 50]
    )
    split_sizes = np.ceil(0.01 * len(train_data) * split_proportions)

    # How much did we overshoot by? We'll just take this from the longest split
    extra = np.sum(split_sizes) - len(train_data)
    split_sizes[len(split_proportions) - 1] -= extra

    splits = random_split_partition(train_data, split_sizes.astype(int).tolist())
    mdls = []

    # Cost to transmit the first via a uniform code
    mdls.append(split_sizes[0])

    for i in tqdm.trange(len(splits), desc="mdl"):
        # If training on the last block, we test on all the data.
        # Otherwise, we train on the next split.
        last_block = i == (len(splits) - 1)

        # setup the train and test split.
        train_split = list(itertools.chain.from_iterable(splits[0 : i + 1]))
        test_split = train_split if last_block else splits[i + 1]

        # re-fresh model.
        datamodule = DataModule(batch_size, train_split, test_split, test_split)
        num_steps = (len(train_split) // batch_size) * num_epochs
        classifier = load_model(model, num_steps)
        trainer = Trainer(
            gpus=1 if spacy.prefer_gpu() else 0,
            limit_train_batches=1.0,
            limit_val_batches=1.0,
            limit_test_batches=1.0,
            val_check_interval=1.0,
            min_epochs=num_epochs,
            max_epochs=num_epochs,
        )
        trainer.fit(classifier, datamodule)

        # Test
        test_result = trainer.test(datamodule=datamodule)[0]
        test_loss = test_result["test_loss"]
        if not last_block:
            mdls.append(test_loss)

    total_mdl = np.sum(np.asarray(mdls))
    # the last test_loss is of the model trained and evaluated on the whole training data,
    # which is interpreted as the data_cost
    data_cost = test_loss
    model_cost = total_mdl - data_cost
    return {"total_mdl": total_mdl, "data_cost": data_cost, "model_cost": model_cost}


class DataModule(pl.LightningDataModule):
    def __init__(self, batch_size, train_data, eval_data, test_data):
        super().__init__()
        self.train_data = train_data
        self.eval_data = eval_data
        self.test_data = test_data
        self.batch_size = batch_size

    def train_dataloader(self):
        return DataLoader(self.train_data, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.eval_data, batch_size=self.batch_size, shuffle=True)

    def test_dataloader(self):
        return DataLoader(self.test_data, batch_size=self.batch_size, shuffle=False)


class LossAuc(Callback):
    def __init__(self):
        super().__init__()
        self.losses = []

    def on_validation_epoch_end(self, trainer, pl_module):
        if trainer.running_sanity_check:
            return
        self.losses.append(trainer.callback_metrics["val_loss"])

    def get(self):
        return sum(self.losses)


if __name__ == "__main__":
    plac.call(main)
