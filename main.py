import itertools
import random

import numpy as np
import pandas as pd
import plac
import sklearn.metrics as metrics
import spacy
import torch
import torch.nn as nn
import tqdm
import transformers
from spacy.util import minibatch
from transformers import BertModel, BertTokenizer

import wandb


@plac.opt(
    "prop",
    "property name",
    choices=["gap_lexical", "gap_flexible", "gap_scoping", "gap_isl", "npi", "sva"],
)
@plac.opt(
    "rate",
    type=float,
    help=(
        "This is the co-occurence rate between the counter examples and the labels"
        "We generate data for rates {0., 0.001, 0.01, 0.1}."
        "We use a rate=-1. when the task is `probing` as a filler value"
        "but its not used or checked, so anything is fine."
    ),
)
@plac.opt("probe", "probing feature", choices=["strong", "weak", "n/a"], abbrev="prb")
@plac.opt("task", "which mode/task we're doing", choices=["probing", "finetune"])
@plac.opt(
    "model",
    "which model to use",
    choices=[
        "bert-base-uncased",
        "bert-large-uncased",
        "bow",
        "simple_cnn",
        "ensemble",
    ],
)
@plac.opt(
    "wandb_entity", "wandb entity. set WANDB_API_KEY (in script or bashrc) to use."
)
def main(
    prop="gap",
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
    num_epochs = 25

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
    label_col = "label"
    # We use huggingface for transformer-based models and spacy for baseline models.
    # The models/pipelines use slightly different APIs.
    using_huggingface = "bert" in model
    if using_huggingface:
        negative_label = 0
        positive_label = 1
    else:
        # NOTE: If you need more than two classes or use different positive labels
        # make sure to also update `prepare_labels_spacy`.
        negative_label = "0"
        positive_label = "1"

    ## configuration

    # NOTE: Set `entity` to your wandb username, and add a line
    # to your `.bashrc` (or whatever) exporting your wandb key.
    # `export WANDB_API_KEY=62831853071795864769252867665590057683943`.
    config = dict(prop=prop, rate=rate, task=task, model=model, probe=probe)
    wandb.init(entity=wandb_entity, project="features", config=config)
    spacy.util.fix_random_seed(0)

    is_using_gpu = spacy.prefer_gpu()
    if is_using_gpu:
        torch.set_default_tensor_type("torch.cuda.FloatTensor")
    (
        (train_texts, train_cats),
        (eval_texts, eval_cats),
        (test_texts, test_cats),
    ) = load_data(
        prop, path, label_col, [positive_label, negative_label], using_huggingface
    )
    train_data = list(zip(train_texts, train_cats))
    eval_data = list(zip(eval_texts, eval_cats))
    test_data = list(zip(test_texts, test_cats))

    num_steps = (len(train_cats) // batch_size) * num_epochs
    nlp, optimizer, scheduler = load_model(
        model, num_steps, using_huggingface, positive_label, negative_label
    )
    if using_huggingface:
        # TODO: spacy nlp model does is not directly a pytorch model.
        # it should be possible to extract the relevant parts of the pipeline.
        wandb.watch(nlp, log="all", log_freq=1000)

    loss_auc = 0
    best_val = np.Infinity
    best_epoch = 0
    last_epoch = 0
    for epoch in tqdm.trange(num_epochs, desc="training"):
        last_epoch = epoch
        random.shuffle(train_data)
        for batch in minibatch(train_data, size=batch_size):
            texts, labels = zip(*batch)
            nlp.update(texts, labels, sgd=optimizer)
            if scheduler is not None:
                scheduler.step()

        if using_huggingface:
            val_scores, _ = evaluate(nlp, eval_data, batch_size)
        else:
            val_scores, _ = evaluate_spacy(
                nlp, eval_data, negative_label, positive_label, batch_size,
            )
        wandb.log({f"val_{k}": v for k, v in val_scores.items()})
        loss_auc += val_scores["loss"]

        # Stop if no improvement in `patience` checkpoints.
        curr = min(val_scores["loss"], best_val)
        if curr < best_val:
            best_val = curr
            best_epoch = epoch
        # We do not want to early-stop. We'll still track when the model does best.
        # 1) For probing this messes up the loss auc, and we would have to do
        # some additional post-processing to make it comparable.
        # 2) For fine-tuning, due to the limited data (as of now), there is overlap
        # between val & test. This is fine as long as we don't use val for early
        # stopping (or anything that will impact the model.)

    # Test the trained model
    if using_huggingface:
        test_scores, test_pred = evaluate(nlp, test_data, batch_size)
    else:
        test_scores, test_pred = evaluate_spacy(
            nlp, test_data, negative_label, positive_label, batch_size
        )
    wandb.log({f"test_{k}": v for k, v in test_scores.items()})

    # Save test predictions.
    test_df = pd.read_table(f"./properties/{prop}/test.tsv")
    test_df["pred"] = test_pred
    test_df.to_csv(
        f"results/raw/{title}.tsv", sep="\t", index=False,
    )

    # Additional evaluation.
    if task == "finetune":
        additional_results = finetune_evaluation(test_df)
    elif task == "probing":
        additional_results = compute_mdl(
            train_data,
            model,
            num_steps,
            using_huggingface,
            positive_label,
            negative_label,
            num_epochs,
            batch_size,
        )

    # Save summary results.
    wandb.log(
        {
            "val_loss_auc": loss_auc,
            "best_val_loss": best_val,
            "best_epoch": best_epoch,
            "last_epoch": last_epoch,
            **{f"test_{k}": v for k, v in test_scores.items()},
            **additional_results,
        }
    )
    pd.DataFrame(
        [
            {
                "val_loss_auc": loss_auc,
                "best_val_loss": best_val,
                "best_epoch": best_epoch,
                "last_epoch": last_epoch,
                **test_scores,
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
        return (trn_txt, trn_lbl), (val_txt, val_lbl), (tst_txt, tst_lbl)
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
        return (trn_txt, trn_lbl), (val_txt, val_lbl), (tst_txt, tst_lbl)


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


def load_model(model, num_steps, using_huggingface, positive_label, negative_label):
    """Loads appropriate model & optimizer (& optionally lr scheduler.)
    """
    if using_huggingface:
        nlp = BertClassifier(
            BertTokenizer.from_pretrained(model), BertModel.from_pretrained(model),
        )
        optimizer = transformers.AdamW(nlp.parameters(), lr=2e-5)
        scheduler = transformers.get_cosine_schedule_with_warmup(
            optimizer, 0.1 * num_steps, num_steps
        )
        nlp.train()
        return nlp, optimizer, scheduler
    else:
        nlp = spacy.load("en_core_web_lg")
        classifier = nlp.create_pipe(
            "textcat", config={"exclusive_classes": True, "architecture": model},
        )
        classifier.add_label(positive_label)
        classifier.add_label(negative_label)
        nlp.add_pipe(classifier, last=True)
        optimizer = nlp.begin_training()
        return nlp, optimizer, None


def finetune_evaluation(df):
    """Compute additional evaluation.

    1. Use `label` for the label.
    2. Use `section` and denote which of `{weak, strong, both, neither} hold.
    """
    df["error"] = df["pred"] != df["label"]
    df["weak"] = ((df.section == "both") | (df.section == "weak")).astype(int)
    both = df[df.section == "both"]
    neither = df[df.section == "neither"]
    strong = df[df.section == "strong"]
    weak = df[df.section == "weak"]

    I_pred_true = metrics.mutual_info_score(df["label"], df["pred"])
    I_pred_weak = metrics.mutual_info_score(df["weak"], df["pred"])
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


def compute_mdl(
    train_data,
    model,
    num_steps,
    using_huggingface,
    positive_label,
    negative_label,
    num_epochs,
    batch_size,
):

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

    random.shuffle(train_data)
    splits = torch.utils.data.random_split(train_data, split_sizes.astype(int).tolist())
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
        nlp, optimizer, scheduler = load_model(
            model, num_steps, using_huggingface, positive_label, negative_label
        )

        # train model on splits 0 to i (inclusive).
        for _ in range(num_epochs):
            random.shuffle(train_split)
            for batch in minibatch(train_split, size=batch_size):
                texts, labels = zip(*batch)
                nlp.update(texts, labels, sgd=optimizer)
                if scheduler is not None:
                    scheduler.step()

        # test the trained model
        if using_huggingface:
            test_scores, _ = evaluate(nlp, test_split, batch_size)
        else:
            test_scores, _ = evaluate_spacy(
                nlp, test_split, negative_label, positive_label, batch_size
            )
        test_loss = test_scores["loss"]
        if not last_block:
            mdls.append(test_loss)

    total_mdl = np.sum(np.asarray(mdls))
    # the last test_loss is of the model trained and evaluated on the whole training data,
    # which is interpreted as the data_cost
    data_cost = test_loss
    model_cost = total_mdl - data_cost
    return {"total_mdl": total_mdl, "data_cost": data_cost, "model_cost": model_cost}


class BertClassifier(nn.Module):
    def __init__(self, tokenizer, encoder, hidden_size=768, num_classes=2):
        super(BertClassifier, self).__init__()
        # TODO: make `hidden_size` contigent on the encoder.
        # `bert-large-*` has a bigger hidden_size.
        self.tokenizer = tokenizer
        self.encoder = encoder
        self.classifier = nn.Linear(hidden_size, num_classes)

    def update(self, texts, labels, sgd):
        """Performs a forward+backward sweep, including optimizer step.
        """
        # This makes the probe compatible with a pipeline setup for spacy.
        labels = torch.tensor(labels)
        logits = self.forward(texts)
        loss = nn.functional.cross_entropy(logits, labels)
        loss.backward()
        sgd.step()
        sgd.zero_grad()
        return logits.detach(), loss.item()

    def forward(self, texts):
        # TODO: `BertTokenizer` ought to pad by default, but was not working.
        batch = torch.nn.utils.rnn.pad_sequence(
            [
                torch.tensor(self.tokenizer.encode(t, add_special_tokens=True))
                for t in texts
            ],
            batch_first=True,
        )
        encoded = self.encoder(batch)[1]
        # TODO: Introduce a commandline arg for freezing bert.
        logits = self.classifier(encoded)
        return logits


if __name__ == "__main__":
    plac.call(main)
