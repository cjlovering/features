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


@plac.opt("prop", "property name", choices=["gap", "isl"])
@plac.opt("rate", "co occurence rate", choices=["0", "1", "5", "weak", "strong"])
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
@plac.opt("entity", "wandb entity. set WANDB_API_KEY (in script or bashrc) to use.")
def main(
    prop="gap",
    rate="0",
    task="finetune",
    model="bert-base-uncased",
    entity="cjlovering",
):
    label_col = "acceptable"
    spacy.util.fix_random_seed(0)
    # We use huggingface for transformer-based models and spacy for baseline models.
    # The models/pipelines use slightly different APIs.
    using_huggingface = "bert" in model
    negative_label = "no"
    positive_label = "yes"

    # NOTE: Set `entity` to your wandb username, and add a line
    # to your `.bashrc` (or whatever) exporting your wandb key.
    # `export WANDB_API_KEY=62831853071795864769252867665590057683943`.
    config = dict(prop=prop, rate=rate, task=task, model=model)
    wandb.init(entity=entity, project="features", config=config)

    is_using_gpu = spacy.prefer_gpu()
    if is_using_gpu:
        torch.set_default_tensor_type("torch.cuda.FloatTensor")
    (
        (train_texts, train_cats),
        (eval_texts, eval_cats),
        (test_texts, test_cats),
    ) = load_data(
        prop, rate, label_col, task, [positive_label, negative_label], using_huggingface
    )
    train_data = list(zip(train_texts, train_cats))
    eval_data = list(zip(eval_texts, eval_cats))
    test_data = list(zip(test_texts, test_cats))

    batch_size = 64
    num_epochs = 50
    num_steps = (len(train_cats) // batch_size) * num_epochs
    nlp, optimizer, scheduler = load_model(model, num_steps, using_huggingface)
    if using_huggingface:
        # TODO: spacy nlp model does is not directly a pytorch model.
        # it should be possible to extract the relevant parts of the pipeline.
        wandb.watch(nlp, log="all", log_freq=100)

    patience = 10
    loss_auc = 0
    best_val = np.Infinity
    best_epoch = 0
    last_epoch = 0
    for epoch in tqdm.trange(num_epochs, desc="epoch"):
        if using_huggingface:
            nlp.train()
        last_epoch = epoch
        random.shuffle(train_data)
        for batch in tqdm.tqdm(minibatch(train_data, size=batch_size), desc="batch"):
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
        elif (epoch - best_epoch) > patience:
            print(
                f"Early stopping: epoch {epoch}, best_epoch {best_epoch}, best val {best_val}."
            )
            break

    # Test the trained model
    if using_huggingface:
        test_scores, test_pred = evaluate(nlp, test_data, batch_size)
    else:
        test_scores, test_pred = evaluate_spacy(
            nlp, test_data, negative_label, positive_label, batch_size
        )
    wandb.log({f"test_{k}": v for k, v in test_scores.items()})

    # Save test predictions.
    test_df = pd.read_table(f"./{prop}/{prop}_test.tsv")
    test_df["pred"] = test_pred
    test_df.to_csv(
        f"results/{prop}_{rate}_{task}_{model}_full.tsv", sep="\t", index=False,
    )

    # Save summary results.
    wandb.log(
        {
            "val_loss_auc": loss_auc,
            "best_val_loss": best_val,
            "best_epoch": best_epoch,
            "last_epoch": last_epoch,
            **{f"test_{k}": v for k, v in test_scores.items()},
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
            }
        ]
    ).to_csv(
        f"results/{prop}_{rate}_{task}_{model}_summary.tsv", sep="\t", index=False,
    )


def prepare_labels_pytorch(labels):
    return [int(y == "yes") for y in labels]


def prepare_labels_spacy(labels, categories):
    """spacy uses a strange label format -- here we set up the labels,
    for that format. [{cats: {yes: bool, no: bool}} ...]
    """
    return [{"cats": {c: y == c for c in categories}} for y in labels]


def load_data(prop, rate, label_col, task, categories, using_huggingface):
    """Load data from the IMDB dataset, splitting off a held-out set."""
    # SHUFFLE
    path = f"{prop}_{task}_{rate}"
    trn = (
        pd.read_table(f"./{prop}/{path}_train.tsv")
        .sample(frac=1)
        .reset_index(drop=True)
    )
    val = (
        pd.read_table(f"./{prop}/{path}_val.tsv").sample(frac=1).reset_index(drop=True)
    )

    # NO SHUFFLE (so we can re-align the results with the input data.)
    tst = pd.read_table(f"./{prop}/{prop}_test.tsv")

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
    nlp.eval()
    with torch.no_grad():
        true = []
        pred = []
        logits = []
        for batch in tqdm.tqdm(minibatch(data, size=batch_size), desc="batch"):
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
        gold = labels[i]
        pred_yes = doc.cats[positive_label] > 0.5
        logits.append([doc.cats[negative_label], doc.cats[positive_label]])
        pred.append(1 if pred_yes else 0)
        true.append(1 if gold["cats"][positive_label] else 0)
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


def load_model(model, num_steps, using_huggingface):
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
        return nlp, optimizer, scheduler
    else:
        nlp = spacy.load("en_core_web_lg")
        classifier = nlp.create_pipe(
            "textcat", config={"exclusive_classes": True, "architecture": model},
        )
        classifier.add_label("yes")
        classifier.add_label("no")
        nlp.add_pipe(classifier, last=True)
        optimizer = nlp.begin_training()
        return nlp, optimizer, None


if __name__ == "__main__":
    plac.call(main)
