import random

import GPUtil
import numpy as np
import pandas as pd
import plac
import sklearn.metrics as metrics
import spacy
import torch
import torch.nn as nn
import tqdm
from spacy.util import compounding, minibatch
from spacy_transformers.util import cyclic_triangular_rate
from transformers import BertModel, BertTokenizer

import wandb


class BertClassifier(nn.Module):
    def __init__(self, tokenizer, encoder, hidden_size=768, num_classes=2):
        super(BertClassifier, self).__init__()
        self.tokenizer = tokenizer
        self.encoder = encoder
        self.classifier = nn.Linear(hidden_size, num_classes)

    def update(self, texts, labels, sgd):
        # This makes the probe compatible with a pipeline setup for spacy.
        logits = self.forward(texts)
        loss = nn.functional.cross_entropy(logits, labels)
        loss.backward()
        sgd.step()
        sgd.zero_grad()
        return logits.detach(), loss.item()

    def forward(self, texts):
        batch = torch.nn.utils.rnn.pad_sequence(
            [
                torch.tensor(self.tokenizer.encode(t, add_special_tokens=True))
                for t in texts
            ],
            batch_first=True,
        )
        encoded = self.encoder(batch)[1]
        logits = self.classifier(encoded)
        return logits


@plac.opt("prop", "property name", choices=["gap", "isl"])
@plac.opt("rate", "co occurence rate", choices=["0", "1", "5", "weak", "strong"])
@plac.opt("task", "which mode/task we're doing", choices=["probing", "finetune"])
@plac.opt(
    "model",
    "which model to use",
    choices=[
        "en_trf_xlnetbasecased_lg",
        "en_trf_bertbaseuncased_lg",
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
    model="en_trf_bertbaseuncased_lg",
    entity="cjlovering",
):
    label_col = "acceptable"
    negative_label = "no"
    positive_label = "yes"
    spacy.util.fix_random_seed(0)

    # NOTE: Set `entity` to your wandb username, and add a line
    # to your `.bashrc` (or whatever) exporting your wandb key.
    # `export WANDB_API_KEY=62831853071795864769252867665590057683943`.
    config = dict(prop=prop, rate=rate, task=task, model=model)
    wandb.init(entity=entity, project="features", config=config)

    # NOTE: Switch to `prefer_gpu` if you want to test things locally.
    is_using_gpu = spacy.prefer_gpu()
    if is_using_gpu:
        torch.set_default_tensor_type("torch.cuda.FloatTensor")
    (
        (train_texts, train_cats),
        (eval_texts, eval_cats),
        (test_texts, test_cats),
    ) = load_data(prop, rate, label_col, task, [positive_label, negative_label])

    # nlp = BertForSequenceClassification.from_pretrained(
    #     "bert-base-uncased", num_labels=2, hidden_dropout_prob=0
    # )
    # encoder = BertModel.from_pretrained("bert-base-uncased")
    # probe = nn.Linear(768, 2)

    # optimizer = transformers.AdamW(nlp.parameters())
    # tokenizer = BertTokenizer.from_pretrained("bert-base-cased")
    nlp = BertClassifier(
        BertTokenizer.from_pretrained("bert-base-cased"),
        BertModel.from_pretrained("bert-base-uncased"),
    )
    optimizer = torch.optim.Adam(nlp.parameters(), lr=2e-5)
    wandb.watch(nlp, log="all", log_freq=100)

    # exit()
    # wandb.watch(nlp, log="all")
    train_data = list(zip(train_texts, train_cats))
    eval_data = list(zip(eval_texts, eval_cats))
    test_data = list(zip(test_texts, test_cats))

    # wandb.watch(nlp, log='all')
    batch_size = 8
    positive_label = "yes"

    # # Initialize the TextCategorizer, and create an optimizer.
    # if model in {"en_trf_bertbaseuncased_lg", "en_trf_xlnetbasecased_lg"}:
    #     optimizer = nlp.resume_training()
    # else:
    #     optimizer = nlp.begin_training()

    # optimizer.alpha = 0.001
    # optimizer.weight = , weight_decay=0.005
    # learn_rates = cyclic_triangular_rate(
    #     learn_rate / 3, learn_rate * 3, 2 * len(train_data) // batch_size
    # )
    patience = 10
    num_epochs = 100
    loss_auc = 0
    best_val = np.Infinity
    best_epoch = 0
    last_epoch = 0
    for epoch in tqdm.trange(num_epochs, desc="epoch"):
        nlp.train()
        last_epoch = epoch
        random.shuffle(train_data)
        for batch in tqdm.tqdm(minibatch(train_data, size=batch_size), desc="batch"):
            texts, labels = zip(*batch)
            labels = torch.tensor(labels)
            logits, loss = nlp.update(texts, labels, sgd=optimizer)
            wandb.log(
                {
                    f"batch_loss": loss,
                    "batch_accuracy": metrics.accuracy_score(
                        logits.argmax(1).cpu().numpy(), labels.int().cpu().numpy()
                    ),
                }
            )

        val_scores, _ = evaluate(nlp, eval_data, positive_label, batch_size)
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
    test_scores, test_pred = evaluate(nlp, test_data, positive_label, batch_size)
    wandb.log({f"test_{k}": v for k, v in test_scores.items()})

    # # Save test predictions.
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


def prepare_labels_pytorch(labels, categories):
    return [int(y == "yes") for y in labels]


def prepare_labels_spacy(labels, categories):
    """spacy uses a strange label format -- here we set up the labels,
    for that format. [{yes: bool, no: bool} ...]
    """
    return [{c: y == c for c in categories} for y in labels]


def load_data(prop, rate, label_col, task, categories):
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
    trn_txt, trn_lbl = (
        trn.sentence.tolist(),
        prepare_labels_pytorch(trn[label_col].tolist(), categories),
    )
    val_txt, val_lbl = (
        val.sentence.tolist(),
        prepare_labels_pytorch(val[label_col].tolist(), categories),
    )
    tst_txt, tst_lbl = (
        tst.sentence.tolist(),
        prepare_labels_pytorch(tst[label_col].tolist(), categories),
    )
    return (trn_txt, trn_lbl), (val_txt, val_lbl), (tst_txt, tst_lbl)


def evaluate(nlp, data, positive_label, batch_size):
    with torch.no_grad():
        nlp.eval()
        positive_label = 1
        true = []
        pred = []
        logits = []
        for batch in tqdm.tqdm(minibatch(data, size=batch_size), desc="batch"):
            texts, labels = zip(*batch)
            _logits = nlp(texts)
            pred.extend(_logits.argmax(1))
            true.extend(labels)
            logits.append(_logits)

        f_score = metrics.f1_score(pred, true, pos_label=positive_label)
        accuracy = metrics.accuracy_score(pred, true)
        precision = metrics.precision_score(pred, true, pos_label=positive_label)
        recall = metrics.recall_score(pred, true, pos_label=positive_label)
        loss = nn.functional.cross_entropy(torch.cat(logits), torch.tensor(true)).item()

    nlp.train()
    return (
        {
            "precision": precision,
            "recall": recall,
            "f_score": f_score,
            "accuracy": accuracy,
            "loss": loss / len(true),
        },
        pred,
    )


def load_model(model):
    if model in {"en_trf_bertbaseuncased_lg", "en_trf_xlnetbasecased_lg"}:

        # nlp = spacy.load(model)
        # classifier = nlp.create_pipe(
        #     "trf_textcat",
        #     config={"exclusive_classes": True, "architecture": "softmax_class_vector"},
        # )
        # classifier.add_label("yes")
        # classifier.add_label("no")
        # nlp.add_pipe(classifier, last=True)
        return model, optimizer
    else:
        nlp = spacy.load("en_core_web_lg")
        classifier = nlp.create_pipe(
            "textcat", config={"exclusive_classes": True, "architecture": model},
        )
        classifier.add_label("yes")
        classifier.add_label("no")
        nlp.add_pipe(classifier, last=True)
        optimizer = nlp.begin_training()
        return nlp, optimizer


if __name__ == "__main__":
    plac.call(main)
