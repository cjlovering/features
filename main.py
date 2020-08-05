import random

import GPUtil
import numpy as np
import pandas as pd
import plac
import sklearn.metrics as metrics
import spacy
import torch
import tqdm
from spacy.util import minibatch
from spacy_transformers.util import cyclic_triangular_rate

import wandb


@plac.opt("prop", "property name", choices=["gap", "isl"])
@plac.opt("rate", "co occurence rate", choices=["0", "1", "5", "weak", "strong"])
@plac.opt("task", "which mode/task we're doing", choices=["probing", "finetune"])
@plac.opt(
    "model_choice",
    "which model to use",
    choices=["en_trf_bertbaseuncased_lg", "bow", "simple_cnn", "ensemble"],
)
@plac.opt("entity", "wandb entity. set WANDB_API_KEY (in script or bashrc) to use.")
def main(
    prop="gap",
    rate="0",
    task="finetune",
    model_choice="en_trf_bertbaseuncased_lg",
    entity="cjlovering",
):
    label_col = "acceptable"
    negative_label = "no"
    positive_label = "yes"
    spacy.util.fix_random_seed(0)

    # NOTE: Set `entity` to your wandb username, and add a line
    # to your `.bashrc` (or whatever) exporting your wandb key.
    # `export WANDB_API_KEY=62831853071795864769252867665590057683943`.
    config = dict(prop=prop, rate=rate, task=task, model_choice=model_choice)
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
    nlp = load_model(model_choice)
    train_data = list(zip(train_texts, [{"cats": cats} for cats in train_cats]))

    batch_size = 16
    learn_rate = 2e-5
    positive_label = "yes"

    # Initialize the TextCategorizer, and create an optimizer.
    if model_choice in {"en_trf_bertbaseuncased_lg"}:
        optimizer = nlp.resume_training()
    else:
        optimizer = nlp.begin_training()
    optimizer.alpha = 0.001
    optimizer.trf_weight_decay = 0.005
    optimizer.L2 = 0.0
    learn_rates = cyclic_triangular_rate(
        learn_rate / 3, learn_rate * 3, 2 * len(train_data) // batch_size
    )
    patience = 10
    num_epochs = 50
    loss_auc = 0
    best_val = np.Infinity
    best_epoch = 0
    last_epoch = 0

    for epoch in tqdm.trange(num_epochs, desc="epoch"):
        last_epoch = epoch
        random.shuffle(train_data)
        batches = minibatch(train_data, size=batch_size)
        for batch in tqdm.tqdm(batches, desc="batch"):
            optimizer.trf_lr = next(learn_rates)
            texts, annotations = zip(*batch)
            nlp.update(texts, annotations, sgd=optimizer, drop=0.1)

        val_scores, _ = evaluate(nlp, eval_texts, eval_cats, positive_label, batch_size)
        val_loss = val_scores["avg_loss"]
        loss_auc += val_loss
        wandb.log(val_scores)

        # Stop if no improvement in `patience` checkpoints.
        curr = min(val_loss, best_val)
        if curr < best_val:
            best_val = curr
            best_epoch = epoch
        elif (epoch - best_epoch) > patience:
            print(
                f"Early stopping: epoch {epoch}, best_epoch {best_epoch}, best val {best_val}."
            )
            break

    # Test the trained model
    test_scores, pred = evaluate(nlp, test_texts, test_cats, positive_label, batch_size)

    # Save test predictions.
    test_df = pd.read_table(f"./{prop}/{prop}_test.tsv")
    test_df["pred"] = pred
    test_df.to_csv(
        f"results/{prop}_{rate}_{task}_{model_choice}_full.tsv", sep="\t", index=False,
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
        f"results/{prop}_{rate}_{task}_{model_choice}_summary.tsv",
        sep="\t",
        index=False,
    )


def prepare_labels(labels, categories):
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
        prepare_labels(trn[label_col].tolist(), categories),
    )
    val_txt, val_lbl = (
        val.sentence.tolist(),
        prepare_labels(val[label_col].tolist(), categories),
    )
    tst_txt, tst_lbl = (
        tst.sentence.tolist(),
        prepare_labels(tst[label_col].tolist(), categories),
    )
    return (trn_txt, trn_lbl), (val_txt, val_lbl), (tst_txt, tst_lbl)


def evaluate(nlp, texts, cats, positive_label, batch_size):
    total_words = sum(len(text.split()) for text in texts)
    loss = 0
    pred = []
    with tqdm.tqdm(total=total_words, leave=False) as pbar:
        for i, doc in enumerate(nlp.pipe(texts, batch_size=batch_size)):
            gold = cats[i]
            loss += -np.log(gold["yes"] * doc.cats["yes"] + gold["no"] * doc.cats["no"])
            pred_yes = doc.cats["yes"] > 0.5
            pred.append("yes" if pred_yes else "no")
            pbar.update(len(doc.text.split()))
    true = ["yes" if c["yes"] else "no" for c in cats]
    f_score = metrics.f1_score(pred, true, pos_label=positive_label)
    accuracy = metrics.accuracy_score(pred, true)
    precision = metrics.precision_score(pred, true, pos_label=positive_label)
    recall = metrics.recall_score(pred, true, pos_label=positive_label)
    return (
        {
            "precision": precision,
            "recall": recall,
            "f_score": f_score,
            "accuracy": accuracy,
            "avg_loss": loss / len(cats),
        },
        pred,
    )


def load_model(model_choice):
    if model_choice in {"en_trf_bertbaseuncased_lg"}:
        nlp = spacy.load(model_choice)
        classifier = nlp.create_pipe(
            "trf_textcat",
            config={"exclusive_classes": True, "architecture": "softmax_class_vector"},
        )
        classifier.add_label("yes")
        classifier.add_label("no")
        nlp.add_pipe(classifier, last=True)
        return nlp
    else:
        nlp = spacy.load("en_core_web_lg")
        classifier = nlp.create_pipe(
            "textcat", config={"exclusive_classes": True, "architecture": model_choice},
        )
        classifier.add_label("yes")
        classifier.add_label("no")
        nlp.add_pipe(classifier, last=True)
        return nlp


if __name__ == "__main__":
    plac.call(main)
