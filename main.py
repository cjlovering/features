import random

import GPUtil
import numpy as np
import pandas as pd
import plac
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
    choices=["en_trf_bertbaseuncased_lg", "bow", "simple_cnn"],
)
@plac.opt("entity", "wandb entity. set WANDB_API_KEY to use.")
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
    wandb.init(entity=entity, project="pytorch-spacy-transformers")
    spacy.util.fix_random_seed(0)
    is_using_gpu = spacy.prefer_gpu()
    if is_using_gpu:
        torch.set_default_tensor_type("torch.cuda.FloatTensor")
        GPUtil.showUtilization()
    (
        (train_texts, train_cats),
        (eval_texts, eval_cats),
        (test_texts, test_cats),
    ) = load_data(prop, rate, label_col, task, [positive_label, negative_label])
    nlp = load_model(model_choice)
    train_data = list(zip(train_texts, [{"cats": cats} for cats in train_cats]))

    batch_size = 8  # batch-szie changed to 4 to relieve pressure on GPU memory
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

    patience = 5
    num_epochs = 50
    loss_auc = 0
    best_val = np.Infinity
    best_epoch = 0
    last_epoch = 0

    for epoch in tqdm.trange(num_epochs, desc="epoch"):
        last_epoch = 0
        random.shuffle(train_data)
        batches = minibatch(train_data, size=batch_size)
        for batch in tqdm.tqdm(batches, desc="batch"):
            optimizer.trf_lr = next(learn_rates)
            texts, annotations = zip(*batch)
            nlp.update(texts, annotations, sgd=optimizer, drop=0.1)

        val_scores, _ = evaluate(nlp, eval_texts, eval_cats, positive_label, batch_size)
        val_loss = val_scores["avg_loss"]
        loss_auc += val_loss
        wandb.log({"trf_lr": optimizer.trf_lr, **val_scores})

        # Stop if no improvement in HP.patience checkpoints
        curr = min(val_loss, best_val)
        if curr < best_val:
            best_val = 0
            best_epoch = epoch
        elif (epoch - best_epoch) >= patience:
            break

    # Test the trained model
    test_scores, labels = evaluate(
        nlp, test_texts, test_cats, positive_label, batch_size
    )

    # Save test predictions.
    test_df = pd.read_table(f"./{prop}/{prop}_test.tsv")
    test_df["prediction"] = labels
    test_df.to_csv(
        f"results/{prop}_{rate}_{task}_{model_choice}_full.tsv", sep="\t", index=False,
    )

    # Save summary results.
    pd.DataFrame(
        [
            {
                "prop": prop,
                "rate": rate,
                "task": task,
                "model_choice": model_choice,
                "val_loss_auc": loss_auc,
                "best_val_loss": best_val,
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
    if task == "finetune":
        path = f"{prop}_finetune_{rate}"
    else:
        path = f"{prop}_probing_{rate}"
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
    tp = 0.0  # True positives
    fp = 0.0  # False positives
    fn = 0.0  # False negatives
    tn = 0.0  # True negatives
    total_words = sum(len(text.split()) for text in texts)
    loss = 0
    labels = []

    # TODO: Simplify this logic -- use sklearn?
    with tqdm.tqdm(total=total_words, leave=False) as pbar:
        for i, doc in enumerate(nlp.pipe(texts, batch_size=batch_size)):
            gold = cats[i]
            loss += -np.log(gold["yes"] * doc.cats["yes"] + gold["no"] * doc.cats["no"])
            for label, score in doc.cats.items():
                if label not in gold:
                    continue
                if label != positive_label:
                    continue
                labels.append("yes" if score > 0.5 else "no")

                if score >= 0.5 and gold[label] >= 0.5:
                    tp += 1.0
                elif score >= 0.5 and gold[label] < 0.5:
                    fp += 1.0
                elif score < 0.5 and gold[label] < 0.5:
                    tn += 1
                elif score < 0.5 and gold[label] >= 0.5:
                    fn += 1
            pbar.update(len(doc.text.split()))
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    accuracy = (tp + tn) / len(cats)
    if (precision + recall) == 0:
        f_score = 0.0
    else:
        f_score = 2 * (precision * recall) / (precision + recall)
    return (
        {
            "precision": precision,
            "recall": recall,
            "f_score": f_score,
            "accuracy": accuracy,
            "avg_loss": loss / len(cats),
        },
        labels,
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
        nlp = spacy.load("en")
        classifier = nlp.create_pipe(
            "textcat", config={"exclusive_classes": True, "architecture": model_choice},
        )
        classifier.add_label("yes")
        classifier.add_label("no")
        nlp.add_pipe(classifier, last=True)
        return nlp


if __name__ == "__main__":
    plac.call(main)
