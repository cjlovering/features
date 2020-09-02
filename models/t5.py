import argparse
import glob
import json
import logging
import os
import random
import re
import time
from string import punctuation
import torch.nn as nn
import pytorch_lightning.metrics.functional as metrics
import itertools
import nltk
import numpy as np
import pandas as pd
import torch
from nltk.tokenize import sent_tokenize
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AdamW,
    T5ForConditionalGeneration,
    T5Tokenizer,
    get_linear_schedule_with_warmup,
    get_cosine_schedule_with_warmup,
    GPT2Model,
)
import sklearn.metrics as sk_metrics

import pytorch_lightning as pl

nltk.download("punkt")


class T5Classifier(pl.LightningModule):
    def __init__(self, model, num_steps):
        super(T5Classifier, self).__init__()
        self.model = T5ForConditionalGeneration.from_pretrained(model)
        self.tokenizer = T5Tokenizer.from_pretrained(model)
        self.num_steps = num_steps

    def step(self, batch):
        print(batch)
        texts, labels = batch
        texts = [format_input(t) for t in texts]
        input_ids = self.tokenizer.batch_encode_plus(
            texts, padding=True, return_tensors="pt", max_length=64
        )
        outputs = self.model(
            input_ids=input_ids["input_ids"],
            labels=labels.unsqueeze(1),
            attention_mask=input_ids["attention_mask"],
        )
        # LM LOSS
        loss = outputs[0]
        logits = outputs[1]
        return loss, logits

    def forward(self, batch):
        """This is used for inference. """
        _, logits = self.step(batch)
        return logits

    def configure_optimizers(self):
        "Prepare optimizer and schedule (linear warmup and decay)"

        model = self.model
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p
                    for n, p in model.named_parameters()
                    if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
            {
                "params": [
                    p
                    for n, p in model.named_parameters()
                    if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=1e-4,)
        scheduler = get_linear_schedule_with_warmup(
            optimizer, 0.1 * self.num_steps, self.num_steps
        )
        return [optimizer], [scheduler]

    def training_step(self, batch, batch_idx):
        # batch is tokenized.
        loss, _ = self.step(batch)
        return {"loss": loss}

    def training_epoch_end(self, outputs):
        training_loss = sum([x["loss"] for x in outputs])
        return {"loss": training_loss, "log": {"train_loss": training_loss}}

    def validation_step(self, batch, batch_idx):
        # This is bad /:
        loss, logits = self.step(batch)
        texts, labels = batch
        return {"val_loss": loss, "pred": logits.argmax(1), "true": labels}

    def validation_epoch_end(self, outputs):
        val_loss = sum([x["val_loss"] for x in outputs])
        pred = torch.cat([x["pred"] for x in outputs])
        true = torch.cat([x["true"] for x in outputs])
        f_score = metrics.f1_score(pred, true)
        accuracy = metrics.accuracy(pred, true)
        # f_score = sk_metrics.f1_score(pred, true, average="macro")
        # accuracy = sk_metrics.accuracy_score(pred, true)
        out = {
            "val_loss": val_loss,
            "val_f_score": f_score,
            "val_accuracy": accuracy,
            "log": {
                "val_loss": val_loss,
                "val_f_score": f_score,
                "val_accuracy": accuracy,
            },
        }
        return out

    def test_step(self, batch, batch_idx):
        _, labels = batch
        loss, logits = self.step(batch)
        # pred = self.model.generate(
        #     input_ids=batch["source_ids"],
        #     attention_mask=batch["source_mask"],
        #     max_length=2,
        # )
        # pred = self.tokenizer.batch_decode(pred, skip_special_tokens=True)
        # true = self.tokenizer.batch_decode(
        #     batch["target_ids"], skip_special_tokens=True
        # )
        return {"test_loss": loss, "pred": logits.argmax(1), "true": labels}

    def test_epoch_end(self, outputs):
        # test_loss = sum([x["test_loss"] for x in outputs])
        # pred = np.array(
        #     list(itertools.chain.from_iterable([x["pred"] for x in outputs]))
        # )
        # true = np.array(
        #     list(itertools.chain.from_iterable([x["true"] for x in outputs]))
        # )

        # f_score = sk_metrics.f1_score(pred, true, average="macro")
        # accuracy = sk_metrics.accuracy_score(pred, true)
        test_loss = sum([x["test_loss"] for x in outputs])
        pred = torch.cat([x["pred"] for x in outputs])
        true = torch.cat([x["true"] for x in outputs])
        f_score = metrics.f1_score(pred, true)
        accuracy = metrics.accuracy(pred, true)
        out = {
            "test_loss": test_loss,
            "test_f_score": f_score,
            "test_accuracy": accuracy,
            "log": {
                "test_loss": test_loss,
                "test_f_score": f_score,
                "test_accuracy": accuracy,
            },
        }
        return out


def format_input(x):
    return f"{x} </s>"
