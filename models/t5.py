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
    T5Model,
    get_linear_schedule_with_warmup,
    get_cosine_schedule_with_warmup,
    GPT2Model,
)
import sklearn.metrics as sk_metrics

import pytorch_lightning as pl

nltk.download("punkt")


class T5Classifier(pl.LightningModule):
    def __init__(self, model, num_steps, num_classes=2):
        super(T5Classifier, self).__init__()
        hidden_size = {"t5-small": 512, "t5-base": 768, "t5-large": 1024,}[model]
        self.model = T5ForConditionalGeneration.from_pretrained(model)
        self.tokenizer = T5Tokenizer.from_pretrained(model)
        self.num_steps = num_steps
        self.classifier = nn.Linear(hidden_size, num_classes)

    def step(self, batch):
        """I was unable to get the model to *work* using the typical
        T5 text api. Here I try just getting the last hidden state
        and using a linear classifier on top of that.
        """
        texts, labels = batch
        texts = [format_input(t) for t in texts]
        labels = [format_output_in(t) for t in labels]

        input_ids = self.tokenizer.batch_encode_plus(
            texts, padding=True, return_tensors="pt", max_length=64
        )
        output_ids = self.tokenizer.batch_encode_plus(
            labels, padding=True, return_tensors="pt", max_length=2
        )
        outputs = self.model(
            input_ids=input_ids["input_ids"],
            attention_mask=input_ids["attention_mask"],
            labels=output_ids["input_ids"],
            return_dict=True,
        )
        loss = outputs.loss
        logits = outputs.logits
        # last_hidden_states = outputs[0][:, -1, :]
        # logits = self.classifier(last_hidden_states)
        # loss = nn.functional.cross_entropy(logits, labels)
        return loss, logits

    def forward(self, batch):
        """This is used for inference. """
        # _, logits = self.step(batch)
        # return logits
        texts, _ = batch
        texts = [format_input(t) for t in texts]
        input_ids = self.tokenizer.batch_encode_plus(
            texts, padding=True, return_tensors="pt", max_length=64
        )
        pred = self.model.generate(
            input_ids=input_ids["input_ids"],
            attention_mask=input_ids["attention_mask"],
            max_length=2,
        )
        pred = self.tokenizer.batch_decode(pred, skip_special_tokens=True)
        return [format_output_out(p) for p in pred]

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=1e-4,)
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
        _, labels = batch

        pred = self.tokenizer.batch_decode(logits.argmax(-1), skip_special_tokens=True)
        pred = torch.tensor([format_output_out(p) for p in pred])
        true = labels
        print(pred)
        print(true)
        return {"val_loss": loss, "pred": pred, "true": true}
        # pred = logits.argmax(-1)
        # print(true, pred)

        # pred = self.model.generate(input_ids=input_ids["input_ids"], max_length=2,)
        # print(logits.size())
        #         _labels = [format_output_in(t) for t in labels]
        # true = self.tokenizer.batch_encode_plus(
        #     _labels, padding=True, return_tensors="pt", max_length=2
        # )["input_ids"][:, 0]

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
        loss, logits = self.step(batch)
        _, labels = batch

        pred = self.tokenizer.batch_decode(logits.argmax(-1), skip_special_tokens=True)
        pred = torch.tensor([format_output_out(p) for p in pred])
        true = labels
        return {"val_loss": loss, "pred": pred, "true": true}

        # _, labels = batch
        # loss, logits = self.step(batch)
        # # pred = self.model.generate(
        # #     input_ids=batch["source_ids"],
        # #     attention_mask=batch["source_mask"],
        # #     max_length=2,
        # # )
        # # pred = self.tokenizer.batch_decode(pred, skip_special_tokens=True)
        # # true = self.tokenizer.batch_decode(
        # #     batch["target_ids"], skip_special_tokens=True
        # # )
        # labels = [format_output_in(t) for t in labels]
        # true = self.tokenizer.batch_encode_plus(
        #     labels, padding=True, return_tensors="pt", max_length=2
        # )["input_ids"][:, 0]
        # pred = logits[:, 0, :].argmax(1)
        # return {"test_loss": loss, "pred": true, "true": pred}

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
    return f"binary classification: {x}"


def format_output_in(x):
    return {0: "False", 1: "True"}[x.item()]


def format_output_out(x):
    if x in {"True", "False"}:
        y = {"False": 0, "True": 1}[x]
    else:
        # Something else!
        y = 2
    return y
