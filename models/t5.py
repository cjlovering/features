import argparse
import glob
import json
import logging
import os
import random
import re
import time
from itertools import chain
from string import punctuation
import torch.nn as nn
import pytorch_lightning.metrics.functional as metrics

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

import pytorch_lightning as pl

nltk.download("punkt")


class T5Classifier(pl.LightningModule):
    def __init__(self, model, num_steps):
        super(T5Classifier, self).__init__()
        self.model = T5ForConditionalGeneration.from_pretrained(model)
        self.tokenizer = T5Tokenizer.from_pretrained(model)
        self.num_steps = num_steps

    def tokenize(self, batch):
        texts, targets = batch
        texts = [format_input(t) for t in texts]
        targets = [format_output(t) for t in targets]
        batch = _tokenize((texts, targets), self.tokenizer)
        return batch

    def step(self, batch):
        batch = self.tokenize(batch)
        outputs = self.model(
            input_ids=batch["source_ids"],
            attention_mask=batch["source_mask"],
            lm_labels=batch["target_ids"],
            decoder_attention_mask=batch["target_mask"],
        )
        # LM LOSS
        loss = outputs[0]
        logits = outputs[1]
        return batch, loss, logits

    def forward(self, texts):
        """This is used for inference. """
        texts = [format_input(t) for t in texts]
        input_encodings = self.tokenizer.batch_encode_plus(
            texts, padding=True, return_tensors="pt"
        )
        pred = self.tokenizer.batch_decode(
            self.model.generate(
                input_ids=input_encodings["input_ids"],
                attention_mask=input_encodings["attention_mask"],
                max_length=2,
            ),
            skip_special_tokens=True,
        )
        return pred

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
                "weight_decay": 0.001,
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
        optimizer = AdamW(optimizer_grouped_parameters, lr=2e-5,)
        scheduler = get_cosine_schedule_with_warmup(
            optimizer, 0.1 * self.num_steps, self.num_steps
        )
        return [optimizer], [scheduler]

    def training_step(self, batch, batch_idx):
        # batch is tokenized.
        batch, loss, _ = self.step(batch)
        return {"loss": loss}

    def training_epoch_end(self, outputs):
        training_loss = sum([x["loss"] for x in outputs])
        return {"loss": training_loss, "log": {"train_loss": training_loss}}

    def validation_step(self, batch, batch_idx):
        # This is bad /:
        batch, loss, _ = self.step(batch)
        pred = self.model.generate(
            input_ids=batch["source_ids"],
            attention_mask=batch["source_mask"],
            max_length=2,
        )
        return {"val_loss": loss, "pred": pred, "true": batch["target_ids"]}

    def validation_epoch_end(self, outputs):
        val_loss = sum([x["val_loss"] for x in outputs])
        pred = torch.cat([x["pred"] for x in outputs])
        true = torch.cat([x["true"] for x in outputs])
        f_score = metrics.f1_score(pred, true)
        accuracy = metrics.accuracy(pred, true)
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
        batch, loss, _ = self.step(batch)
        pred = self.model.generate(
            input_ids=batch["source_ids"],
            attention_mask=batch["source_mask"],
            max_length=2,
        )
        return {"test_loss": loss, "pred": pred, "true": batch["target_ids"]}

    def test_epoch_end(self, outputs):
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
        print("TEST OUTPUT")
        print(out)
        return out


# process the examples in input and target text format and the eos token at the end
def format_input(x):
    return f"{x} </s>"


def format_output(x):
    return f"{x} </s>"


def _tokenize(batch, tokenizer):
    texts, targets = batch
    input_encodings = tokenizer.batch_encode_plus(
        texts, padding=True, return_tensors="pt"
    )
    target_encodings = tokenizer.batch_encode_plus(
        targets, padding=True, return_tensors="pt"
    )
    return {
        "source_ids": input_encodings["input_ids"],
        "source_mask": input_encodings["attention_mask"],
        "target_ids": target_encodings["input_ids"],
        "target_mask": target_encodings["attention_mask"],
    }
