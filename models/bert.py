import nltk
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn as nn
from transformers import (
    AdamW,
    BertModel,
    BertTokenizer,
    get_cosine_schedule_with_warmup,
)

import pytorch_lightning.metrics.functional as metrics


class BertClassifier(pl.LightningModule):
    def __init__(self, model, num_steps, num_classes=2):
        super(BertClassifier, self).__init__()
        hidden_size = {"bert-base-uncased": 768, "bert-large-uncased": 1024,}[model]

        # TODO: make `hidden_size` contigent on the encoder.
        # `bert-large-*` has a bigger hidden_size.
        self.tokenizer = BertTokenizer.from_pretrained(model)
        self.encoder = BertModel.from_pretrained(model)
        self.classifier = nn.Linear(hidden_size, num_classes)
        self.num_steps = num_steps

    def forward(self, texts):
        encoded = self.encoder(self.tokenize(texts))[1]
        logits = self.classifier(encoded)
        return logits

    def tokenize(self, texts):
        X = torch.nn.utils.rnn.pad_sequence(
            [
                torch.tensor(self.tokenizer.encode(t, add_special_tokens=True))
                for t in texts
            ],
            batch_first=True,
        )
        return X

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=2e-5)
        scheduler = get_cosine_schedule_with_warmup(
            optimizer, 0.1 * self.num_steps, self.num_steps
        )
        return [optimizer], [scheduler]

    def training_step(self, batch, batch_idx):
        texts, labels = batch
        logits = self.forward(texts)
        loss = nn.functional.cross_entropy(logits, labels)
        return {"loss": loss}

    def training_epoch_end(self, outputs):
        training_loss = sum([x["loss"] for x in outputs])
        return {"loss": training_loss, "log": {"train_loss": training_loss}}

    def validation_step(self, batch, batch_idx):
        texts, labels = batch
        logits = self.forward(texts)
        loss = nn.functional.cross_entropy(logits, labels)
        return {"val_loss": loss, "pred": logits.argmax(1), "true": labels}

    def validation_epoch_end(self, outputs):
        val_loss = torch.stack([x["val_loss"] for x in outputs]).sum()
        pred = torch.stack([x["pred"] for x in outputs])
        true = torch.stack([x["true"] for x in outputs])
        f_score = metrics.f1_score(pred, true)
        accuracy = metrics.accuracy(pred, true)
        out = {"val_loss": val_loss, "val_f_score": f_score, "val_accuracy": accuracy}
        return {**out, "log": out}

    def test_step(self, batch, batch_idx):
        texts, labels = batch
        logits = self.forward(texts)
        loss = nn.functional.cross_entropy(logits, labels)
        return {"test_loss": loss, "pred": logits.argmax(1), "true": labels}

    def test_epoch_end(self, outputs):
        avg_loss = torch.stack([x["test_loss"] for x in outputs]).mean()
        pred = torch.stack([x["pred"] for x in outputs])
        true = torch.stack([x["true"] for x in outputs])
        f_score = metrics.f1_score(pred, true)
        accuracy = metrics.accuracy(pred, true)
        return {
            "test_loss": avg_loss,
            "test_f_score": f_score,
            "test_accuracy": accuracy,
        }
