import os

import torch

import numpy as np
import pytorch_lightning as pl
import torch.nn as nn

import pytorch_lightning.metrics.functional as metrics
from transformers import AdamW


class LstmGloveClassifier(pl.LightningModule):
    def __init__(
        self, model, num_classes=2, glove_path: str = "./data/glove",
    ):
        super(LstmGloveClassifier, self).__init__()

        if not os.path.exists(f"{glove_path}/glove.6B.300d.txt"):
            assert (
                False
            ), "Download glove: `wget http://downloads.cs.stanford.edu/nlp/data/glove.6B.300d.zip`"

        glove = {}
        with open(f"{glove_path}/glove.6B.300d.txt", encoding="utf8") as f:
            for line in f:
                values = line.split()
                word = values[0]
                coefs = np.asarray(values[1:], dtype="float32")
                glove[word] = coefs

        word2idx = {word: idx for idx, word in enumerate(glove.keys())}
        matrix_len = len(glove)
        hidden_size = 300
        weights_matrix = np.zeros((matrix_len, hidden_size))
        for idx, (_, embedding_vector) in enumerate(glove.items()):
            weights_matrix[idx] = embedding_vector

        self.word2idx = word2idx
        self.embedding = create_emb_layer(weights_matrix, trainable=False)
        self.lstm = nn.LSTM(hidden_size, hidden_size, batch_first=True)
        self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(self, batch):
        texts, _ = batch
        tokens = self.tokenize(texts)
        embeddings = self.embedding(tokens)
        _, (ht, _) = self.lstm(embeddings)
        logits = self.classifier(ht[-1])
        return logits

    def tokenize(self, texts):
        X = torch.nn.utils.rnn.pad_sequence(
            [
                torch.tensor([self.word2idx[w.lower()] for w in t.split()])
                for t in texts
            ],
            batch_first=True,
        )
        return X

    def training_step(self, batch, batch_idx):
        _, labels = batch
        logits = self.forward(batch)
        loss = nn.functional.cross_entropy(logits, labels)
        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        _, labels = batch
        logits = self.forward(batch)
        loss = nn.functional.cross_entropy(logits, labels)
        return {"val_loss": loss, "pred": logits.argmax(1), "true": labels}

    def validation_epoch_end(self, outputs):
        val_loss = torch.stack([x["val_loss"] for x in outputs]).sum()
        pred = torch.stack([x["pred"] for x in outputs])
        true = torch.stack([x["true"] for x in outputs])
        f_score = metrics.f1_score(pred, true)
        accuracy = metrics.accuracy(pred, true)
        return {
            "val_loss": val_loss,
            "f_score": f_score,
            "accuracy": accuracy,
        }

    def test_step(self, batch, batch_idx):
        _, labels = batch
        logits = self.forward(batch)
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
            "f_score": f_score,
            "accuracy": accuracy,
        }

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=2e-5)
        return [optimizer]


def create_emb_layer(weights_matrix, trainable):
    num_embeddings, embedding_dim = weights_matrix.shape
    emb_layer = nn.Embedding(num_embeddings, embedding_dim)
    emb_layer.load_state_dict({"weight": torch.tensor(weights_matrix)})
    emb_layer.weight.requires_grad = trainable
    return emb_layer
