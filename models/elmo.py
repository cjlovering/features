import pytorch_lightning as pl
import torch
import torch.nn as nn

from torch.optim import Adam

import pytorch_lightning.metrics.functional as metrics
from . import head


from allennlp.modules.elmo import Elmo, batch_to_ids


class ElmoClassifier(pl.LightningModule):
    def __init__(self, hidden_size: int = 300, num_classes=2):
        super(ElmoClassifier, self).__init__()

        options_file = "./resources/options.json"
        weight_file = "./resources/weights.hdf5"

        self.elmo = Elmo(options_file, weight_file, 1, dropout=0)
        # 1024: Elmo hidden size.
        self.lstm = nn.LSTM(1024, hidden_size, batch_first=True)
        self.classifier = head.ClassificationHead(hidden_size, num_classes)

        if torch.cuda.is_available():
            self._device = "cuda"
            self.elmo.cuda()
        else:
            self._device = "cpu"

    def forward(self, batch):
        texts, _ = batch

        # Embed with Elmo.
        word_ids = batch_to_ids(texts).to(self._device)

        torch.set_default_tensor_type("torch.FloatTensor")
        elmo_out = self.elmo(word_ids)
        embeddings = elmo_out["elmo_representations"][0]
        mask = elmo_out["mask"]
        lengths = mask.sum(axis=1).cpu().to(torch.int64)
        packed_embeddings = torch.nn.utils.rnn.pack_padded_sequence(
            embeddings, lengths, batch_first=True, enforce_sorted=False
        )
        if torch.cuda.is_available():
            torch.set_default_tensor_type("torch.cuda.FloatTensor")
        # Process with another LSTM and then classify.
        _, (ht, _) = self.lstm(packed_embeddings)
        logits = self.classifier(ht[-1])
        return logits

    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=2e-5)
        return [optimizer]

    def training_step(self, batch, batch_idx):
        _, labels = batch
        logits = self.forward(batch)
        loss = nn.functional.cross_entropy(logits, labels)
        return {"loss": loss}

    def training_epoch_end(self, outputs):
        training_loss = sum([x["loss"] for x in outputs])
        return {"train_loss": training_loss, "log": {"train_loss": training_loss}}

    def validation_step(self, batch, batch_idx):
        _, labels = batch
        logits = self.forward(batch)
        loss = nn.functional.cross_entropy(logits, labels)
        return {"val_loss": loss, "pred": logits.argmax(1), "true": labels}

    def validation_epoch_end(self, outputs):
        val_loss = torch.stack([x["val_loss"] for x in outputs]).sum()
        pred = torch.cat([x["pred"] for x in outputs])
        true = torch.cat([x["true"] for x in outputs])
        f_score = metrics.f1_score(pred, true)
        accuracy = metrics.accuracy(pred, true)
        out = {
            "val_loss": val_loss,
            "val_f_score": f_score,
            "val_accuracy": accuracy,
        }
        return {**out, "log": out}

    def test_step(self, batch, batch_idx):
        _, labels = batch
        logits = self.forward(batch)
        loss = nn.functional.cross_entropy(logits, labels, reduction="sum")
        return {"test_loss": loss, "pred": logits.argmax(1), "true": labels}

    def test_epoch_end(self, outputs):
        avg_loss = torch.stack([x["test_loss"] for x in outputs]).sum()
        pred = torch.cat([x["pred"] for x in outputs])
        true = torch.cat([x["true"] for x in outputs])
        f_score = metrics.f1_score(pred, true)
        accuracy = metrics.accuracy(pred, true)
        out = {
            "test_loss": avg_loss,
            "test_f_score": f_score,
            "test_accuracy": accuracy,
        }
        return {**out, "log": out}

