"""LiLAWFusionLightningModule — drop-in replacement for FusionLightningModule.

Wraps MultiModalFusionTransformer with LiLAW bilevel training:
  - Inner step: update model weights on a training batch using LiLAW-weighted BCE.
  - Outer step: update LiLAW meta-parameters (alpha, beta, delta) on a validation
    batch drawn from the val dataloader.

Usage (replace in your training script):

    from lilaw_module import LiLAWFusionLightningModule

    model = LiLAWFusionLightningModule(
        tabular_dim=tabular_dim,
        text_dim=text_dim,
        pos_weight=pos_weight[mlmd_name],
        lr=1e-3,
        use_pos_weight=True,
        warmup_epochs=1,   # epochs of vanilla BCE before LiLAW activates
        meta_lr=0.005,
        meta_wd=0.0001,
        alpha_init=10.0,
        beta_init=2.0,
        delta_init=6.0,
    )

Everything else (Trainer config, callbacks, dataloaders, save_model) is
unchanged. The only other required change is placing lilaw.py in the same
package and adjusting the import path below if needed.

Hyperparameters to tune:
  - warmup_epochs: 1–3 recommended. LiLAW needs a semi-trained model to
    produce meaningful probability estimates before meta-learning begins.
  - meta_lr: 0.001–0.01. Controls how fast alpha/beta/delta adapt.
  - alpha_init / beta_init / delta_init: paper defaults (10, 2, 6) are a
    reasonable starting point.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from torchmetrics.classification import BinaryAUROC, BinaryAveragePrecision, BinaryF1Score

from lilaw import LiLAWWeighter


# ---------------------------------------------------------------------------
# Architecture (unchanged from the student notebook)
# ---------------------------------------------------------------------------

class AttentionPooling(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        B, T, D = x.shape
        q = self.query.expand(B, -1, -1)
        attn_scores = torch.bmm(q, x.transpose(1, 2))
        attn_weights = self.softmax(attn_scores)
        return torch.bmm(attn_weights, x).squeeze(1)


class MultiModalFusionTransformer(nn.Module):
    def __init__(self, tabular_dim, text_dim, hidden_dim=1024, num_layers=4,
                 n_heads=4, ff_dims=None, dropout=0.1):
        super().__init__()

        self.tabular_proj = nn.Sequential(
            nn.Linear(tabular_dim, 2 * text_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(2 * text_dim, text_dim),
            nn.ReLU(),
        )

        self.pos_embed = nn.Parameter(torch.zeros(1, 4, text_dim))
        nn.init.xavier_uniform_(self.pos_embed)

        if ff_dims is None:
            ff_dims = [1024] * num_layers

        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=text_dim,
                nhead=n_heads,
                dim_feedforward=ff_dims[i],
                dropout=dropout,
                batch_first=True,
            )
            for i in range(num_layers)
        ])

        self.attn_pool = AttentionPooling(text_dim)

        self.classifier = nn.Sequential(
            nn.LayerNorm(text_dim),
            nn.Linear(text_dim, 1),
        )

    def forward(self, tabular_x, text1_x, text2_x, text3_x):
        tabular_proj = self.tabular_proj(tabular_x).unsqueeze(1)
        text_x = torch.stack([text1_x, text2_x, text3_x], dim=1)
        x = torch.cat([tabular_proj, text_x], dim=1)
        x = x + self.pos_embed
        for layer in self.layers:
            x = layer(x)
        pooled = self.attn_pool(x)
        return self.classifier(pooled).squeeze(-1)


# ---------------------------------------------------------------------------
# LiLAW Lightning module
# ---------------------------------------------------------------------------

class LiLAWFusionLightningModule(pl.LightningModule):
    """FusionLightningModule with LiLAW bilevel training.

    During warmup_epochs the model trains with standard weighted BCE
    (identical to the baseline). After warmup, each training batch triggers:
      1. Model update on the training batch using LiLAW-weighted per-sample BCE.
      2. Meta-parameter update on the next validation batch.

    Logging is identical to the baseline: train/val loss, AUROC, AUPRC, F1.
    Meta-parameter trajectories are additionally logged as
    lilaw_alpha / lilaw_beta / lilaw_delta.
    """

    def __init__(
        self,
        tabular_dim: int,
        text_dim: int,
        pos_weight: float | None = None,
        lr: float = 1e-3,
        use_pos_weight: bool = False,
        warmup_epochs: int = 1,
        meta_lr: float = 0.005,
        meta_wd: float = 0.0001,
        alpha_init: float = 10.0,
        beta_init: float = 2.0,
        delta_init: float = 6.0,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.automatic_optimization = False

        self.model = MultiModalFusionTransformer(
            tabular_dim=tabular_dim,
            text_dim=text_dim,
            hidden_dim=1024,
            num_layers=4,
            n_heads=4,
            ff_dims=None,
            dropout=0.1,
        )

        self.use_pos_weight = use_pos_weight
        if use_pos_weight and pos_weight is not None:
            capped = min(pos_weight, 10.0)
            self.register_buffer("pos_weight", torch.tensor(capped))
        else:
            self.register_buffer("pos_weight", None)

        # Used during warmup and for validation loss logging
        pw = self.pos_weight if (use_pos_weight and self.pos_weight is not None) else None
        self.criterion = nn.BCEWithLogitsLoss(pos_weight=pw)

        self.weighter = LiLAWWeighter(
            alpha_init=alpha_init,
            beta_init=beta_init,
            delta_init=delta_init,
        )

        self._val_iter = None

        # Metrics
        self.auroc = BinaryAUROC()
        self.auprc = BinaryAveragePrecision()
        self.f1 = BinaryF1Score()

        # For compatibility with MetricsLogger callback
        self.train_losses: list[float] = []
        self.val_losses: list[float] = []

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def on_train_start(self) -> None:
        # Move LiLAW meta-params to the training device.
        # (They are plain tensors outside Lightning's parameter management.)
        self.weighter.to(self.device)

    def _get_val_batch(self):
        """Draw the next batch from the validation dataloader."""
        if self._val_iter is None:
            self._val_iter = iter(self.trainer.val_dataloaders)
        try:
            batch = next(self._val_iter)
        except StopIteration:
            self._val_iter = iter(self.trainer.val_dataloaders)
            batch = next(self._val_iter)
        # Move to device (Lightning does not do this for manually fetched batches)
        return tuple(t.to(self.device) if isinstance(t, torch.Tensor) else t
                     for t in batch)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, tabular_x, text1_x, text2_x, text3_x):
        return self.model(tabular_x, text1_x, text2_x, text3_x)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def training_step(self, batch, batch_idx):
        opt = self.optimizers()
        tabular_x, text1_x, text2_x, text3_x, labels = batch
        use_lilaw = self.current_epoch >= self.hparams.warmup_epochs

        # --- Inner step: update model weights ---
        opt.zero_grad()
        logits = self(tabular_x, text1_x, text2_x, text3_x)

        if use_lilaw:
            self.weighter.set_requires_grad(requires_grad=False)
            # Per-sample BCE, respecting pos_weight if set
            bce = F.binary_cross_entropy_with_logits(
                logits, labels.float(),
                pos_weight=self.pos_weight,
                reduction="none",
            )
            # Compute LiLAW weights from detached probabilities
            with torch.no_grad():
                probs = torch.sigmoid(logits)
            s_label = probs * labels + (1 - probs) * (1 - labels)
            s_max = torch.max(probs, 1 - probs)
            _, _, _, w = self.weighter.compute_weights(s_label, s_max)
            loss = (w.detach() * bce).mean()
        else:
            loss = self.criterion(logits, labels.float())

        self.manual_backward(loss)
        opt.step()

        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)

        with torch.no_grad():
            probs = torch.sigmoid(logits)
        self.auroc.update(probs, labels.int())
        self.auprc.update(probs, labels.int())
        self.f1.update(probs, labels.int())
        self.log("train_auroc", self.auroc.compute(), on_step=False, on_epoch=True, prog_bar=True)
        self.log("train_auprc", self.auprc.compute(), on_step=False, on_epoch=True, prog_bar=True)
        self.log("train_f1", self.f1.compute(), on_step=False, on_epoch=True, prog_bar=True)
        self.auroc.reset()
        self.auprc.reset()
        self.f1.reset()

        # --- Outer step: update LiLAW meta-parameters on a val batch ---
        if use_lilaw:
            val_tabular, val_text1, val_text2, val_text3, val_labels = self._get_val_batch()

            self.weighter.set_requires_grad(requires_grad=True)
            self.weighter.zero_grad()

            with torch.no_grad():
                val_logits = self(val_tabular, val_text1, val_text2, val_text3)

            val_bce = F.binary_cross_entropy_with_logits(
                val_logits, val_labels.float(), reduction="none"
            )
            val_probs = torch.sigmoid(val_logits)
            s_label_val = val_probs * val_labels + (1 - val_probs) * (1 - val_labels)
            s_max_val = torch.max(val_probs, 1 - val_probs)
            _, _, _, w_val = self.weighter.compute_weights(s_label_val, s_max_val)
            meta_loss = (w_val * val_bce).mean()
            meta_loss.backward()
            self.weighter.meta_step(
                lr=self.hparams.meta_lr,
                wd=self.hparams.meta_wd,
            )

            self.log("lilaw_alpha", self.weighter.alpha.item(), on_step=False, on_epoch=True)
            self.log("lilaw_beta", self.weighter.beta.item(), on_step=False, on_epoch=True)
            self.log("lilaw_delta", self.weighter.delta.item(), on_step=False, on_epoch=True)

        return loss

    def on_validation_epoch_end(self) -> None:
        sch = self.lr_schedulers()
        val_loss = self.trainer.callback_metrics.get("val_loss")
        if sch is not None and val_loss is not None:
            sch.step(val_loss)

    # ------------------------------------------------------------------
    # Validation (unchanged from baseline)
    # ------------------------------------------------------------------

    def validation_step(self, batch, batch_idx):
        tabular_x, text1_x, text2_x, text3_x, labels = batch
        logits = self(tabular_x, text1_x, text2_x, text3_x)
        probs = torch.sigmoid(logits)
        loss = self.criterion(logits, labels.float())

        if batch_idx == 0:
            with torch.no_grad():
                pos_indices = (labels == 1).nonzero(as_tuple=True)[0]
                neg_indices = (labels == 0).nonzero(as_tuple=True)[0]
                k = 5
                sampled = torch.cat([
                    pos_indices[torch.randperm(len(pos_indices))[:min(len(pos_indices), k)]],
                    neg_indices[torch.randperm(len(neg_indices))[:min(len(neg_indices), k)]],
                ])
                self.print(f"[VAL] Logits stats: min={logits.min():.2f}, max={logits.max():.2f}, mean={logits.mean():.2f}")
                self.print("Sample probs:", probs[sampled].detach().cpu().numpy())
                self.print("Sample labels:", labels[sampled].cpu().numpy())

        self.auroc.update(probs, labels.int())
        self.auprc.update(probs, labels.int())
        self.f1.update(probs, labels.int())
        self.log("val_loss", loss, on_epoch=True, prog_bar=True)
        self.log("val_auroc", self.auroc.compute(), on_epoch=True, prog_bar=True)
        self.log("val_auprc", self.auprc.compute(), on_epoch=True, prog_bar=True)
        self.log("val_f1", self.f1.compute(), on_epoch=True, prog_bar=True)
        self.auroc.reset()
        self.auprc.reset()
        self.f1.reset()
        return loss

    # ------------------------------------------------------------------
    # Optimizer (unchanged from baseline)
    # ------------------------------------------------------------------

    def configure_optimizers(self):
        optimizer = torch.optim.RAdam(
            self.parameters(), lr=self.hparams.lr, weight_decay=1e-4
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.1, patience=2, min_lr=1e-4,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "monitor": "val_loss"},
        }
