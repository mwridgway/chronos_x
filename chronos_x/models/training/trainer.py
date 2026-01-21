"""Training module for CryptoMamba models.

Provides a flexible training loop with:
- Mixed precision training
- Gradient accumulation
- Early stopping
- Checkpointing
- MLflow integration
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

import numpy as np
import structlog
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, OneCycleLR
from torch.utils.data import DataLoader, Dataset

from chronos_x.models.training.objectives import CombinedLoss

if TYPE_CHECKING:
    from torch import Tensor

logger = structlog.get_logger(__name__)


@dataclass
class TrainerConfig:
    """Configuration for model training."""

    # Optimization
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    betas: tuple[float, float] = (0.9, 0.999)

    # Training loop
    epochs: int = 100
    batch_size: int = 64
    gradient_clip: float = 1.0
    accumulation_steps: int = 1

    # Learning rate schedule
    scheduler: str = "cosine"  # "cosine", "onecycle", "none"
    warmup_steps: int = 1000
    min_lr: float = 1e-6

    # Early stopping
    early_stopping: bool = True
    patience: int = 10
    min_delta: float = 0.001

    # Checkpointing
    checkpoint_dir: str = "./checkpoints"
    save_top_k: int = 3

    # Mixed precision
    use_amp: bool = True
    amp_dtype: str = "float16"  # "float16" or "bfloat16"

    # Logging
    log_interval: int = 100
    eval_interval: int = 1000

    # Loss weights
    classification_weight: float = 0.7
    regression_weight: float = 0.3

    # Device
    device: str = "cuda"

    # MLflow
    use_mlflow: bool = True
    mlflow_experiment: str = "chronos-x"


@dataclass
class TrainingState:
    """Current training state."""

    epoch: int = 0
    global_step: int = 0
    best_metric: float = float("-inf")
    best_epoch: int = 0
    patience_counter: int = 0
    train_losses: list[float] = field(default_factory=list)
    val_metrics: list[dict] = field(default_factory=list)


class CryptoDataset(Dataset):
    """Dataset for CryptoMamba training."""

    def __init__(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        returns: np.ndarray | None = None,
        sample_weights: np.ndarray | None = None,
    ) -> None:
        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)

        if returns is not None:
            self.returns = torch.tensor(returns, dtype=torch.float32)
        else:
            self.returns = torch.zeros(len(labels), dtype=torch.float32)

        if sample_weights is not None:
            self.weights = torch.tensor(sample_weights, dtype=torch.float32)
        else:
            self.weights = torch.ones(len(labels), dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> dict[str, Tensor]:
        return {
            "features": self.features[idx],
            "label": self.labels[idx],
            "return": self.returns[idx],
            "weight": self.weights[idx],
        }


class EarlyStopping:
    """Early stopping handler."""

    def __init__(
        self,
        patience: int = 10,
        min_delta: float = 0.001,
        mode: str = "max",
    ) -> None:
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score: float | None = None
        self.should_stop = False

    def __call__(self, score: float) -> bool:
        """Check if training should stop.

        Args:
            score: Current validation score

        Returns:
            True if training should stop
        """
        if self.best_score is None:
            self.best_score = score
            return False

        if self.mode == "max":
            improved = score > self.best_score + self.min_delta
        else:
            improved = score < self.best_score - self.min_delta

        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True

        return self.should_stop


class ModelCheckpoint:
    """Model checkpointing handler."""

    def __init__(
        self,
        save_dir: str | Path,
        save_top_k: int = 3,
        mode: str = "max",
        filename_format: str = "model-epoch={epoch}-metric={metric:.4f}.pt",
    ) -> None:
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.save_top_k = save_top_k
        self.mode = mode
        self.filename_format = filename_format
        self.best_checkpoints: list[tuple[float, Path]] = []

    def __call__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        epoch: int,
        metric: float,
        state: TrainingState,
    ) -> Path | None:
        """Save checkpoint if metric is in top-k.

        Returns:
            Path to saved checkpoint or None
        """
        # Check if this should be saved
        if len(self.best_checkpoints) < self.save_top_k:
            should_save = True
        else:
            worst_metric = self.best_checkpoints[-1][0]
            if self.mode == "max":
                should_save = metric > worst_metric
            else:
                should_save = metric < worst_metric

        if not should_save:
            return None

        # Save checkpoint
        filename = self.filename_format.format(epoch=epoch, metric=metric)
        path = self.save_dir / filename

        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "epoch": epoch,
                "metric": metric,
                "state": state,
            },
            path,
        )

        # Update best checkpoints list
        self.best_checkpoints.append((metric, path))
        self.best_checkpoints.sort(
            key=lambda x: x[0],
            reverse=(self.mode == "max"),
        )

        # Remove excess checkpoints
        while len(self.best_checkpoints) > self.save_top_k:
            _, old_path = self.best_checkpoints.pop()
            if old_path.exists():
                old_path.unlink()

        return path


class CryptoMambaTrainer:
    """Trainer for CryptoMamba models."""

    def __init__(
        self,
        model: nn.Module,
        config: TrainerConfig | None = None,
    ) -> None:
        self.config = config or TrainerConfig()
        self.model = model.to(self.config.device)
        self._log = logger.bind(component="trainer")

        # Initialize components
        self._init_optimizer()
        self._init_loss()
        self._init_callbacks()

        # Training state
        self.state = TrainingState()

        # Mixed precision
        if self.config.use_amp:
            self.scaler = GradScaler()
        else:
            self.scaler = None

        # MLflow
        self._mlflow_run = None
        if self.config.use_mlflow:
            self._init_mlflow()

    def _init_optimizer(self) -> None:
        """Initialize optimizer and scheduler."""
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
            betas=self.config.betas,
        )

    def _init_scheduler(self, num_training_steps: int) -> None:
        """Initialize learning rate scheduler."""
        if self.config.scheduler == "cosine":
            self.scheduler = CosineAnnealingWarmRestarts(
                self.optimizer,
                T_0=num_training_steps // 4,
                T_mult=2,
                eta_min=self.config.min_lr,
            )
        elif self.config.scheduler == "onecycle":
            self.scheduler = OneCycleLR(
                self.optimizer,
                max_lr=self.config.learning_rate,
                total_steps=num_training_steps,
                pct_start=self.config.warmup_steps / num_training_steps,
            )
        else:
            self.scheduler = None

    def _init_loss(self) -> None:
        """Initialize loss function."""
        self.loss_fn = CombinedLoss(
            classification_weight=self.config.classification_weight,
            regression_weight=self.config.regression_weight,
        )

    def _init_callbacks(self) -> None:
        """Initialize callbacks."""
        if self.config.early_stopping:
            self.early_stopping = EarlyStopping(
                patience=self.config.patience,
                min_delta=self.config.min_delta,
            )
        else:
            self.early_stopping = None

        self.checkpointer = ModelCheckpoint(
            save_dir=self.config.checkpoint_dir,
            save_top_k=self.config.save_top_k,
        )

    def _init_mlflow(self) -> None:
        """Initialize MLflow tracking."""
        try:
            import mlflow

            mlflow.set_experiment(self.config.mlflow_experiment)
            self._mlflow_run = mlflow.start_run()
            mlflow.log_params(
                {
                    "learning_rate": self.config.learning_rate,
                    "batch_size": self.config.batch_size,
                    "epochs": self.config.epochs,
                    "weight_decay": self.config.weight_decay,
                }
            )
        except ImportError:
            self._log.warning("MLflow not available, skipping tracking")

    def _train_step(self, batch: dict[str, Tensor]) -> dict[str, float]:
        """Execute single training step."""
        features = batch["features"].to(self.config.device)
        labels = batch["label"].to(self.config.device)
        returns = batch["return"].to(self.config.device)

        # Mixed precision forward pass
        with autocast(enabled=self.config.use_amp):
            outputs = self.model(features)
            losses = self.loss_fn(
                outputs["class_logits"],
                labels,
                outputs["regression"],
                returns,
                outputs.get("aux_logits"),
            )

        loss = losses["total"] / self.config.accumulation_steps

        # Backward pass
        if self.scaler is not None:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()

        return {k: v.item() for k, v in losses.items()}

    def _optimizer_step(self) -> None:
        """Execute optimizer step with gradient clipping."""
        if self.scaler is not None:
            self.scaler.unscale_(self.optimizer)

        torch.nn.utils.clip_grad_norm_(
            self.model.parameters(),
            self.config.gradient_clip,
        )

        if self.scaler is not None:
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            self.optimizer.step()

        self.optimizer.zero_grad()

        if self.scheduler is not None:
            self.scheduler.step()

    @torch.no_grad()
    def evaluate(self, dataloader: DataLoader) -> dict[str, float]:
        """Evaluate model on validation set."""
        self.model.eval()

        total_loss = 0.0
        all_preds = []
        all_labels = []
        all_returns = []
        all_pred_returns = []

        for batch in dataloader:
            features = batch["features"].to(self.config.device)
            labels = batch["label"].to(self.config.device)
            returns = batch["return"].to(self.config.device)

            outputs = self.model(features)
            losses = self.loss_fn(
                outputs["class_logits"],
                labels,
                outputs["regression"],
                returns,
            )

            total_loss += losses["total"].item()
            all_preds.extend(outputs["class_logits"].argmax(dim=-1).cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_returns.extend(returns.cpu().numpy())
            all_pred_returns.extend(outputs["regression"].squeeze().cpu().numpy())

        self.model.train()

        # Compute metrics
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        all_returns = np.array(all_returns)

        accuracy = (all_preds == all_labels).mean()

        # Direction accuracy (ignoring neutral predictions)
        direction_mask = all_preds != 1  # Assuming 1 is neutral
        if direction_mask.sum() > 0:
            direction_acc = (
                (all_preds[direction_mask] == all_labels[direction_mask]).mean()
            )
        else:
            direction_acc = 0.0

        return {
            "val_loss": total_loss / len(dataloader),
            "accuracy": accuracy,
            "direction_accuracy": direction_acc,
        }

    def train(
        self,
        train_dataset: Dataset,
        val_dataset: Dataset | None = None,
        callbacks: list[Callable] | None = None,
    ) -> TrainingState:
        """Train the model.

        Args:
            train_dataset: Training dataset
            val_dataset: Optional validation dataset
            callbacks: Optional list of callback functions

        Returns:
            Final training state
        """
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
        )

        val_loader = None
        if val_dataset is not None:
            val_loader = DataLoader(
                val_dataset,
                batch_size=self.config.batch_size * 2,
                shuffle=False,
                num_workers=4,
                pin_memory=True,
            )

        # Initialize scheduler
        num_training_steps = len(train_loader) * self.config.epochs
        self._init_scheduler(num_training_steps)

        self._log.info(
            "starting_training",
            epochs=self.config.epochs,
            train_samples=len(train_dataset),
            val_samples=len(val_dataset) if val_dataset else 0,
        )

        self.model.train()
        start_time = time.time()

        for epoch in range(self.config.epochs):
            self.state.epoch = epoch
            epoch_losses = []

            for batch_idx, batch in enumerate(train_loader):
                losses = self._train_step(batch)
                epoch_losses.append(losses["total"])

                # Optimizer step (with accumulation)
                if (batch_idx + 1) % self.config.accumulation_steps == 0:
                    self._optimizer_step()
                    self.state.global_step += 1

                # Logging
                if self.state.global_step % self.config.log_interval == 0:
                    avg_loss = np.mean(epoch_losses[-self.config.log_interval :])
                    self._log.info(
                        "train_step",
                        epoch=epoch,
                        step=self.state.global_step,
                        loss=avg_loss,
                        lr=self.optimizer.param_groups[0]["lr"],
                    )

                    if self.config.use_mlflow:
                        try:
                            import mlflow

                            mlflow.log_metrics(
                                {"train_loss": avg_loss},
                                step=self.state.global_step,
                            )
                        except Exception:
                            pass

            # End of epoch
            avg_epoch_loss = np.mean(epoch_losses)
            self.state.train_losses.append(avg_epoch_loss)

            # Validation
            if val_loader is not None:
                val_metrics = self.evaluate(val_loader)
                self.state.val_metrics.append(val_metrics)

                self._log.info(
                    "epoch_complete",
                    epoch=epoch,
                    train_loss=avg_epoch_loss,
                    **val_metrics,
                )

                # Checkpointing
                metric = val_metrics.get("accuracy", -val_metrics["val_loss"])
                self.checkpointer(
                    self.model,
                    self.optimizer,
                    epoch,
                    metric,
                    self.state,
                )

                # Early stopping
                if self.early_stopping is not None:
                    if self.early_stopping(metric):
                        self._log.info("early_stopping", epoch=epoch)
                        break

                    if metric > self.state.best_metric:
                        self.state.best_metric = metric
                        self.state.best_epoch = epoch

            # Run callbacks
            if callbacks:
                for callback in callbacks:
                    callback(self.state, self.model)

        elapsed = time.time() - start_time
        self._log.info(
            "training_complete",
            elapsed_seconds=elapsed,
            best_epoch=self.state.best_epoch,
            best_metric=self.state.best_metric,
        )

        # End MLflow run
        if self.config.use_mlflow and self._mlflow_run:
            try:
                import mlflow

                mlflow.end_run()
            except Exception:
                pass

        return self.state

    def load_checkpoint(self, path: str | Path) -> None:
        """Load model from checkpoint."""
        checkpoint = torch.load(path, map_location=self.config.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.state = checkpoint.get("state", TrainingState())

    def save_model(self, path: str | Path) -> None:
        """Save model state dict only."""
        torch.save(self.model.state_dict(), path)
