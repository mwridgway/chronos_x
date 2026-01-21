"""Custom loss functions and objectives for CryptoMamba training.

Includes:
- Focal Loss for imbalanced classification
- Weighted Cross Entropy
- Huber Loss for regression
- Combined loss for multi-task learning
- Sharpe Ratio-based loss
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn as nn
import torch.nn.functional as F

if TYPE_CHECKING:
    from torch import Tensor


class FocalLoss(nn.Module):
    """Focal Loss for imbalanced classification.

    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    Args:
        alpha: Class weights (list or tensor)
        gamma: Focusing parameter (default: 2.0)
        reduction: Reduction method ('mean', 'sum', 'none')
    """

    def __init__(
        self,
        alpha: list[float] | Tensor | None = None,
        gamma: float = 2.0,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction

        if alpha is not None:
            if isinstance(alpha, list):
                alpha = torch.tensor(alpha)
            self.register_buffer("alpha", alpha)
        else:
            self.alpha = None

    def forward(self, inputs: Tensor, targets: Tensor) -> Tensor:
        """Compute focal loss.

        Args:
            inputs: Predictions (batch, num_classes)
            targets: Ground truth labels (batch,)

        Returns:
            Focal loss value
        """
        ce_loss = F.cross_entropy(inputs, targets, reduction="none")
        pt = torch.exp(-ce_loss)

        focal_weight = (1 - pt) ** self.gamma

        if self.alpha is not None:
            alpha_t = self.alpha[targets]
            focal_weight = alpha_t * focal_weight

        loss = focal_weight * ce_loss

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


class LabelSmoothingCrossEntropy(nn.Module):
    """Cross entropy with label smoothing.

    Args:
        smoothing: Label smoothing factor (default: 0.1)
        reduction: Reduction method
    """

    def __init__(self, smoothing: float = 0.1, reduction: str = "mean") -> None:
        super().__init__()
        self.smoothing = smoothing
        self.reduction = reduction

    def forward(self, inputs: Tensor, targets: Tensor) -> Tensor:
        """Compute label-smoothed cross entropy.

        Args:
            inputs: Predictions (batch, num_classes)
            targets: Ground truth labels (batch,)

        Returns:
            Loss value
        """
        num_classes = inputs.size(-1)
        log_probs = F.log_softmax(inputs, dim=-1)

        # Create smoothed targets
        targets_one_hot = F.one_hot(targets, num_classes).float()
        targets_smooth = (
            targets_one_hot * (1 - self.smoothing)
            + self.smoothing / num_classes
        )

        loss = -(targets_smooth * log_probs).sum(dim=-1)

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


class HuberLoss(nn.Module):
    """Huber loss for robust regression.

    L_delta(a) = 0.5 * a^2           if |a| <= delta
                 delta * (|a| - 0.5 * delta)  otherwise

    Args:
        delta: Threshold for switching between L1 and L2
        reduction: Reduction method
    """

    def __init__(self, delta: float = 1.0, reduction: str = "mean") -> None:
        super().__init__()
        self.delta = delta
        self.reduction = reduction

    def forward(self, inputs: Tensor, targets: Tensor) -> Tensor:
        """Compute Huber loss.

        Args:
            inputs: Predictions
            targets: Ground truth

        Returns:
            Huber loss value
        """
        return F.huber_loss(
            inputs, targets, delta=self.delta, reduction=self.reduction
        )


class QuantileLoss(nn.Module):
    """Quantile loss for probabilistic regression.

    Args:
        quantiles: List of quantiles to predict
        reduction: Reduction method
    """

    def __init__(
        self,
        quantiles: list[float] = [0.1, 0.5, 0.9],
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        self.quantiles = quantiles
        self.reduction = reduction

    def forward(self, inputs: Tensor, targets: Tensor) -> Tensor:
        """Compute quantile loss.

        Args:
            inputs: Predictions (batch, num_quantiles)
            targets: Ground truth (batch,)

        Returns:
            Quantile loss value
        """
        losses = []
        for i, q in enumerate(self.quantiles):
            errors = targets - inputs[:, i]
            losses.append(torch.max((q - 1) * errors, q * errors))

        loss = torch.stack(losses, dim=1).sum(dim=1)

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


class DirectionalLoss(nn.Module):
    """Loss that penalizes wrong direction predictions more heavily.

    Args:
        alpha: Weight for direction penalty
        reduction: Reduction method
    """

    def __init__(self, alpha: float = 1.0, reduction: str = "mean") -> None:
        super().__init__()
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, inputs: Tensor, targets: Tensor) -> Tensor:
        """Compute directional loss.

        Args:
            inputs: Return predictions
            targets: Actual returns

        Returns:
            Loss value
        """
        # MSE component
        mse = (inputs - targets) ** 2

        # Direction component: penalize when signs differ
        sign_match = torch.sign(inputs) == torch.sign(targets)
        direction_penalty = torch.where(
            sign_match,
            torch.zeros_like(mse),
            torch.abs(inputs) + torch.abs(targets),
        )

        loss = mse + self.alpha * direction_penalty

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


class SharpeLoss(nn.Module):
    """Loss based on negative Sharpe ratio for direct optimization.

    Maximizes risk-adjusted returns during training.

    Args:
        annualization_factor: Factor for annualizing Sharpe
        reduction: Reduction method
    """

    def __init__(
        self,
        annualization_factor: float = 252.0,
        min_std: float = 1e-6,
    ) -> None:
        super().__init__()
        self.annualization_factor = annualization_factor
        self.min_std = min_std

    def forward(
        self,
        positions: Tensor,
        returns: Tensor,
    ) -> Tensor:
        """Compute negative Sharpe ratio loss.

        Args:
            positions: Predicted positions/signals
            returns: Actual returns

        Returns:
            Negative Sharpe ratio (to minimize)
        """
        # Strategy returns
        strategy_returns = positions * returns

        # Sharpe ratio
        mean_return = strategy_returns.mean()
        std_return = strategy_returns.std() + self.min_std

        sharpe = (mean_return / std_return) * torch.sqrt(
            torch.tensor(self.annualization_factor)
        )

        # Return negative Sharpe (we minimize loss)
        return -sharpe


class SortinoLoss(nn.Module):
    """Loss based on negative Sortino ratio.

    Only penalizes downside volatility.

    Args:
        annualization_factor: Factor for annualizing ratio
        target_return: Minimum acceptable return
    """

    def __init__(
        self,
        annualization_factor: float = 252.0,
        target_return: float = 0.0,
        min_std: float = 1e-6,
    ) -> None:
        super().__init__()
        self.annualization_factor = annualization_factor
        self.target_return = target_return
        self.min_std = min_std

    def forward(
        self,
        positions: Tensor,
        returns: Tensor,
    ) -> Tensor:
        """Compute negative Sortino ratio loss.

        Args:
            positions: Predicted positions/signals
            returns: Actual returns

        Returns:
            Negative Sortino ratio (to minimize)
        """
        strategy_returns = positions * returns

        # Downside returns
        downside = torch.clamp(strategy_returns - self.target_return, max=0)
        downside_std = torch.sqrt((downside ** 2).mean()) + self.min_std

        mean_return = strategy_returns.mean()
        sortino = (mean_return - self.target_return) / downside_std
        sortino = sortino * torch.sqrt(torch.tensor(self.annualization_factor))

        return -sortino


class CombinedLoss(nn.Module):
    """Combined loss for multi-task learning.

    Combines classification and regression losses with configurable weights.

    Args:
        classification_weight: Weight for classification loss
        regression_weight: Weight for regression loss
        auxiliary_weight: Weight for auxiliary losses
        focal_gamma: Gamma for focal loss
        focal_alpha: Alpha for focal loss
        huber_delta: Delta for Huber loss
    """

    def __init__(
        self,
        classification_weight: float = 0.7,
        regression_weight: float = 0.3,
        auxiliary_weight: float = 0.1,
        focal_gamma: float = 2.0,
        focal_alpha: list[float] | None = None,
        huber_delta: float = 1.0,
    ) -> None:
        super().__init__()

        self.classification_weight = classification_weight
        self.regression_weight = regression_weight
        self.auxiliary_weight = auxiliary_weight

        # Classification loss
        self.classification_loss = FocalLoss(
            alpha=focal_alpha,
            gamma=focal_gamma,
        )

        # Regression loss
        self.regression_loss = HuberLoss(delta=huber_delta)

    def forward(
        self,
        class_logits: Tensor,
        class_targets: Tensor,
        regression: Tensor,
        regression_targets: Tensor,
        auxiliary_logits: list[Tensor] | None = None,
    ) -> dict[str, Tensor]:
        """Compute combined loss.

        Args:
            class_logits: Classification predictions
            class_targets: Classification targets
            regression: Regression predictions
            regression_targets: Regression targets
            auxiliary_logits: Optional auxiliary classification outputs

        Returns:
            Dictionary with total loss and components
        """
        # Classification loss
        cls_loss = self.classification_loss(class_logits, class_targets)

        # Regression loss
        reg_loss = self.regression_loss(
            regression.squeeze(), regression_targets.squeeze()
        )

        # Total loss
        total_loss = (
            self.classification_weight * cls_loss
            + self.regression_weight * reg_loss
        )

        # Auxiliary losses
        aux_loss = torch.tensor(0.0, device=class_logits.device)
        if auxiliary_logits:
            for aux_logit in auxiliary_logits:
                aux_loss = aux_loss + self.classification_loss(aux_logit, class_targets)
            aux_loss = aux_loss / len(auxiliary_logits)
            total_loss = total_loss + self.auxiliary_weight * aux_loss

        return {
            "total": total_loss,
            "classification": cls_loss,
            "regression": reg_loss,
            "auxiliary": aux_loss,
        }


class AsymmetricLoss(nn.Module):
    """Asymmetric loss that penalizes false negatives more than false positives.

    Useful when missing a good trade is worse than taking a bad one.

    Args:
        gamma_pos: Focusing parameter for positive samples
        gamma_neg: Focusing parameter for negative samples
        clip: Probability clipping threshold
    """

    def __init__(
        self,
        gamma_pos: float = 0.0,
        gamma_neg: float = 4.0,
        clip: float = 0.05,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        self.gamma_pos = gamma_pos
        self.gamma_neg = gamma_neg
        self.clip = clip
        self.reduction = reduction

    def forward(self, inputs: Tensor, targets: Tensor) -> Tensor:
        """Compute asymmetric loss.

        Args:
            inputs: Predictions (batch, num_classes)
            targets: One-hot targets (batch, num_classes)

        Returns:
            Loss value
        """
        # Sigmoid for multi-label
        probs = torch.sigmoid(inputs)

        # Positive samples
        pos_loss = targets * torch.log(probs.clamp(min=1e-8))
        if self.gamma_pos > 0:
            pos_loss = pos_loss * ((1 - probs) ** self.gamma_pos)

        # Negative samples
        neg_probs = (probs - self.clip).clamp(min=0)
        neg_loss = (1 - targets) * torch.log((1 - neg_probs).clamp(min=1e-8))
        if self.gamma_neg > 0:
            neg_loss = neg_loss * (neg_probs ** self.gamma_neg)

        loss = -(pos_loss + neg_loss)

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss
