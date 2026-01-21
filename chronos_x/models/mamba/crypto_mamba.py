"""CryptoMamba: Full model architecture for cryptocurrency prediction.

A Mamba-based architecture designed for financial time series with:
- Feature encoder
- Multiple Mamba blocks
- Classification and regression heads
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import torch
import torch.nn as nn
import torch.nn.functional as F

from chronos_x.models.mamba.block import CryptoMambaBlock, MambaBlockConfig

if TYPE_CHECKING:
    from torch import Tensor


@dataclass
class CryptoMambaConfig:
    """Configuration for CryptoMamba model."""

    # Input
    input_dim: int = 64  # Number of input features
    seq_len: int = 256  # Sequence length

    # Architecture
    hidden_dim: int = 256
    num_layers: int = 4
    dropout: float = 0.1

    # Mamba block settings
    d_state: int = 16
    d_conv: int = 4
    expand: int = 2

    # Output heads
    num_classes: int = 3  # Classification classes (down, neutral, up)
    regression_dim: int = 1  # Regression output dimension

    # Features
    use_time_encoding: bool = True
    use_volatility_gate: bool = True

    # Training
    use_auxiliary_heads: bool = True  # Use auxiliary prediction heads


class FeatureEncoder(nn.Module):
    """Encode raw features into model dimension."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )

    def forward(self, x: Tensor) -> Tensor:
        """Encode features.

        Args:
            x: Input features (batch, seq_len, input_dim)

        Returns:
            Encoded features (batch, seq_len, hidden_dim)
        """
        return self.encoder(x)


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding."""

    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)

        self.register_buffer("pe", pe)

    def forward(self, x: Tensor) -> Tensor:
        """Add positional encoding.

        Args:
            x: Input tensor (batch, seq_len, d_model)

        Returns:
            Tensor with positional encoding added
        """
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


class ClassificationHead(nn.Module):
    """Classification head for direction prediction."""

    def __init__(
        self,
        hidden_dim: int,
        num_classes: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes),
        )

    def forward(self, x: Tensor) -> Tensor:
        """Compute class logits.

        Args:
            x: Hidden states (batch, seq_len, hidden_dim) or (batch, hidden_dim)

        Returns:
            Class logits (batch, num_classes) or (batch, seq_len, num_classes)
        """
        return self.head(x)


class RegressionHead(nn.Module):
    """Regression head for return prediction."""

    def __init__(
        self,
        hidden_dim: int,
        output_dim: int = 1,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, output_dim),
        )

    def forward(self, x: Tensor) -> Tensor:
        """Compute regression output.

        Args:
            x: Hidden states (batch, seq_len, hidden_dim) or (batch, hidden_dim)

        Returns:
            Predictions (batch, output_dim) or (batch, seq_len, output_dim)
        """
        return self.head(x)


class CryptoMamba(nn.Module):
    """Full CryptoMamba model for cryptocurrency prediction.

    Architecture:
    1. Feature encoder: Project input features to hidden dimension
    2. Positional encoding: Add position information
    3. Mamba blocks: Process sequence with selective SSM
    4. Output heads: Classification and regression predictions
    """

    def __init__(self, config: CryptoMambaConfig) -> None:
        super().__init__()
        self.config = config

        # Feature encoder
        self.encoder = FeatureEncoder(
            config.input_dim,
            config.hidden_dim,
            config.dropout,
        )

        # Positional encoding
        self.pos_encoding = PositionalEncoding(
            config.hidden_dim,
            max_len=config.seq_len * 2,
            dropout=config.dropout,
        )

        # Mamba blocks
        self.blocks = nn.ModuleList([
            CryptoMambaBlock(
                d_model=config.hidden_dim,
                d_state=config.d_state,
                d_conv=config.d_conv,
                expand=config.expand,
                dropout=config.dropout,
                use_time_encoding=config.use_time_encoding,
                use_volatility_gate=config.use_volatility_gate,
            )
            for _ in range(config.num_layers)
        ])

        # Final normalization
        self.final_norm = nn.LayerNorm(config.hidden_dim)

        # Output heads
        self.classification_head = ClassificationHead(
            config.hidden_dim,
            config.num_classes,
            config.dropout,
        )

        self.regression_head = RegressionHead(
            config.hidden_dim,
            config.regression_dim,
            config.dropout,
        )

        # Auxiliary heads (for intermediate supervision)
        if config.use_auxiliary_heads:
            self.aux_heads = nn.ModuleList([
                ClassificationHead(config.hidden_dim, config.num_classes, config.dropout)
                for _ in range(config.num_layers // 2)
            ])
        else:
            self.aux_heads = None

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        """Initialize weights."""
        if isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(
        self,
        x: Tensor,
        time_delta: Tensor | None = None,
        volatility: Tensor | None = None,
        return_hidden: bool = False,
    ) -> dict[str, Tensor]:
        """Forward pass.

        Args:
            x: Input features (batch, seq_len, input_dim)
            time_delta: Time deltas (batch, seq_len, 1)
            volatility: Volatility estimates (batch, seq_len, 1)
            return_hidden: Whether to return hidden states

        Returns:
            Dictionary containing:
            - class_logits: Classification logits (batch, num_classes)
            - regression: Regression predictions (batch, regression_dim)
            - aux_logits: Auxiliary classification logits (list)
            - hidden: Hidden states (if return_hidden=True)
        """
        # Encode features
        h = self.encoder(x)

        # Add positional encoding
        h = self.pos_encoding(h)

        # Process through Mamba blocks
        aux_outputs = []
        for i, block in enumerate(self.blocks):
            h = block(h, time_delta, volatility)

            # Collect auxiliary outputs
            if self.aux_heads is not None and i % 2 == 1:
                aux_idx = i // 2
                if aux_idx < len(self.aux_heads):
                    aux_out = self.aux_heads[aux_idx](h[:, -1, :])
                    aux_outputs.append(aux_out)

        # Final normalization
        h = self.final_norm(h)

        # Use last hidden state for predictions
        last_hidden = h[:, -1, :]

        # Output heads
        class_logits = self.classification_head(last_hidden)
        regression = self.regression_head(last_hidden)

        output = {
            "class_logits": class_logits,
            "regression": regression,
            "aux_logits": aux_outputs,
        }

        if return_hidden:
            output["hidden"] = h

        return output

    def predict_proba(self, x: Tensor, **kwargs) -> Tensor:
        """Get class probabilities.

        Args:
            x: Input features

        Returns:
            Class probabilities (batch, num_classes)
        """
        output = self.forward(x, **kwargs)
        return F.softmax(output["class_logits"], dim=-1)

    def predict(self, x: Tensor, **kwargs) -> Tensor:
        """Get predicted class labels.

        Args:
            x: Input features

        Returns:
            Predicted labels (batch,)
        """
        output = self.forward(x, **kwargs)
        return output["class_logits"].argmax(dim=-1)

    @property
    def num_parameters(self) -> int:
        """Get total number of parameters."""
        return sum(p.numel() for p in self.parameters())

    @property
    def num_trainable_parameters(self) -> int:
        """Get number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class CryptoMambaEnsemble(nn.Module):
    """Ensemble of CryptoMamba models."""

    def __init__(
        self,
        config: CryptoMambaConfig,
        num_models: int = 3,
    ) -> None:
        super().__init__()

        self.models = nn.ModuleList([
            CryptoMamba(config) for _ in range(num_models)
        ])
        self.num_models = num_models

    def forward(self, x: Tensor, **kwargs) -> dict[str, Tensor]:
        """Forward pass with ensemble averaging.

        Args:
            x: Input features

        Returns:
            Averaged predictions from ensemble
        """
        all_logits = []
        all_regression = []

        for model in self.models:
            output = model(x, **kwargs)
            all_logits.append(output["class_logits"])
            all_regression.append(output["regression"])

        # Average predictions
        avg_logits = torch.stack(all_logits).mean(dim=0)
        avg_regression = torch.stack(all_regression).mean(dim=0)

        return {
            "class_logits": avg_logits,
            "regression": avg_regression,
            "individual_logits": all_logits,
            "individual_regression": all_regression,
        }

    def predict_with_uncertainty(self, x: Tensor, **kwargs) -> dict[str, Tensor]:
        """Get predictions with uncertainty estimates.

        Args:
            x: Input features

        Returns:
            Predictions with uncertainty (std across ensemble)
        """
        output = self.forward(x, **kwargs)

        # Compute probabilities for each model
        probs = [F.softmax(logits, dim=-1) for logits in output["individual_logits"]]
        probs_stack = torch.stack(probs)

        # Mean and std
        mean_probs = probs_stack.mean(dim=0)
        std_probs = probs_stack.std(dim=0)

        # Regression uncertainty
        reg_stack = torch.stack(output["individual_regression"])
        mean_reg = reg_stack.mean(dim=0)
        std_reg = reg_stack.std(dim=0)

        return {
            "proba": mean_probs,
            "proba_std": std_probs,
            "regression": mean_reg,
            "regression_std": std_reg,
        }


def create_model(
    input_dim: int = 64,
    hidden_dim: int = 256,
    num_layers: int = 4,
    num_classes: int = 3,
    dropout: float = 0.1,
    **kwargs,
) -> CryptoMamba:
    """Factory function to create CryptoMamba model.

    Args:
        input_dim: Number of input features
        hidden_dim: Hidden dimension
        num_layers: Number of Mamba blocks
        num_classes: Number of output classes
        dropout: Dropout rate
        **kwargs: Additional config parameters

    Returns:
        CryptoMamba model
    """
    config = CryptoMambaConfig(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        num_classes=num_classes,
        dropout=dropout,
        **kwargs,
    )

    return CryptoMamba(config)
