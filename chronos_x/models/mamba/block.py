"""CryptoMamba block implementation.

A specialized Mamba block for cryptocurrency time series with
state space model (SSM) components optimized for financial data.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
import torch.nn as nn
import torch.nn.functional as F

if TYPE_CHECKING:
    from torch import Tensor


@dataclass
class MambaBlockConfig:
    """Configuration for Mamba block."""

    d_model: int = 256  # Model dimension
    d_state: int = 16  # SSM state dimension
    d_conv: int = 4  # Local convolution width
    expand: int = 2  # Block expansion factor
    dt_rank: str | int = "auto"  # Rank of dt projection
    dt_min: float = 0.001
    dt_max: float = 0.1
    dt_init: str = "random"  # "random" or "constant"
    dt_scale: float = 1.0
    dt_init_floor: float = 1e-4
    bias: bool = False
    conv_bias: bool = True
    pscan: bool = True  # Use parallel scan if available


class MambaBlock(nn.Module):
    """Mamba block with selective state space model.

    This implements the core Mamba architecture with:
    - Input projection
    - 1D convolution for local context
    - Selective SSM for long-range dependencies
    - Output projection
    """

    def __init__(self, config: MambaBlockConfig) -> None:
        super().__init__()
        self.config = config
        self.d_model = config.d_model
        self.d_state = config.d_state
        self.d_conv = config.d_conv
        self.expand = config.expand
        self.d_inner = int(self.expand * self.d_model)

        # Compute dt_rank
        if config.dt_rank == "auto":
            self.dt_rank = math.ceil(self.d_model / 16)
        else:
            self.dt_rank = config.dt_rank

        # Input projection (projects to 2x inner dim for gating)
        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=config.bias)

        # Convolution for local context
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=config.d_conv,
            padding=config.d_conv - 1,
            groups=self.d_inner,
            bias=config.conv_bias,
        )

        # SSM parameters
        # x_proj: project to dt, B, C
        self.x_proj = nn.Linear(
            self.d_inner, self.dt_rank + self.d_state * 2, bias=False
        )

        # dt_proj: project dt from dt_rank to d_inner
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)

        # Initialize dt bias
        dt_init_std = self.dt_rank**-0.5 * config.dt_scale
        if config.dt_init == "constant":
            nn.init.constant_(self.dt_proj.weight, dt_init_std)
        elif config.dt_init == "random":
            nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)

        # Initialize dt bias to log-space
        dt = torch.exp(
            torch.rand(self.d_inner) * (math.log(config.dt_max) - math.log(config.dt_min))
            + math.log(config.dt_min)
        ).clamp(min=config.dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)

        # SSM matrices
        # A is initialized to negative values for stability
        A = torch.arange(1, self.d_state + 1, dtype=torch.float32).repeat(self.d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(self.d_inner))

        # Output projection
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=config.bias)

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass through Mamba block.

        Args:
            x: Input tensor of shape (batch, seq_len, d_model)

        Returns:
            Output tensor of shape (batch, seq_len, d_model)
        """
        batch, seq_len, _ = x.shape

        # Input projection and split for gating
        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=-1)

        # Convolution
        x = x.transpose(1, 2)  # (batch, d_inner, seq_len)
        x = self.conv1d(x)[:, :, :seq_len]  # Remove padding
        x = x.transpose(1, 2)  # (batch, seq_len, d_inner)

        # Activation
        x = F.silu(x)

        # SSM
        y = self.ssm(x)

        # Gating
        y = y * F.silu(z)

        # Output projection
        return self.out_proj(y)

    def ssm(self, x: Tensor) -> Tensor:
        """Selective state space model computation.

        Args:
            x: Input tensor of shape (batch, seq_len, d_inner)

        Returns:
            Output tensor of shape (batch, seq_len, d_inner)
        """
        batch, seq_len, d_inner = x.shape

        # Get A from log space (ensures negative values)
        A = -torch.exp(self.A_log.float())  # (d_inner, d_state)

        # Project x to get dt, B, C
        x_dbl = self.x_proj(x)  # (batch, seq_len, dt_rank + 2*d_state)

        dt, B, C = torch.split(
            x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1
        )

        # Project dt
        dt = self.dt_proj(dt)  # (batch, seq_len, d_inner)
        dt = F.softplus(dt)  # Ensure positive

        # Discretize A and B
        # dA = exp(dt * A)
        # dB = dt * B
        dA = torch.exp(dt.unsqueeze(-1) * A)  # (batch, seq_len, d_inner, d_state)
        dB = dt.unsqueeze(-1) * B.unsqueeze(2)  # (batch, seq_len, d_inner, d_state)

        # Run SSM (selective scan)
        y = self.selective_scan(x, dA, dB, C)

        # Add skip connection with D
        y = y + x * self.D

        return y

    def selective_scan(
        self,
        x: Tensor,
        dA: Tensor,
        dB: Tensor,
        C: Tensor,
    ) -> Tensor:
        """Selective scan operation.

        This is the core of the Mamba algorithm, implementing
        a linear recurrence with input-dependent transitions.

        Args:
            x: Input (batch, seq_len, d_inner)
            dA: Discretized A (batch, seq_len, d_inner, d_state)
            dB: Discretized B (batch, seq_len, d_inner, d_state)
            C: Output matrix (batch, seq_len, d_state)

        Returns:
            Output (batch, seq_len, d_inner)
        """
        batch, seq_len, d_inner = x.shape

        # Initialize state
        h = torch.zeros(batch, d_inner, self.d_state, device=x.device, dtype=x.dtype)

        outputs = []
        for t in range(seq_len):
            # h_t = dA_t * h_{t-1} + dB_t * x_t
            h = dA[:, t] * h + dB[:, t] * x[:, t, :, None]

            # y_t = C_t @ h_t
            y = torch.einsum("bdn,bn->bd", h, C[:, t])
            outputs.append(y)

        return torch.stack(outputs, dim=1)


class ResidualBlock(nn.Module):
    """Residual block with Mamba and normalization."""

    def __init__(
        self,
        d_model: int,
        mamba_config: MambaBlockConfig | None = None,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        if mamba_config is None:
            mamba_config = MambaBlockConfig(d_model=d_model)

        self.norm = nn.LayerNorm(d_model)
        self.mamba = MambaBlock(mamba_config)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        """Forward with residual connection."""
        return x + self.dropout(self.mamba(self.norm(x)))


class CryptoMambaBlock(nn.Module):
    """Specialized Mamba block for cryptocurrency time series.

    Includes additional components for financial time series:
    - Time encoding
    - Volatility gating
    - Multi-scale processing
    """

    def __init__(
        self,
        d_model: int = 256,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dropout: float = 0.1,
        use_time_encoding: bool = True,
        use_volatility_gate: bool = True,
    ) -> None:
        super().__init__()

        self.d_model = d_model
        self.use_time_encoding = use_time_encoding
        self.use_volatility_gate = use_volatility_gate

        # Time encoding
        if use_time_encoding:
            self.time_proj = nn.Linear(1, d_model)

        # Volatility gating
        if use_volatility_gate:
            self.vol_gate = nn.Sequential(
                nn.Linear(1, d_model),
                nn.Sigmoid(),
            )

        # Main Mamba block
        mamba_config = MambaBlockConfig(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )
        self.mamba = ResidualBlock(d_model, mamba_config, dropout)

        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        x: Tensor,
        time_delta: Tensor | None = None,
        volatility: Tensor | None = None,
    ) -> Tensor:
        """Forward pass.

        Args:
            x: Input features (batch, seq_len, d_model)
            time_delta: Time deltas between observations (batch, seq_len, 1)
            volatility: Volatility estimates (batch, seq_len, 1)

        Returns:
            Output features (batch, seq_len, d_model)
        """
        # Add time encoding
        if self.use_time_encoding and time_delta is not None:
            time_enc = self.time_proj(time_delta)
            x = x + time_enc

        # Mamba processing
        x = self.mamba(x)

        # Volatility gating
        if self.use_volatility_gate and volatility is not None:
            gate = self.vol_gate(volatility)
            x = x * gate

        # FFN
        x = x + self.ffn(x)

        return x
