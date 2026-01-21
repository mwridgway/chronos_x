# Chronos-X

A cryptocurrency trading system built with Mamba-based ML models for BTC/USDT trading on Binance.

## Features

- **Data Pipeline**: Real-time and historical data ingestion via CCXT/WebSocket
- **Feature Engineering**: Fractional differentiation, microstructure features (OFI, VPIN)
- **ML Model**: CryptoMamba - SSM-based architecture for sequence modeling
- **Labeling**: Triple Barrier method with meta-labeling for signal filtering
- **Validation**: Combinatorial Purged Cross-Validation (CPCV) with Deflated Sharpe Ratio
- **Risk Management**: HRP portfolio allocation, volatility targeting, Kelly criterion
- **Execution**: TWAP/VWAP algorithms with adaptive execution
- **Production**: Async live trading with paper trading mode and drift detection

## Project Structure

```
chronos_x/
├── docker/                    # Docker Compose for ClickHouse
├── config/                    # Hydra configuration files
├── chronos_x/
│   ├── data/
│   │   ├── ingestion/        # ClickHouse client, WebSocket, CCXT loader
│   │   ├── processing/       # FracDiff, microstructure, labeling
│   │   └── schema/           # ClickHouse SQL schema
│   ├── models/
│   │   ├── mamba/           # CryptoMamba model architecture
│   │   ├── agents/          # Meta-labeler
│   │   └── training/        # Trainer, loss functions
│   ├── strategy/            # Signal generation, risk, execution
│   ├── validation/          # CPCV, metrics (DSR, Sharpe)
│   ├── production/          # Live trader, monitoring
│   └── scripts/             # CLI tools
└── tests/
```

## Quick Start

### 1. Setup Environment

```bash
# Install dependencies
pip install -e .

# Start ClickHouse
cd docker && docker-compose up -d
```

### 2. Initialize Database

```bash
python -m chronos_x.scripts.ingest_data init-schema
```

### 3. Backfill Historical Data

```bash
python -m chronos_x.scripts.ingest_data backfill BTC/USDT --start 2024-01-01
```

### 4. Train Model

```bash
python -m chronos_x.scripts.train_model train BTC/USDT --epochs 100
```

### 5. Run Backtest

```bash
python -m chronos_x.scripts.run_backtest backtest BTC/USDT --model-path ./models/crypto_mamba.pt
```

### 6. Start Paper Trading

```python
import asyncio
from chronos_x.production.live_trader import run_paper_trading

asyncio.run(run_paper_trading(
    model=model,
    symbols=["BTC/USDT"],
    duration_hours=24,
))
```

## Configuration

Configuration uses Hydra with hierarchical YAML files:

- `config/config.yaml` - Main configuration
- `config/data/binance.yaml` - Exchange settings
- `config/model/mamba_default.yaml` - Model architecture
- `config/backtest/fees_slippage.yaml` - Backtest parameters
- `config/sweep/optimizing_dsr.yaml` - Hyperparameter optimization

Override via CLI:
```bash
python -m chronos_x.scripts.train_model train \
    model.hidden_dim=512 \
    training.batch_size=128
```

## Key Components

### CryptoMamba Model

State Space Model (SSM) architecture optimized for financial time series:
- Input encoder with positional encoding
- Multiple Mamba blocks with selective SSM
- Classification and regression heads
- Optional auxiliary heads for deep supervision

### Triple Barrier Labeling

Labels based on:
- Take profit barrier (2x volatility)
- Stop loss barrier (2x volatility)
- Maximum holding period (60 bars)

### Meta-Labeling

Secondary model (LightGBM/XGBoost) predicts probability that primary model signals are correct, filtering low-confidence trades.

### CPCV Validation

Combinatorial Purged K-Fold with:
- Purging: removes overlapping samples
- Embargo: adds gap after test sets
- Multiple test combinations for robust estimation

### Risk Management

- Volatility targeting position sizing
- Kelly criterion (fractional)
- Maximum drawdown limits
- Daily loss limits

## Environment Variables

```bash
BINANCE_API_KEY=your_api_key
BINANCE_API_SECRET=your_secret
BINANCE_TESTNET_API_KEY=testnet_key
BINANCE_TESTNET_API_SECRET=testnet_secret
```

## Dependencies

- Python >= 3.12
- PyTorch >= 2.2
- ClickHouse for data storage
- CCXT for exchange connectivity
- Polars/Numba for fast feature computation

## License

MIT
