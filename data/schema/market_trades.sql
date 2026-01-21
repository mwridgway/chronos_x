-- Chronos-X ClickHouse Schema
-- Optimized for high-frequency trade data storage and aggregation

-- Create database
CREATE DATABASE IF NOT EXISTS chronos;

-- Raw trades table with compression codecs
CREATE TABLE IF NOT EXISTS chronos.trades
(
    exchange LowCardinality(String),
    symbol LowCardinality(String),
    trade_id String CODEC(ZSTD(3)),
    timestamp DateTime64(3, 'UTC') CODEC(Delta, ZSTD(1)),
    price Float64 CODEC(Delta, ZSTD(3)),
    quantity Float64 CODEC(Delta, ZSTD(3)),
    side Enum8('buy' = 1, 'sell' = -1),
    is_maker Bool DEFAULT false,

    -- Derived fields for faster queries
    notional Float64 MATERIALIZED price * quantity,
    date Date MATERIALIZED toDate(timestamp)
)
ENGINE = MergeTree()
PARTITION BY (exchange, toYYYYMM(timestamp))
ORDER BY (exchange, symbol, timestamp, trade_id)
TTL timestamp + INTERVAL 2 YEAR
SETTINGS index_granularity = 8192;

-- Order book snapshots (L2)
CREATE TABLE IF NOT EXISTS chronos.orderbook_snapshots
(
    exchange LowCardinality(String),
    symbol LowCardinality(String),
    timestamp DateTime64(3, 'UTC') CODEC(Delta, ZSTD(1)),

    -- Top 20 levels each side
    bid_prices Array(Float64) CODEC(ZSTD(3)),
    bid_quantities Array(Float64) CODEC(ZSTD(3)),
    ask_prices Array(Float64) CODEC(ZSTD(3)),
    ask_quantities Array(Float64) CODEC(ZSTD(3)),

    -- Summary metrics
    mid_price Float64 MATERIALIZED (bid_prices[1] + ask_prices[1]) / 2,
    spread Float64 MATERIALIZED ask_prices[1] - bid_prices[1],
    spread_bps Float64 MATERIALIZED (ask_prices[1] - bid_prices[1]) / mid_price * 10000
)
ENGINE = MergeTree()
PARTITION BY (exchange, toYYYYMM(timestamp))
ORDER BY (exchange, symbol, timestamp)
TTL timestamp + INTERVAL 6 MONTH
SETTINGS index_granularity = 8192;

-- OHLCV Materialized View (1-minute candles)
CREATE MATERIALIZED VIEW IF NOT EXISTS chronos.ohlcv_1m
ENGINE = SummingMergeTree()
PARTITION BY (exchange, toYYYYMM(timestamp))
ORDER BY (exchange, symbol, timestamp)
AS SELECT
    exchange,
    symbol,
    toStartOfMinute(timestamp) AS timestamp,
    argMin(price, timestamp) AS open,
    max(price) AS high,
    min(price) AS low,
    argMax(price, timestamp) AS close,
    sum(quantity) AS volume,
    sum(notional) AS notional_volume,
    count() AS trade_count,
    sum(if(side = 'buy', quantity, 0)) AS buy_volume,
    sum(if(side = 'sell', quantity, 0)) AS sell_volume,
    sum(if(side = 'buy', notional, 0)) AS buy_notional,
    sum(if(side = 'sell', notional, 0)) AS sell_notional
FROM chronos.trades
GROUP BY exchange, symbol, toStartOfMinute(timestamp);

-- OHLCV 5-minute candles
CREATE MATERIALIZED VIEW IF NOT EXISTS chronos.ohlcv_5m
ENGINE = SummingMergeTree()
PARTITION BY (exchange, toYYYYMM(timestamp))
ORDER BY (exchange, symbol, timestamp)
AS SELECT
    exchange,
    symbol,
    toStartOfFiveMinutes(timestamp) AS timestamp,
    argMin(price, timestamp) AS open,
    max(price) AS high,
    min(price) AS low,
    argMax(price, timestamp) AS close,
    sum(quantity) AS volume,
    sum(notional) AS notional_volume,
    count() AS trade_count,
    sum(if(side = 'buy', quantity, 0)) AS buy_volume,
    sum(if(side = 'sell', quantity, 0)) AS sell_volume
FROM chronos.trades
GROUP BY exchange, symbol, toStartOfFiveMinutes(timestamp);

-- OHLCV 1-hour candles
CREATE MATERIALIZED VIEW IF NOT EXISTS chronos.ohlcv_1h
ENGINE = SummingMergeTree()
PARTITION BY (exchange, toYYYYMM(timestamp))
ORDER BY (exchange, symbol, timestamp)
AS SELECT
    exchange,
    symbol,
    toStartOfHour(timestamp) AS timestamp,
    argMin(price, timestamp) AS open,
    max(price) AS high,
    min(price) AS low,
    argMax(price, timestamp) AS close,
    sum(quantity) AS volume,
    sum(notional) AS notional_volume,
    count() AS trade_count,
    sum(if(side = 'buy', quantity, 0)) AS buy_volume,
    sum(if(side = 'sell', quantity, 0)) AS sell_volume
FROM chronos.trades
GROUP BY exchange, symbol, toStartOfHour(timestamp);

-- Daily OHLCV
CREATE MATERIALIZED VIEW IF NOT EXISTS chronos.ohlcv_1d
ENGINE = SummingMergeTree()
PARTITION BY (exchange, toYear(timestamp))
ORDER BY (exchange, symbol, timestamp)
AS SELECT
    exchange,
    symbol,
    toStartOfDay(timestamp) AS timestamp,
    argMin(price, timestamp) AS open,
    max(price) AS high,
    min(price) AS low,
    argMax(price, timestamp) AS close,
    sum(quantity) AS volume,
    sum(notional) AS notional_volume,
    count() AS trade_count,
    sum(if(side = 'buy', quantity, 0)) AS buy_volume,
    sum(if(side = 'sell', quantity, 0)) AS sell_volume
FROM chronos.trades
GROUP BY exchange, symbol, toStartOfDay(timestamp);

-- Microstructure metrics table (computed by Python)
CREATE TABLE IF NOT EXISTS chronos.microstructure
(
    exchange LowCardinality(String),
    symbol LowCardinality(String),
    timestamp DateTime64(3, 'UTC') CODEC(Delta, ZSTD(1)),
    window_minutes UInt16,

    -- Order Flow Imbalance
    ofi Float64 CODEC(ZSTD(3)),
    ofi_normalized Float64 CODEC(ZSTD(3)),

    -- Volume metrics
    vpin Float64 CODEC(ZSTD(3)),
    volume_imbalance Float64 CODEC(ZSTD(3)),

    -- Spread metrics
    effective_spread Float64 CODEC(ZSTD(3)),
    realized_spread Float64 CODEC(ZSTD(3)),

    -- Volatility
    realized_volatility Float64 CODEC(ZSTD(3)),
    parkinson_volatility Float64 CODEC(ZSTD(3)),

    -- Trade metrics
    trade_count UInt32,
    avg_trade_size Float64
)
ENGINE = MergeTree()
PARTITION BY (exchange, toYYYYMM(timestamp))
ORDER BY (exchange, symbol, timestamp, window_minutes)
SETTINGS index_granularity = 8192;

-- Feature store for ML
CREATE TABLE IF NOT EXISTS chronos.features
(
    exchange LowCardinality(String),
    symbol LowCardinality(String),
    timestamp DateTime64(3, 'UTC') CODEC(Delta, ZSTD(1)),

    -- Feature vector (stored as array for flexibility)
    feature_names Array(String) CODEC(ZSTD(3)),
    feature_values Array(Float64) CODEC(ZSTD(3)),

    -- Metadata
    feature_version UInt16,
    created_at DateTime64(3, 'UTC') DEFAULT now64(3)
)
ENGINE = MergeTree()
PARTITION BY (exchange, toYYYYMM(timestamp))
ORDER BY (exchange, symbol, timestamp)
SETTINGS index_granularity = 8192;

-- Labels for training
CREATE TABLE IF NOT EXISTS chronos.labels
(
    exchange LowCardinality(String),
    symbol LowCardinality(String),
    timestamp DateTime64(3, 'UTC') CODEC(Delta, ZSTD(1)),

    -- Triple barrier labels
    label Int8,  -- -1, 0, 1
    return_pct Float64,
    holding_period_minutes UInt32,
    exit_type Enum8('tp' = 1, 'sl' = -1, 'timeout' = 0),

    -- Meta-label
    meta_label Float64,  -- Probability of correct prediction

    -- Barrier parameters used
    tp_multiple Float32,
    sl_multiple Float32,
    max_holding_minutes UInt32,
    volatility_window UInt16
)
ENGINE = MergeTree()
PARTITION BY (exchange, toYYYYMM(timestamp))
ORDER BY (exchange, symbol, timestamp)
SETTINGS index_granularity = 8192;

-- Model predictions log
CREATE TABLE IF NOT EXISTS chronos.predictions
(
    exchange LowCardinality(String),
    symbol LowCardinality(String),
    timestamp DateTime64(3, 'UTC') CODEC(Delta, ZSTD(1)),

    model_name String CODEC(ZSTD(3)),
    model_version String CODEC(ZSTD(3)),

    -- Raw predictions
    pred_proba Array(Float32) CODEC(ZSTD(3)),  -- [prob_down, prob_neutral, prob_up]
    pred_label Int8,

    -- Meta-label filter
    meta_proba Float32,
    signal_strength Float32,

    -- Execution
    executed Bool DEFAULT false,
    execution_price Float64,
    execution_timestamp DateTime64(3, 'UTC')
)
ENGINE = MergeTree()
PARTITION BY (exchange, toYYYYMM(timestamp))
ORDER BY (exchange, symbol, timestamp)
TTL timestamp + INTERVAL 1 YEAR
SETTINGS index_granularity = 8192;

-- Execution log
CREATE TABLE IF NOT EXISTS chronos.executions
(
    exchange LowCardinality(String),
    symbol LowCardinality(String),
    order_id String CODEC(ZSTD(3)),

    signal_timestamp DateTime64(3, 'UTC'),
    order_timestamp DateTime64(3, 'UTC'),
    fill_timestamp DateTime64(3, 'UTC'),

    side Enum8('buy' = 1, 'sell' = -1),
    order_type LowCardinality(String),

    requested_quantity Float64,
    filled_quantity Float64,
    avg_fill_price Float64,

    -- Slippage analysis
    expected_price Float64,
    slippage_bps Float64 MATERIALIZED (avg_fill_price - expected_price) / expected_price * 10000,

    -- Fees
    commission Float64,
    commission_asset LowCardinality(String),

    status LowCardinality(String)
)
ENGINE = MergeTree()
PARTITION BY (exchange, toYYYYMM(order_timestamp))
ORDER BY (exchange, symbol, order_timestamp, order_id)
SETTINGS index_granularity = 8192;

-- Create useful indices
ALTER TABLE chronos.trades ADD INDEX idx_symbol symbol TYPE bloom_filter GRANULARITY 1;
ALTER TABLE chronos.trades ADD INDEX idx_timestamp timestamp TYPE minmax GRANULARITY 1;
