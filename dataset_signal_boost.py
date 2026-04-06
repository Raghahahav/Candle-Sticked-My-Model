from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import pandas as pd

try:
    import yfinance as yf
except ImportError as exc:  # pragma: no cover
    yf = None
    YFINANCE_IMPORT_ERROR = exc
else:
    YFINANCE_IMPORT_ERROR = None


PREFERRED_LABEL_ORDER = ("down", "neutral", "up")


@dataclass
class SignalConfig:
    data_root: str = "stock_dataset"
    output_csv: str = "stock_dataset/metadata_binary_1pct.csv"
    interval: str = "1h"
    period: str = "1y"
    timezone: str = "Asia/Kolkata"
    market_start: str = "09:15"
    market_end: str = "15:30"
    label_mode: Literal["binary_fixed", "ternary_fixed", "binary_quantile", "ternary_quantile"] = "binary_fixed"
    threshold: float = 0.01
    quantile: float = 0.35
    lookahead_days: int = 1
    min_abs_return: float = 0.0
    train_ratio: float = 0.7
    val_ratio: float = 0.2
    purge_gap: int = 19
    drop_missing_images: bool = True


def load_base_metadata(data_root: str | Path) -> pd.DataFrame:
    root = Path(data_root)
    df = pd.read_csv(root / "labels.csv")
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["image_idx"] = df["image_path"].str.extract(r"_(\d+)\.png$").astype(int)
    df["resolved_path"] = [
        str((root / "raw_images" / stock / Path(image_path).name).resolve())
        for image_path, stock in zip(df["image_path"], df["stock"])
    ]
    return df.sort_values(["stock", "timestamp", "image_idx"]).reset_index(drop=True)


def fetch_stock_ohlc(stock: str, config: SignalConfig) -> pd.DataFrame:
    if yf is None:
        raise ImportError("yfinance is required to rebuild labels.") from YFINANCE_IMPORT_ERROR

    ticker = yf.Ticker(f"{stock}.NS")
    df = ticker.history(period=config.period, interval=config.interval, auto_adjust=True)
    if df.empty:
        raise ValueError(f"No OHLC data returned for {stock}")

    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    df.index = df.index.tz_convert(config.timezone)
    df = df.between_time(config.market_start, config.market_end)
    df = df[["Open", "High", "Low", "Close"]].copy()
    df = df.sort_index()
    df = df.dropna()
    return df


def compute_future_returns(ohlc: pd.DataFrame, lookahead_days: int) -> pd.DataFrame:
    frame = ohlc.copy()
    frame["date"] = frame.index.normalize()
    daily_close = frame.groupby("date")["Close"].last()
    future_close = daily_close.shift(-lookahead_days)
    frame["future_close"] = frame["date"].map(future_close)
    frame["future_return"] = (frame["future_close"] - frame["Close"]) / frame["Close"]
    return frame


def label_fixed(return_value: float, threshold: float, binary: bool) -> str | None:
    if pd.isna(return_value):
        return None
    if return_value >= threshold:
        return "up"
    if return_value <= -threshold:
        return "down"
    return None if binary else "neutral"


def assign_labels(group: pd.DataFrame, config: SignalConfig) -> pd.DataFrame:
    group = group.copy()
    returns = group["future_return"]

    if config.label_mode == "binary_fixed":
        group["label"] = returns.map(lambda value: label_fixed(value, config.threshold, binary=True))
    elif config.label_mode == "ternary_fixed":
        group["label"] = returns.map(lambda value: label_fixed(value, config.threshold, binary=False))
    elif config.label_mode == "binary_quantile":
        lower = returns.quantile(config.quantile)
        upper = returns.quantile(1 - config.quantile)
        group["label"] = returns.map(
            lambda value: "down" if value <= lower else ("up" if value >= upper else None)
        )
    elif config.label_mode == "ternary_quantile":
        lower = returns.quantile(config.quantile)
        upper = returns.quantile(1 - config.quantile)
        group["label"] = returns.map(
            lambda value: "down" if value <= lower else ("up" if value >= upper else "neutral")
        )
    else:
        raise ValueError(f"Unknown label_mode: {config.label_mode}")

    if config.min_abs_return > 0:
        group.loc[group["future_return"].abs() < config.min_abs_return, "label"] = None
    return group


def build_purged_split(df: pd.DataFrame, train_ratio: float, val_ratio: float, gap: int) -> pd.DataFrame:
    parts = []
    for stock, stock_df in df.groupby("stock", sort=False):
        stock_df = stock_df.sort_values(["timestamp", "image_idx"]).reset_index(drop=True).copy()
        n = len(stock_df)
        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + val_ratio))
        val_start = min(train_end + gap, n)
        test_start = min(val_end + gap, n)

        stock_df["split"] = "gap"
        stock_df.loc[: train_end - 1, "split"] = "train"
        stock_df.loc[val_start : val_end - 1, "split"] = "val"
        stock_df.loc[test_start:, "split"] = "test"
        parts.append(stock_df)

    out = pd.concat(parts, ignore_index=True)
    return out[out["split"] != "gap"].reset_index(drop=True)


def preferred_label_order(labels: pd.Series) -> list[str]:
    unique = [str(label) for label in pd.Index(labels.dropna().unique()).tolist()]
    ordered = [label for label in PREFERRED_LABEL_ORDER if label in unique]
    ordered.extend(sorted(label for label in unique if label not in ordered))
    return ordered


def _to_python_nested(value):
    if isinstance(value, dict):
        return {key: _to_python_nested(val) for key, val in value.items()}
    if isinstance(value, list):
        return [_to_python_nested(item) for item in value]
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            return value
    return value


def build_signal_metadata(config: SignalConfig) -> tuple[pd.DataFrame, dict]:
    base = load_base_metadata(config.data_root)
    output_rows = []

    for stock, stock_meta in base.groupby("stock", sort=False):
        ohlc = fetch_stock_ohlc(stock, config)
        ohlc = compute_future_returns(ohlc, lookahead_days=config.lookahead_days)
        aligned = stock_meta.merge(
            ohlc[["future_close", "future_return"]],
            left_on="timestamp",
            right_index=True,
            how="left",
        )
        aligned = assign_labels(aligned, config)
        output_rows.append(aligned)

    meta = pd.concat(output_rows, ignore_index=True)
    meta = meta.dropna(subset=["label", "future_return"]).copy()

    if config.drop_missing_images:
        meta = meta[meta["resolved_path"].map(lambda path: Path(path).exists())].copy()

    meta = build_purged_split(meta, config.train_ratio, config.val_ratio, config.purge_gap)
    class_names = preferred_label_order(meta["label"])
    label_to_id = {label: idx for idx, label in enumerate(class_names)}
    meta["label_id"] = meta["label"].map(label_to_id)
    meta = meta.sort_values(["stock", "timestamp", "image_idx"]).reset_index(drop=True)

    summary = {
        "config": asdict(config),
        "rows": int(len(meta)),
        "stocks": int(meta["stock"].nunique()),
        "class_names": class_names,
        "label_counts": _to_python_nested(meta["label"].value_counts().to_dict()),
        "split_counts": _to_python_nested(
            meta.groupby(["split", "label"]).size().unstack(fill_value=0).to_dict(orient="index")
        ),
        "return_summary": _to_python_nested(meta["future_return"].describe().round(6).to_dict()),
    }
    return meta, summary


def save_signal_metadata(config: SignalConfig) -> tuple[Path, Path]:
    meta, summary = build_signal_metadata(config)
    output_csv = Path(config.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    meta.to_csv(output_csv, index=False)

    summary_path = output_csv.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2))
    return output_csv, summary_path


if __name__ == "__main__":
    config = SignalConfig()
    csv_path, summary_path = save_signal_metadata(config)
    print(f"Saved metadata to {csv_path}")
    print(f"Saved summary to {summary_path}")
