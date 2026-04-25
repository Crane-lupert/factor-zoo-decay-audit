"""Load Chen-Zimmermann Open Asset Pricing data and cache to parquet.

The first call to each loader downloads from the OAP S3 bucket
(~30s for portfolio returns). Subsequent calls hit the local parquet cache.

Layout (relative to repo root):
    cache/oap_signal_doc.parquet   - 331-row predictor metadata
    cache/oap_op_returns.parquet   - long+short+deciles monthly returns
    cache/oap_ls_returns.parquet   - LS-only slice (filter on port == 'LS')
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
CACHE_DIR = REPO_ROOT / "cache"

SIGNAL_DOC_PARQUET = CACHE_DIR / "oap_signal_doc.parquet"
OP_RETURNS_PARQUET = CACHE_DIR / "oap_op_returns.parquet"
LS_RETURNS_PARQUET = CACHE_DIR / "oap_ls_returns.parquet"


def _ensure_cache_dir() -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)


def load_signal_doc(*, force: bool = False) -> pd.DataFrame:
    """Predictor-level metadata (331 rows: Acronym, Authors, Year, SampleStart/EndYear, Sign, Return, T-Stat, ...)."""
    if SIGNAL_DOC_PARQUET.exists() and not force:
        return pd.read_parquet(SIGNAL_DOC_PARQUET)

    import openassetpricing as oap

    _ensure_cache_dir()
    op = oap.OpenAP()
    df = op.dl_signal_doc("pandas")
    df.to_parquet(SIGNAL_DOC_PARQUET, index=False)
    return df


def load_op_returns(*, force: bool = False) -> pd.DataFrame:
    """All original-paper portfolio returns (LS + decile/quintile rows). Columns:
    signalname, port, date, ret, signallag, Nlong, Nshort. `ret` is in percent.
    """
    if OP_RETURNS_PARQUET.exists() and not force:
        return pd.read_parquet(OP_RETURNS_PARQUET)

    import openassetpricing as oap

    _ensure_cache_dir()
    op = oap.OpenAP()
    df = op.dl_port("op", "pandas")
    df.to_parquet(OP_RETURNS_PARQUET, index=False)
    return df


def load_ls_returns(*, force: bool = False) -> pd.DataFrame:
    """LS-only slice — one row per (signal, month). Columns: signalname, date, ret."""
    if LS_RETURNS_PARQUET.exists() and not force:
        return pd.read_parquet(LS_RETURNS_PARQUET)

    df = load_op_returns(force=force)
    ls = df.loc[df["port"] == "LS", ["signalname", "date", "ret"]].copy()
    ls = ls.sort_values(["signalname", "date"]).reset_index(drop=True)
    _ensure_cache_dir()
    ls.to_parquet(LS_RETURNS_PARQUET, index=False)
    return ls


def load_all(*, force: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Convenience: (LS returns, signal doc)."""
    return load_ls_returns(force=force), load_signal_doc(force=force)
