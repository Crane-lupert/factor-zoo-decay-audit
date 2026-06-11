"""Day-2: classify all covered factors with the 3-model ensemble.

Saves incrementally to results/day2/classifications.parquet so we can resume
on transient failures. Each row is one (factor, model) call.
"""

from __future__ import annotations

import sys
import time
from dataclasses import asdict
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import os  # noqa: E402

from dotenv import load_dotenv  # noqa: E402

load_dotenv(Path(os.environ.get("PORTFOLIO_COORD_ROOT", "D:/vscode/portfolio-coordination")) / ".env")

from shared_utils.openrouter_client import OpenRouterClient  # noqa: E402

from factor_zoo_decay_audit.load import load_ls_returns, load_signal_doc  # noqa: E402
from factor_zoo_decay_audit.mechanism import (  # noqa: E402
    DEFAULT_ENSEMBLE,
    classify_one,
)

OUT_DIR = REPO_ROOT / "results" / "day2"
CLASSIFICATIONS_PARQUET = OUT_DIR / "classifications.parquet"


def _load_existing() -> pd.DataFrame:
    if CLASSIFICATIONS_PARQUET.exists():
        return pd.read_parquet(CLASSIFICATIONS_PARQUET)
    return pd.DataFrame(
        columns=[
            "acronym", "model", "label", "confidence", "rationale",
            "raw", "cost_usd", "tokens_in", "tokens_out",
        ]
    )


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    doc = load_signal_doc()
    ls = load_ls_returns()
    covered = doc[doc["Acronym"].isin(ls["signalname"].unique())].copy()
    print(f"factors covered: {len(covered)}    models: {len(DEFAULT_ENSEMBLE)}")

    existing = _load_existing()
    done = set(zip(existing["acronym"], existing["model"]))
    print(f"already done    : {len(done)} rows")

    client = OpenRouterClient(project="F")
    new_rows: list[dict] = []
    total_cost = float(existing["cost_usd"].sum()) if len(existing) else 0.0
    save_every = 25  # checkpoint every 25 calls
    t0 = time.time()
    n_calls = 0
    n_fail_parse = 0
    n_fail_call = 0

    for _, row in covered.iterrows():
        for model in DEFAULT_ENSEMBLE:
            key = (str(row["Acronym"]), model)
            if key in done:
                continue
            try:
                c = classify_one(client, model, row)
            except Exception as e:
                print(f"  ERR  [{row['Acronym']:<22}] {model:<35} {e!r}")
                n_fail_call += 1
                continue
            d = asdict(c)
            new_rows.append(d)
            done.add(key)
            total_cost += c.cost_usd
            n_calls += 1
            if c.label == "PARSE_FAIL":
                n_fail_parse += 1

            if n_calls % save_every == 0:
                merged = pd.concat([existing, pd.DataFrame(new_rows)], ignore_index=True)
                merged.to_parquet(CLASSIFICATIONS_PARQUET, index=False)
                elapsed = time.time() - t0
                rate = n_calls / elapsed if elapsed else 0
                remaining = (len(covered) * len(DEFAULT_ENSEMBLE) - len(merged)) / rate if rate else float("inf")
                print(
                    f"  [{n_calls:>4} new / {len(merged):>4} total] "
                    f"${total_cost:.3f}  fails(parse={n_fail_parse}, call={n_fail_call})  "
                    f"rate={rate:.1f}/s  eta={remaining/60:.1f}m"
                )

    if new_rows:
        merged = pd.concat([existing, pd.DataFrame(new_rows)], ignore_index=True)
        merged.to_parquet(CLASSIFICATIONS_PARQUET, index=False)
    else:
        merged = existing

    elapsed = time.time() - t0
    print()
    print(f"DONE: {len(merged)} rows total ({n_calls} new in this run)")
    print(f"      ${total_cost:.4f} total cost  parse_fails={n_fail_parse}  call_fails={n_fail_call}")
    print(f"      elapsed: {elapsed/60:.1f} min")

    coverage = (
        merged.groupby("model")["label"]
        .apply(lambda s: (s != "PARSE_FAIL").sum())
        .to_dict()
    )
    print(f"      per-model parsed-OK count: {coverage}")
    return 0 if n_fail_parse + n_fail_call == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
