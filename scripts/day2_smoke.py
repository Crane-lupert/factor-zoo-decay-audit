"""Smoke-test mechanism classification on 5 factors x 3 models. Verifies:

  1. OpenRouter connectivity + budget tracking
  2. Each model's response parses to a valid LABEL
  3. Total cost is small enough to scale to 212 factors
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import os  # noqa: E402

from dotenv import load_dotenv  # noqa: E402

# load coord-level .env (has OPENROUTER_API_KEY)
load_dotenv(Path(os.environ.get("PORTFOLIO_COORD_ROOT", "D:/vscode/portfolio-coordination")) / ".env")

from shared_utils.openrouter_client import OpenRouterClient  # noqa: E402

from factor_zoo_decay_audit.load import load_signal_doc, load_ls_returns  # noqa: E402
from factor_zoo_decay_audit.mechanism import (  # noqa: E402
    DEFAULT_ENSEMBLE,
    classify_one,
)


def main() -> int:
    doc = load_signal_doc()
    ls = load_ls_returns()
    covered = doc[doc["Acronym"].isin(ls["signalname"].unique())].copy()
    sample = covered.head(5)

    client = OpenRouterClient(project="F")
    print(f"Smoke test: {len(sample)} factors x {len(DEFAULT_ENSEMBLE)} models = {len(sample)*len(DEFAULT_ENSEMBLE)} calls")
    total_cost = 0.0
    fails = 0
    for _, row in sample.iterrows():
        for model in DEFAULT_ENSEMBLE:
            try:
                c = classify_one(client, model, row)
            except Exception as e:
                print(f"  [{row['Acronym']:<22}] {model:<35} ERROR: {e!r}")
                fails += 1
                continue
            total_cost += c.cost_usd
            print(
                f"  [{row['Acronym']:<22}] {c.model:<35} -> {c.label:<22} "
                f"conf={c.confidence:.2f}  ${c.cost_usd:.5f}  "
                f"in={c.tokens_in:>4} out={c.tokens_out:>3}"
            )
            if c.label == "PARSE_FAIL":
                fails += 1
                print(f"      raw: {c.raw[:200]}")
    print()
    print(f"smoke total cost: ${total_cost:.4f}    parse fails: {fails}")
    print(f"projected 212-factor cost: ${total_cost * 212 / len(sample):.4f}")
    return 0 if fails == 0 else 2


if __name__ == "__main__":
    sys.exit(main())
