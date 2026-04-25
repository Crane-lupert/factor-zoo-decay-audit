"""Mechanism classification for Chen-Zimmermann factors.

Each factor is classified into one of four mechanism labels matching the
Engelberg-McLean-Pontiff (2024) decomposition (with `data_mining_suspect`
added as a residual bucket per CLAUDE.md):

    behavioral             -- investor psychology / sentiment / attention / biases
    risk_premium           -- rational compensation for systematic risk
    mispricing             -- limits-to-arbitrage / frictions / slow info diffusion
    data_mining_suspect    -- weak ex-ante theory; pattern-mining vibes

Input text per signal: OAP doc columns
    LongDescription, Cat.Economic, Cat.Form, Authors, Year, Journal,
    Notes, Detailed Definition.

3-model ensemble. Majority vote = ensemble label. Pairwise Cohen's kappa +
oracle kappa (CLAUDE.md Day-2 gate: oracle kappa > 0.7).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Iterable

import pandas as pd

LABELS = ("behavioral", "risk_premium", "mispricing", "data_mining_suspect")
LABEL_SET = set(LABELS)

# Short, vendor-neutral models — picked for ensemble diversity (1 each from
# Anthropic / OpenAI / Google).
DEFAULT_ENSEMBLE = (
    "anthropic/claude-haiku-4-5",
    "openai/gpt-4o-mini",
    "google/gemini-2.5-flash-lite",
)

PROMPT_TEMPLATE = """\
You are an asset-pricing analyst classifying a published equity-return predictor
into ONE of four mechanism categories per Engelberg-McLean-Pontiff (2024)-style
taxonomy. Read the description below and respond ONLY with a JSON object -- no
extra prose.

Categories (pick exactly one):

  - "behavioral" : driven by investor psychology. Limited attention, anchoring,
      sentiment, overreaction, underreaction, lottery preferences, disposition.
      The signal works because investors systematically MISPERCEIVE information.
      Canonical examples: post-earnings announcement drift; price momentum
      (slow diffusion / under-reaction); short-term reversal (over-reaction);
      MAX-return lottery preference; analyst sluggishness; investor inattention.

  - "risk_premium" : rational compensation for exposure to a priced systematic
      risk factor or discount-rate channel (distress, intermediary capital,
      q-theory, hedging demand). Canonical examples: size (Banz/FF distress),
      book-to-market (FF risk interpretation), gross profitability (Novy-Marx
      expected-profitability), investment / asset growth (q-theory discount
      rate), CAPM beta, low-beta / BAB (Frazzini-Pedersen leverage constraint),
      coskewness, intermediary capital factors.

  - "mispricing" : a documented mispricing sustained by limits-to-arbitrage /
      frictions -- short-sale constraints, transaction costs, capacity limits,
      or information frictions. NOT primarily a psychology story. Canonical
      examples: Sloan's accruals; abnormal/discretionary accruals (Xie); equity
      issuance puzzle (Daniel-Titman, Bradshaw-Richardson-Sloan); idiosyncratic
      volatility puzzle (LTA-attributed); Piotroski F-score; fundamental-value
      (V/P) signals; net stock issues; composite mispricing scores.

  - "data_mining_suspect" : RARE residual category. Use ONLY when the
      description provides no published economic rationale at all -- pure
      statistical pattern from data exploration. Do NOT assign this just
      because the formula looks complex or composite: virtually every
      Chen-Zimmermann predictor was motivated by a paper with a stated story.
      If you can think of any plausible behavioral / risk / mispricing reading
      from the description, prefer that over data_mining_suspect.

Tie-break:
  * Psychology rationale (attention, anchoring, etc.) -> behavioral
  * Priced systematic risk / discount-rate channel    -> risk_premium
  * Known mispricing sustained by frictions / LTA     -> mispricing
  * Truly no documented economic story                -> data_mining_suspect
    (this should apply to <= 10% of factors)

Predictor:
  Acronym         : {acronym}
  Title           : {long_description}
  Authors / Year  : {authors} / {year}
  Journal         : {journal}
  OAP Cat.Economic: {cat_economic}
  Cat.Form        : {cat_form}
  Notes (OAP)     : {notes}
  Definition      : {definition}

Respond with this JSON shape only:
{{"label": "<one of: behavioral, risk_premium, mispricing, data_mining_suspect>",
  "confidence": <float in [0,1]>,
  "rationale": "<<= 25 words>"}}
"""

DEFINITION_CHAR_LIMIT = 600
NOTES_CHAR_LIMIT = 300


def _clip(s: object, n: int) -> str:
    if s is None:
        return ""
    s = str(s)
    if s == "nan":
        return ""
    return s if len(s) <= n else s[:n].rstrip() + " ..."


def build_prompt(row: pd.Series | dict) -> str:
    get = row.get if hasattr(row, "get") else (lambda k, d="": row.get(k, d))  # type: ignore[union-attr]
    return PROMPT_TEMPLATE.format(
        acronym=get("Acronym", ""),
        long_description=_clip(get("LongDescription", ""), 100),
        authors=_clip(get("Authors", ""), 80),
        year=get("Year", ""),
        journal=_clip(get("Journal", ""), 30),
        cat_economic=_clip(get("Cat.Economic", ""), 60),
        cat_form=_clip(get("Cat.Form", ""), 30),
        notes=_clip(get("Notes", ""), NOTES_CHAR_LIMIT),
        definition=_clip(get("Detailed Definition", ""), DEFINITION_CHAR_LIMIT),
    )


@dataclass(frozen=True)
class Classification:
    acronym: str
    model: str
    label: str          # one of LABELS, or "PARSE_FAIL"
    confidence: float
    rationale: str
    raw: str            # raw model output for audit
    cost_usd: float
    tokens_in: int
    tokens_out: int


_JSON_BLOCK_RE = re.compile(r"\{[^{}]*\}", re.DOTALL)


def parse_response(raw_text: str) -> tuple[str, float, str]:
    """Best-effort parse of the JSON body. Returns (label, confidence, rationale).

    Strategy: find the first {...} block, json.loads it, validate label.
    Falls back to label="PARSE_FAIL" on any failure.
    """
    text = raw_text.strip()

    # strip code fences if present
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.MULTILINE).strip()

    candidates: list[str] = []
    if text.startswith("{") and text.endswith("}"):
        candidates.append(text)
    candidates.extend(_JSON_BLOCK_RE.findall(text))

    for cand in candidates:
        try:
            obj = json.loads(cand)
        except Exception:
            continue
        label = str(obj.get("label", "")).strip().lower().replace("-", "_").replace(" ", "_")
        if label in LABEL_SET:
            try:
                conf = float(obj.get("confidence", 0.0))
            except Exception:
                conf = 0.0
            rationale = str(obj.get("rationale", ""))[:200]
            return label, conf, rationale

    return "PARSE_FAIL", 0.0, raw_text[:200]


def classify_one(client, model: str, row: pd.Series | dict) -> Classification:
    prompt = build_prompt(row)
    resp = client.complete(model=model, prompt=prompt, max_tokens=200, temperature=0.0)
    choice = resp["choices"][0]["message"]["content"]
    label, conf, rationale = parse_response(choice)
    usage = resp.get("usage", {}) or {}
    return Classification(
        acronym=str(row["Acronym"] if hasattr(row, "__getitem__") else row.get("Acronym", "")),
        model=model,
        label=label,
        confidence=conf,
        rationale=rationale,
        raw=choice[:1000],
        cost_usd=float(usage.get("cost", 0.0)),
        tokens_in=int(usage.get("prompt_tokens", 0)),
        tokens_out=int(usage.get("completion_tokens", 0)),
    )


def classify_ensemble(
    client,
    rows: Iterable[pd.Series | dict],
    models: tuple[str, ...] = DEFAULT_ENSEMBLE,
) -> list[Classification]:
    out: list[Classification] = []
    for row in rows:
        for m in models:
            out.append(classify_one(client, m, row))
    return out


def majority_label(model_labels: list[str]) -> tuple[str, int]:
    """Return (winning label, vote count). Ties broken by LABELS order."""
    counts: dict[str, int] = {}
    for lab in model_labels:
        if lab in LABEL_SET:
            counts[lab] = counts.get(lab, 0) + 1
    if not counts:
        return "PARSE_FAIL", 0
    best = max(counts.items(), key=lambda kv: (kv[1], -LABELS.index(kv[0])))
    return best[0], best[1]
