import json
import subprocess
import re
import pandas as pd
from typing import Any
from config import POSITIVE_WORDS, NEGATIVE_WORDS

def _parse_json_loose(text: str) -> dict[str, Any]:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            return json.loads(text[start : end + 1])
        raise

def list_ollama_models() -> list[str]:
    try:
        proc = subprocess.run(
            ["ollama", "list"],
            capture_output=True,
            text=True,
            timeout=8,
            check=False,
        )
    except (FileNotFoundError, subprocess.SubprocessError):
        return []

    if proc.returncode != 0:
        return []

    lines = [ln.strip() for ln in proc.stdout.splitlines() if ln.strip()]
    if not lines:
        return []

    models: list[str] = []
    for ln in lines[1:]:
        cols = ln.split()
        if cols:
            models.append(cols[0])
    return models

def run_ollama_sentiment(model: str, stock: str, snippets: list[str]) -> tuple[float | None, str, str]:
    prompt = (
        "You are a financial sentiment analyst for Indian stocks.\n"
        "Classify sentiment for the given stock using ONLY JSON.\n"
        'Return {"score": number_between_-1_and_1, "label":"bullish|neutral|bearish", "reason":"short reason"}.\n'
        f"Stock: {stock}\n"
        "Texts:\n"
        + "\n".join(f"- {s}" for s in snippets[:8])
    )
    try:
        proc = subprocess.run(
            ["ollama", "run", model, prompt],
            capture_output=True,
            text=True,
            timeout=35,
            check=False,
        )
        if proc.returncode != 0:
            return None, "unavailable", (proc.stderr or "llm_failed").strip()
        data = _parse_json_loose(proc.stdout.strip())
        score = float(data.get("score"))
        score = max(-1.0, min(1.0, score))
        label = str(data.get("label") or "neutral").lower()
        reason = str(data.get("reason") or "")
        return score, label, reason
    except Exception as exc:
        return None, "unavailable", str(exc)

def run_groq_sentiment(stock: str, snippets: list[str]) -> tuple[float | None, str, str]:
    # Placeholder for Groq integration
    return None, "unavailable", "Groq integration not implemented yet"

def run_gemini_sentiment(stock: str, snippets: list[str]) -> tuple[float | None, str, str]:
    # Placeholder for Gemini integration
    return None, "unavailable", "Gemini integration not implemented yet"

def lexicon_sentiment(text: str) -> float:
    tokens = re.findall(r"[a-zA-Z]+", text.lower())
    pos = sum(1 for t in tokens if t in POSITIVE_WORDS)
    neg = sum(1 for t in tokens if t in NEGATIVE_WORDS)
    if pos == 0 and neg == 0:
        return 0.0
    return (pos - neg) / float(pos + neg)

def compute_sentiment_table(
    feed_df: pd.DataFrame,
    llm_model: str | None,
    use_llm: bool,
    llm_calls_limit: int,
) -> tuple[pd.DataFrame, int]:
    if feed_df.empty:
        return pd.DataFrame(), 0

    llm_calls = 0
    rows: list[dict[str, Any]] = []
    grouped = feed_df.groupby("stock", dropna=False)
    for stock, gdf in grouped:
        snippets = gdf["text"].astype(str).tolist()
        lex_scores = [lexicon_sentiment(s) for s in snippets]
        lex_score = sum(lex_scores) / len(lex_scores) if lex_scores else 0.0

        llm_score = None
        llm_label = "not-used"
        llm_reason = ""
        if use_llm and llm_model and llm_calls < llm_calls_limit:
            llm_score, llm_label, llm_reason = run_ollama_sentiment(llm_model, stock, snippets)
            if llm_score is not None:
                llm_calls += 1

        final_score = lex_score if llm_score is None else (0.65 * lex_score + 0.35 * llm_score)
        final_score = max(-1.0, min(1.0, final_score))
        confidence = int(min(95, max(52, 56 + abs(final_score) * 35 + min(10, len(snippets) * 2))))

        if final_score > 0.18:
            bias = "Bullish"
        elif final_score < -0.18:
            bias = "Bearish"
        else:
            bias = "Neutral"

        rows.append(
            {
                "stock": stock,
                "mentions": len(snippets),
                "lexicon_score": round(lex_score, 3),
                "llm_score": None if llm_score is None else round(llm_score, 3),
                "final_score": round(final_score, 3),
                "bias": bias,
                "confidence": confidence,
                "llm_label": llm_label,
                "llm_reason": llm_reason,
            }
        )

    out = pd.DataFrame(rows).sort_values("final_score", ascending=False)
    return out, llm_calls
