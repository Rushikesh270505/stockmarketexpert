#!/usr/bin/env python3
"""
StockMarketExpert
Sentiment-driven (mock) Indian stock trading agent with Streamlit UI.
"""
from __future__ import annotations

import json
import os
import random
import re
import subprocess
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st
from bs4 import BeautifulSoup


STOCK_CATALOG: dict[str, dict[str, Any]] = {
    "RELIANCE": {"name": "Reliance Industries", "sector": "Energy", "base_price": 2940.0},
    "TCS": {"name": "Tata Consultancy Services", "sector": "IT", "base_price": 4030.0},
    "INFY": {"name": "Infosys", "sector": "IT", "base_price": 1680.0},
    "HDFCBANK": {"name": "HDFC Bank", "sector": "Banking", "base_price": 1650.0},
    "ICICIBANK": {"name": "ICICI Bank", "sector": "Banking", "base_price": 1065.0},
    "SBIN": {"name": "State Bank of India", "sector": "Banking", "base_price": 780.0},
    "LT": {"name": "Larsen & Toubro", "sector": "Infra", "base_price": 3600.0},
    "ITC": {"name": "ITC", "sector": "FMCG", "base_price": 420.0},
    "BHARTIARTL": {"name": "Bharti Airtel", "sector": "Telecom", "base_price": 1175.0},
    "HINDUNILVR": {"name": "Hindustan Unilever", "sector": "FMCG", "base_price": 2460.0},
    "ASIANPAINT": {"name": "Asian Paints", "sector": "Consumer", "base_price": 3010.0},
    "MARUTI": {"name": "Maruti Suzuki", "sector": "Auto", "base_price": 11100.0},
}

POSITIVE_WORDS = {
    "surge",
    "rally",
    "upgrade",
    "beat",
    "strong",
    "outperform",
    "record",
    "growth",
    "profit",
    "bullish",
    "buy",
    "expansion",
    "optimistic",
}

NEGATIVE_WORDS = {
    "fall",
    "selloff",
    "downgrade",
    "miss",
    "weak",
    "underperform",
    "loss",
    "concern",
    "bearish",
    "sell",
    "risk",
    "slowdown",
    "volatility",
}


def _parse_json_loose(text: str) -> dict[str, Any]:
    """Parse JSON even if extra text surrounds it."""
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


def lexicon_sentiment(text: str) -> float:
    tokens = re.findall(r"[a-zA-Z]+", text.lower())
    pos = sum(1 for t in tokens if t in POSITIVE_WORDS)
    neg = sum(1 for t in tokens if t in NEGATIVE_WORDS)
    if pos == 0 and neg == 0:
        return 0.0
    return (pos - neg) / float(pos + neg)


def _mock_sentiment_bucket(rng: random.Random) -> str:
    x = rng.uniform(-1.0, 1.0)
    if x > 0.3:
        return "positive"
    if x < -0.3:
        return "negative"
    return "neutral"


def build_mock_news_html(stocks: list[str], seed: int) -> str:
    rng = random.Random(seed + 100)
    positive_templates = [
        "{stock} posts strong quarterly growth; brokerages turn bullish",
        "{stock} wins major contract and signals expansion plans",
        "Institutional flows support {stock} as outlook improves",
    ]
    negative_templates = [
        "{stock} faces margin pressure amid sector slowdown concerns",
        "Analysts downgrade {stock} citing valuation risk",
        "{stock} underperforms peers after weak guidance",
    ]
    neutral_templates = [
        "{stock} trades range-bound as market awaits macro triggers",
        "{stock} sees mixed commentary from analysts",
        "{stock} remains stable ahead of earnings week",
    ]
    sources = ["Moneycontrol", "Economic Times", "Business Standard", "Mint"]

    blocks = ["<section id='news-feed'>"]
    for sym in stocks:
        bucket = _mock_sentiment_bucket(rng)
        if bucket == "positive":
            template = rng.choice(positive_templates)
        elif bucket == "negative":
            template = rng.choice(negative_templates)
        else:
            template = rng.choice(neutral_templates)
        headline = template.format(stock=sym)
        summary = f"Mock news signal for {sym} from monitored media stream."
        blocks.append(
            f"""
            <article class="news-item" data-stock="{sym}">
              <h3>{headline}</h3>
              <p>{summary}</p>
              <span class="source">{rng.choice(sources)}</span>
              <span class="time">{rng.randint(2, 58)}m ago</span>
            </article>
            """
        )
    blocks.append("</section>")
    return "".join(blocks)


LIVE_NEWS_FEEDS: list[tuple[str, str]] = [
    ("Economic Times", "https://economictimes.indiatimes.com/markets/rssfeeds/1977021501.cms"),
    ("Business Standard", "https://www.business-standard.com/rss/markets-106.rss"),
    ("Moneycontrol", "https://www.moneycontrol.com/rss/business.xml"),
    ("Mint", "https://www.livemint.com/rss/markets"),
]


def infer_stock_from_text(text: str, stocks: list[str]) -> str:
    text_u = (text or "").upper()
    for sym in stocks:
        if sym in text_u:
            return sym
        company = str(STOCK_CATALOG.get(sym, {}).get("name", "")).upper()
        company_tokens = [token for token in re.findall(r"[A-Z]+", company) if len(token) >= 4]
        if any(tok in text_u for tok in company_tokens):
            return sym
    return "MARKET"


def fetch_live_news_items(stocks: list[str], max_items: int = 30) -> pd.DataFrame:
    items: list[dict[str, Any]] = []
    for source, url in LIVE_NEWS_FEEDS:
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "StockMarketExpert/1.0"})
            with urllib.request.urlopen(req, timeout=8) as resp:
                xml_text = resp.read().decode("utf-8", errors="replace")
        except (urllib.error.URLError, TimeoutError, ValueError):
            continue

        # Streamlit Cloud images may not include an XML parser backend.
        try:
            soup = BeautifulSoup(xml_text, "xml")
        except Exception:
            soup = BeautifulSoup(xml_text, "html.parser")
        entries = soup.find_all("item")
        if not entries:
            entries = soup.find_all("entry")

        for entry in entries[:20]:
            title = entry.find("title").get_text(" ", strip=True) if entry.find("title") else ""
            desc_node = entry.find("description") or entry.find("summary") or entry.find("content")
            raw_summary = desc_node.get_text(" ", strip=True) if desc_node else ""
            summary = BeautifulSoup(raw_summary, "html.parser").get_text(" ", strip=True)
            time_node = entry.find("pubDate") or entry.find("updated")
            when = time_node.get_text(strip=True) if time_node else "recent"
            text = f"{title}. {summary}".strip(". ").strip()
            if not text:
                continue
            stock = infer_stock_from_text(text, stocks)
            items.append(
                {
                    "channel": "News",
                    "stock": stock,
                    "text": text[:420],
                    "source": source,
                    "time": when[:60],
                }
            )

    if not items:
        return pd.DataFrame(columns=["channel", "stock", "text", "source", "time"])

    news_df = pd.DataFrame(items).drop_duplicates(subset=["text"]).head(max_items).reset_index(drop=True)
    return news_df


def build_mock_social_html(stocks: list[str], seed: int) -> str:
    rng = random.Random(seed + 200)
    positive_posts = [
        "Retail chatter is bullish on {stock}; momentum buy calls rising.",
        "{stock} sentiment improving after positive management commentary.",
        "Traders expect breakout in {stock} if volumes hold.",
    ]
    negative_posts = [
        "Social sentiment turns bearish on {stock} after weak trend.",
        "{stock} seeing sell signals in community channels.",
        "Risk-off mood hits {stock}; users discussing downside protection.",
    ]
    neutral_posts = [
        "{stock} discussion remains mixed with hold bias.",
        "{stock} mentions are high but conviction is moderate.",
        "Community waiting for confirmation before taking exposure in {stock}.",
    ]
    handles = ["@marketpulse", "@equitybytes", "@niftywatch", "@traderdesk"]

    blocks = ["<section id='social-feed'>"]
    for sym in stocks:
        bucket = _mock_sentiment_bucket(rng)
        if bucket == "positive":
            template = rng.choice(positive_posts)
        elif bucket == "negative":
            template = rng.choice(negative_posts)
        else:
            template = rng.choice(neutral_posts)
        post = template.format(stock=sym)
        blocks.append(
            f"""
            <div class="social-item" data-stock="{sym}">
              <p>{post}</p>
              <span class="source">{rng.choice(handles)}</span>
              <span class="time">{rng.randint(1, 45)}m ago</span>
            </div>
            """
        )
    blocks.append("</section>")
    return "".join(blocks)


def parse_mock_html_items(html: str, channel: str) -> list[dict[str, Any]]:
    soup = BeautifulSoup(html, "html.parser")
    records: list[dict[str, Any]] = []
    if channel == "news":
        nodes = soup.select("article.news-item")
        for node in nodes:
            stock = (node.get("data-stock") or "").upper()
            headline = (node.select_one("h3").get_text(strip=True) if node.select_one("h3") else "")
            summary = (node.select_one("p").get_text(strip=True) if node.select_one("p") else "")
            source = (node.select_one(".source").get_text(strip=True) if node.select_one(".source") else "Unknown")
            when = (node.select_one(".time").get_text(strip=True) if node.select_one(".time") else "NA")
            text = f"{headline}. {summary}"
            records.append(
                {
                    "channel": "News",
                    "stock": stock,
                    "text": text,
                    "source": source,
                    "time": when,
                }
            )
    else:
        nodes = soup.select("div.social-item")
        for node in nodes:
            stock = (node.get("data-stock") or "").upper()
            post = (node.select_one("p").get_text(strip=True) if node.select_one("p") else "")
            source = (node.select_one(".source").get_text(strip=True) if node.select_one(".source") else "Unknown")
            when = (node.select_one(".time").get_text(strip=True) if node.select_one(".time") else "NA")
            records.append(
                {
                    "channel": "Social",
                    "stock": stock,
                    "text": post,
                    "source": source,
                    "time": when,
                }
            )
    return records


def generate_mock_market_data(stocks: list[str], seed: int) -> pd.DataFrame:
    rng = random.Random(seed + 300)
    rows: list[dict[str, Any]] = []
    for sym in stocks:
        meta = STOCK_CATALOG.get(sym, {})
        base = float(meta.get("base_price", rng.uniform(150.0, 5000.0)))
        ret = rng.uniform(-2.8, 2.8)
        price = round(base * (1 + ret / 100.0), 2)
        volatility = round(max(0.2, abs(rng.gauss(1.2, 0.55))), 2)
        volume_lakhs = round(rng.uniform(2.5, 65.0), 2)
        rows.append(
            {
                "stock": sym,
                "company": meta.get("name", sym),
                "sector": meta.get("sector", "Unknown"),
                "price_inr": price,
                "change_pct": round(ret, 2),
                "intraday_vol_pct": volatility,
                "volume_lakhs": volume_lakhs,
            }
        )
    return pd.DataFrame(rows)


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


def derive_risk_profile(sentiment_df: pd.DataFrame, market_df: pd.DataFrame) -> dict[str, Any]:
    if sentiment_df.empty:
        market_sentiment = 0.0
    else:
        weighted = sentiment_df["final_score"] * sentiment_df["mentions"]
        denom = float(sentiment_df["mentions"].sum()) or 1.0
        market_sentiment = float(weighted.sum()) / denom

    avg_vol = float(market_df["intraday_vol_pct"].mean()) if not market_df.empty else 1.0
    risk_score = 50.0 + (market_sentiment * 35.0) - (max(0.0, avg_vol - 1.2) * 12.0)
    risk_score = max(5.0, min(95.0, risk_score))

    if risk_score < 35:
        level = "LOW"
        risk_budget_pct = 0.08
    elif risk_score < 65:
        level = "MEDIUM"
        risk_budget_pct = 0.14
    else:
        level = "HIGH"
        risk_budget_pct = 0.22

    return {
        "risk_score": round(risk_score, 2),
        "level": level,
        "risk_budget_pct": risk_budget_pct,
        "market_sentiment": round(market_sentiment, 3),
        "avg_volatility": round(avg_vol, 3),
    }


def draft_orders(
    sentiment_df: pd.DataFrame,
    market_df: pd.DataFrame,
    holdings: dict[str, int],
    cash_inr: float,
    risk_profile: dict[str, Any],
) -> pd.DataFrame:
    if sentiment_df.empty or market_df.empty:
        return pd.DataFrame(columns=["stock", "action", "qty", "price", "reason", "risk_level"])

    price_map = {row["stock"]: float(row["price_inr"]) for _, row in market_df.iterrows()}
    risk_capital = cash_inr * float(risk_profile["risk_budget_pct"])

    orders: list[dict[str, Any]] = []
    ranked = sentiment_df.sort_values("final_score", ascending=False)
    for _, row in ranked.iterrows():
        stock = str(row["stock"])
        score = float(row["final_score"])
        confidence = float(row["confidence"]) / 100.0
        price = float(price_map.get(stock, 0.0))
        held_qty = int(holdings.get(stock, 0))
        if price <= 0:
            continue

        if score >= 0.24:
            allocation = risk_capital * min(0.35, 0.12 + (score * 0.40) + (confidence * 0.08))
            qty = int(allocation // price)
            if qty > 0:
                orders.append(
                    {
                        "stock": stock,
                        "action": "BUY",
                        "qty": qty,
                        "price": round(price, 2),
                        "reason": f"Bullish sentiment score {score:+.2f} with confidence {int(confidence * 100)}%",
                        "risk_level": risk_profile["level"],
                    }
                )
        elif score <= -0.24 and held_qty > 0:
            qty = max(1, int(held_qty * min(0.9, abs(score))))
            orders.append(
                {
                    "stock": stock,
                    "action": "SELL",
                    "qty": qty,
                    "price": round(price, 2),
                    "reason": f"Bearish sentiment score {score:+.2f}; de-risking exposure",
                    "risk_level": risk_profile["level"],
                }
            )

    if not orders:
        top = ranked.head(3)
        for _, row in top.iterrows():
            stock = str(row["stock"])
            price = float(price_map.get(stock, 0.0))
            orders.append(
                {
                    "stock": stock,
                    "action": "HOLD",
                    "qty": 0,
                    "price": round(price, 2),
                    "reason": "Signal not strong enough for execution",
                    "risk_level": risk_profile["level"],
                }
            )

    return pd.DataFrame(orders)


def init_portfolio(default_stocks: list[str]) -> None:
    if "portfolio_cash_inr" not in st.session_state:
        st.session_state.portfolio_cash_inr = 500000.0
    if "portfolio_holdings" not in st.session_state:
        st.session_state.portfolio_holdings = {
            default_stocks[0]: 22,
            default_stocks[1]: 14,
            default_stocks[2]: 18,
        }
    if "risk_history" not in st.session_state:
        st.session_state.risk_history = []
    if "order_history" not in st.session_state:
        st.session_state.order_history = []
    if "live_calls" not in st.session_state:
        st.session_state.live_calls = []
    if "active_call_id" not in st.session_state:
        st.session_state.active_call_id = None
    if "manual_nonce" not in st.session_state:
        st.session_state.manual_nonce = 0


def portfolio_snapshot(price_map: dict[str, float]) -> tuple[pd.DataFrame, float, float]:
    rows = []
    market_value = 0.0
    for stock, qty in st.session_state.portfolio_holdings.items():
        qty_i = int(qty)
        if qty_i <= 0:
            continue
        px = float(price_map.get(stock, STOCK_CATALOG.get(stock, {}).get("base_price", 0.0)))
        value = qty_i * px
        market_value += value
        rows.append(
            {
                "stock": stock,
                "qty": qty_i,
                "ltp_inr": round(px, 2),
                "value_inr": round(value, 2),
            }
        )
    cash = float(st.session_state.portfolio_cash_inr)
    total = cash + market_value
    return pd.DataFrame(rows), cash, total


def apply_drafted_orders(orders_df: pd.DataFrame) -> int:
    if orders_df.empty:
        return 0

    cash = float(st.session_state.portfolio_cash_inr)
    holdings = dict(st.session_state.portfolio_holdings)
    executed = 0
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    for _, row in orders_df.iterrows():
        action = str(row.get("action") or "").upper()
        stock = str(row.get("stock") or "")
        qty = int(row.get("qty") or 0)
        price = float(row.get("price") or 0.0)
        if qty <= 0 or price <= 0:
            continue

        if action == "BUY":
            cost = qty * price
            if cost <= cash:
                cash -= cost
                holdings[stock] = int(holdings.get(stock, 0)) + qty
                executed += 1
                st.session_state.order_history.append(
                    {"time": timestamp, "stock": stock, "action": "BUY", "qty": qty, "price": round(price, 2)}
                )
        elif action == "SELL":
            held = int(holdings.get(stock, 0))
            sell_qty = min(held, qty)
            if sell_qty > 0:
                cash += sell_qty * price
                holdings[stock] = held - sell_qty
                executed += 1
                st.session_state.order_history.append(
                    {"time": timestamp, "stock": stock, "action": "SELL", "qty": sell_qty, "price": round(price, 2)}
                )

    st.session_state.portfolio_cash_inr = cash
    st.session_state.portfolio_holdings = holdings
    st.session_state.order_history = st.session_state.order_history[-200:]
    return executed


def execute_single_order(order: dict[str, Any]) -> tuple[bool, str, str | None]:
    action = str(order.get("action") or "").upper()
    stock = str(order.get("stock") or "").upper()
    qty = int(order.get("qty") or 0)
    price = float(order.get("price") or 0.0)
    if action not in ("BUY", "SELL"):
        return False, "Only BUY/SELL orders can be executed.", None
    if qty <= 0 or price <= 0:
        return False, "Invalid order size or price.", None

    cash = float(st.session_state.portfolio_cash_inr)
    holdings = dict(st.session_state.portfolio_holdings)
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    executed_qty = qty

    if action == "BUY":
        cost = qty * price
        if cost > cash:
            return False, f"Insufficient cash for BUY {stock}. Need INR {cost:,.2f}.", None
        cash -= cost
        holdings[stock] = int(holdings.get(stock, 0)) + qty
    else:
        held = int(holdings.get(stock, 0))
        if held <= 0:
            return False, f"No holdings available to SELL for {stock}.", None
        executed_qty = min(held, qty)
        cash += executed_qty * price
        holdings[stock] = held - executed_qty

    st.session_state.portfolio_cash_inr = cash
    st.session_state.portfolio_holdings = holdings
    st.session_state.order_history.append(
        {"time": timestamp, "stock": stock, "action": action, "qty": executed_qty, "price": round(price, 2)}
    )
    st.session_state.order_history = st.session_state.order_history[-200:]

    call_id = f"{stock}_{action}_{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S%f')}"
    st.session_state.live_calls.insert(
        0,
        {
            "call_id": call_id,
            "stock": stock,
            "action": action,
            "qty": executed_qty,
            "entry_price": round(price, 2),
            "current_price": round(price, 2),
            "pnl_inr": 0.0,
            "pnl_pct": 0.0,
            "status": "OPEN",
            "entry_time": timestamp,
            "price_history": [{"time": datetime.now(timezone.utc).strftime("%H:%M:%S"), "price": round(price, 2)}],
        },
    )
    st.session_state.live_calls = st.session_state.live_calls[:80]
    st.session_state.active_call_id = call_id
    return True, f"Executed {action} {executed_qty} of {stock} at INR {price:,.2f}.", call_id


def update_live_calls(price_map: dict[str, float]) -> None:
    if "live_calls" not in st.session_state:
        st.session_state.live_calls = []
        return

    point_time = datetime.now(timezone.utc).strftime("%H:%M:%S")
    for call in st.session_state.live_calls:
        if call.get("status") != "OPEN":
            continue

        stock = str(call.get("stock") or "")
        latest = float(price_map.get(stock, call.get("current_price") or call.get("entry_price") or 0.0))
        if latest <= 0:
            continue

        entry = float(call.get("entry_price") or latest)
        qty = int(call.get("qty") or 0)
        action = str(call.get("action") or "BUY").upper()

        if action == "BUY":
            pnl = (latest - entry) * qty
        else:
            pnl = (entry - latest) * qty
        denom = entry * qty if (entry > 0 and qty > 0) else 0.0

        call["current_price"] = round(latest, 2)
        call["pnl_inr"] = round(pnl, 2)
        call["pnl_pct"] = round((pnl / denom) * 100.0, 2) if denom > 0 else 0.0

        history = call.setdefault("price_history", [])
        if not history or history[-1].get("time") != point_time:
            history.append({"time": point_time, "price": round(latest, 2)})
        if len(history) > 240:
            call["price_history"] = history[-240:]


def live_calls_summary_df() -> pd.DataFrame:
    rows = []
    for call in st.session_state.live_calls:
        rows.append(
            {
                "call_id": call.get("call_id"),
                "stock": call.get("stock"),
                "action": call.get("action"),
                "qty": int(call.get("qty") or 0),
                "entry_price": float(call.get("entry_price") or 0.0),
                "current_price": float(call.get("current_price") or 0.0),
                "pnl_inr": float(call.get("pnl_inr") or 0.0),
                "pnl_pct": float(call.get("pnl_pct") or 0.0),
                "status": call.get("status", "OPEN"),
                "entry_time": call.get("entry_time", ""),
            }
        )
    return pd.DataFrame(rows)


def inject_ui_styles() -> None:
    st.markdown(
        """
<style>
@import url('https://fonts.googleapis.com/css2?family=Manrope:wght@400;600;700;800&display=swap');

html, body, [class*="st-"] {
    font-family: "Manrope", "Segoe UI", sans-serif;
}

.block-container {
    padding-top: 1.1rem;
    padding-bottom: 2rem;
    max-width: 1300px;
}

.sme-hero {
    border: 1px solid rgba(56, 189, 248, 0.18);
    background: linear-gradient(130deg, rgba(15, 23, 42, 0.95), rgba(22, 78, 99, 0.35));
    border-radius: 14px;
    padding: 1.2rem 1.25rem;
    margin-bottom: 0.9rem;
}

.sme-hero h1 {
    margin: 0 0 0.3rem 0;
    font-size: 2rem;
    font-weight: 800;
    letter-spacing: 0.3px;
}

.sme-hero p {
    margin: 0;
    opacity: 0.88;
    font-size: 0.95rem;
}

.sme-kpi {
    border: 1px solid rgba(100, 116, 139, 0.28);
    background: rgba(15, 23, 42, 0.42);
    border-radius: 12px;
    padding: 0.72rem 0.75rem;
    min-height: 106px;
}

.sme-kpi .kpi-title {
    font-size: 0.82rem;
    opacity: 0.86;
    font-weight: 700;
    letter-spacing: 0.2px;
}

.sme-kpi .kpi-value {
    margin-top: 0.28rem;
    font-size: 1.45rem;
    font-weight: 800;
}

.sme-kpi .kpi-caption {
    margin-top: 0.2rem;
    font-size: 0.76rem;
    opacity: 0.78;
}

.sme-kpi.bull {
    border-color: rgba(16, 185, 129, 0.35);
}

.sme-kpi.bear {
    border-color: rgba(244, 63, 94, 0.35);
}

.sme-kpi.warn {
    border-color: rgba(245, 158, 11, 0.35);
}

.sme-section {
    margin-top: 0.25rem;
}
</style>
        """,
        unsafe_allow_html=True,
    )


def render_kpi_card(title: str, value: str, caption: str, tone: str = "neutral") -> None:
    tone_class = {
        "bull": "bull",
        "bear": "bear",
        "warn": "warn",
    }.get(tone, "")
    st.markdown(
        f"""
<div class="sme-kpi {tone_class}">
  <div class="kpi-title">{title}</div>
  <div class="kpi-value">{value}</div>
  <div class="kpi-caption">{caption}</div>
</div>
        """,
        unsafe_allow_html=True,
    )


def main() -> None:
    st.set_page_config(page_title="StockMarketExpert", page_icon="📈", layout="wide")
    inject_ui_styles()

    st.markdown(
        """
<div class="sme-hero">
  <h1>StockMarketExpert</h1>
  <p>We track all the Indian stocks with a sentiment agent, dynamic risk controls, and autonomous draft orders.</p>
</div>
        """,
        unsafe_allow_html=True,
    )

    default_watchlist = list(STOCK_CATALOG.keys())[:10]
    init_portfolio(default_watchlist)

    installed_models = list_ollama_models()
    configured_model = os.environ.get("SME_OLLAMA_MODEL")
    default_model = configured_model if configured_model in installed_models else (installed_models[0] if installed_models else "")
    model_index = installed_models.index(default_model) if default_model and default_model in installed_models else 0

    with st.sidebar:
        st.markdown("### Control Center")
        st.markdown("#### Market Monitor")
        watchlist = st.multiselect(
            "Stocks to monitor",
            options=list(STOCK_CATALOG.keys()),
            default=default_watchlist,
            help="Select NSE symbols to include in mock sentiment and trade generation.",
        )
        if not watchlist:
            watchlist = default_watchlist

        if st.button("Run refresh now", use_container_width=True):
            st.session_state.manual_nonce += 1
        news_interval_sec = st.slider("Live news update interval (seconds)", min_value=10, max_value=120, value=20, step=5)

        st.divider()
        st.markdown("#### LLM Engine")
        use_llm = st.checkbox("Use Ollama LLM", value=bool(installed_models))
        llm_model = st.selectbox(
            "LLM model",
            options=installed_models if installed_models else ["No local model found"],
            index=model_index if installed_models else 0,
            disabled=not bool(installed_models),
        )
        llm_limit = st.slider(
            "LLM analyses per refresh",
            min_value=0,
            max_value=len(watchlist),
            value=min(4, len(watchlist)),
        )

        if not installed_models:
            st.info("No local Ollama model found. Running lexicon fallback mode.")

        if st.button("Download project models", use_container_width=True):
            with st.spinner("Pulling models from ollama_models.txt ..."):
                proc = subprocess.run(
                    ["bash", "scripts/setup_models.sh"],
                    cwd=Path(__file__).parent,
                    capture_output=True,
                    text=True,
                    check=False,
                )
                if proc.returncode == 0:
                    st.success("Model setup completed. Refresh the page.")
                else:
                    st.error((proc.stderr or proc.stdout or "setup failed").strip())

        st.divider()
        st.markdown("#### Runtime Status")
        mode = "LLM + Lexicon" if (use_llm and bool(installed_models)) else "Lexicon Fallback"
        st.caption(f"Mode: {mode}")
        st.caption(f"Tracked universe: {len(watchlist)} stocks")
        st.caption(f"UTC: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')}")

    # Full-page refresh is disabled; we use in-place live updates for news.
    seed = int(datetime.now(timezone.utc).timestamp() // 60) + int(st.session_state.manual_nonce)
    market_df = generate_mock_market_data(watchlist, seed)
    price_map = {row["stock"]: float(row["price_inr"]) for _, row in market_df.iterrows()}
    update_live_calls(price_map)

    live_news_df = fetch_live_news_items(watchlist, max_items=30)
    if live_news_df.empty:
        # Fallback so the app remains usable if RSS sources are blocked.
        news_html = build_mock_news_html(watchlist, seed)
        live_news_df = pd.DataFrame(parse_mock_html_items(news_html, "news"))

    social_html = build_mock_social_html(watchlist, seed)
    social_df = pd.DataFrame(parse_mock_html_items(social_html, "social"))
    feed_df = pd.concat([live_news_df, social_df], ignore_index=True)

    selected_model = llm_model if llm_model in installed_models else None
    sentiment_df, llm_used = compute_sentiment_table(feed_df, selected_model, use_llm and selected_model is not None, llm_limit)
    risk_profile = derive_risk_profile(sentiment_df, market_df)

    st.session_state.risk_history.append(
        {
            "time": datetime.now(timezone.utc).strftime("%H:%M:%S"),
            "risk_score": risk_profile["risk_score"],
        }
    )
    st.session_state.risk_history = st.session_state.risk_history[-180:]

    orders_df = draft_orders(
        sentiment_df=sentiment_df,
        market_df=market_df,
        holdings=st.session_state.portfolio_holdings,
        cash_inr=float(st.session_state.portfolio_cash_inr),
        risk_profile=risk_profile,
    )

    sentiment_tone = "bull" if risk_profile["market_sentiment"] > 0.10 else ("bear" if risk_profile["market_sentiment"] < -0.10 else "warn")
    risk_tone = "bull" if risk_profile["level"] == "LOW" else ("warn" if risk_profile["level"] == "MEDIUM" else "bear")
    actionable = int((orders_df["action"] != "HOLD").sum()) if not orders_df.empty else 0

    top_cols = st.columns(5)
    with top_cols[0]:
        render_kpi_card("Tracked Stocks", str(len(watchlist)), "Indian equities in watchlist")
    with top_cols[1]:
        render_kpi_card("LLM Analyses", str(llm_used), "Per current refresh cycle")
    with top_cols[2]:
        render_kpi_card("Market Sentiment", f"{risk_profile['market_sentiment']:+.2f}", "Aggregate sentiment score", sentiment_tone)
    with top_cols[3]:
        render_kpi_card("Risk Posture", risk_profile["level"], f"Risk score {risk_profile['risk_score']:.1f}/100", risk_tone)
    with top_cols[4]:
        render_kpi_card("Actionable Orders", str(actionable), "BUY/SELL recommendations")

    tab_overview, tab_agent, tab_portfolio, tab_orders, tab_docs = st.tabs(
        ["Overview", "Sentiment Agent", "Portfolio Dashboard", "Drafted Orders", "Documentation"]
    )

    with tab_overview:
        left, middle, right = st.columns([1.2, 1.2, 1.0])
        with left:
            st.subheader("Top Opportunity Snapshot")
            if sentiment_df.empty:
                st.info("No signal data available.")
            else:
                quick_signals = sentiment_df[["stock", "final_score", "bias", "confidence"]].head(6)
                st.dataframe(quick_signals, use_container_width=True, hide_index=True)

        with middle:
            st.subheader("Market Movers (Mock)")
            movers = market_df[["stock", "sector", "price_inr", "change_pct"]].sort_values("change_pct", ascending=False)
            st.dataframe(movers.head(8), use_container_width=True, hide_index=True)

        with right:
            st.subheader("Agent Health")
            risk_pct = min(1.0, max(0.0, float(risk_profile["risk_score"]) / 100.0))
            st.progress(risk_pct, text=f"Risk score: {risk_profile['risk_score']}/100")
            sentiment_pct = min(1.0, max(0.0, (float(risk_profile["market_sentiment"]) + 1.0) / 2.0))
            st.progress(sentiment_pct, text=f"Sentiment intensity: {risk_profile['market_sentiment']:+.3f}")
            vol_norm = min(1.0, max(0.0, float(risk_profile["avg_volatility"]) / 3.0))
            st.progress(vol_norm, text=f"Avg volatility: {risk_profile['avg_volatility']:.2f}%")
            mode = "LLM + Lexicon" if (use_llm and selected_model is not None) else "Lexicon fallback"
            st.caption(f"Runtime mode: {mode}")

    with tab_agent:
        left, right = st.columns([1.35, 1.0])
        with left:
            st.subheader("Signal Matrix")
            if not sentiment_df.empty:
                sentiment_view = sentiment_df[
                    ["stock", "mentions", "lexicon_score", "llm_score", "final_score", "bias", "confidence", "llm_label"]
                ]
                st.dataframe(sentiment_view, use_container_width=True, hide_index=True)
                st.caption("Final score blends lexicon and LLM sentiment when LLM is available.")
                chart_df = sentiment_df.set_index("stock")[["final_score"]]
                st.line_chart(chart_df, use_container_width=True)
            else:
                st.info("No sentiment data available for this cycle.")

            with st.expander("Market Snapshot (Mock)", expanded=False):
                st.dataframe(market_df, use_container_width=True, hide_index=True)

        with right:
            st.subheader("Live News Feed")

            if hasattr(st, "fragment"):
                @st.fragment(run_every=f"{news_interval_sec}s")
                def _live_news_fragment() -> None:
                    current_news_df = fetch_live_news_items(watchlist, max_items=36)
                    if current_news_df.empty:
                        st.warning("Unable to fetch live RSS news right now. Check network or try again.")
                        return

                    stock_filter_live = st.selectbox("Stock filter", ["All"] + watchlist, key="live_news_stock_filter")
                    source_options = ["All"] + sorted(current_news_df["source"].dropna().astype(str).unique().tolist())
                    source_filter_live = st.selectbox("Source filter", source_options, key="live_news_source_filter")

                    news_view = current_news_df.copy()
                    if stock_filter_live != "All":
                        news_view = news_view[news_view["stock"] == stock_filter_live]
                    if source_filter_live != "All":
                        news_view = news_view[news_view["source"] == source_filter_live]

                    st.caption(f"Auto-updates every {news_interval_sec}s without full-page refresh.")
                    st.dataframe(
                        news_view[["stock", "source", "time", "text"]],
                        use_container_width=True,
                        hide_index=True,
                        height=320,
                    )

                _live_news_fragment()
            else:
                fallback_news_view = live_news_df.copy()
                st.caption("Live fragment updates are not available in this runtime.")
                st.dataframe(
                    fallback_news_view[["stock", "source", "time", "text"]],
                    use_container_width=True,
                    hide_index=True,
                    height=320,
                )

            with st.expander("Social Signal Stream (Mock)", expanded=False):
                social_view = social_df.copy()
                if not social_view.empty:
                    st.dataframe(
                        social_view[["stock", "source", "time", "text"]],
                        use_container_width=True,
                        hide_index=True,
                        height=180,
                    )
                else:
                    st.info("No social feed records in this cycle.")

            st.subheader("LLM Rationale")
            if sentiment_df.empty:
                st.info("No rationale available.")
            else:
                reason_view = sentiment_df[sentiment_df["llm_reason"].astype(str).str.len() > 0][
                    ["stock", "bias", "llm_label", "llm_reason"]
                ]
                if reason_view.empty:
                    st.info("No LLM rationale this cycle (fallback mode or limited calls).")
                else:
                    st.dataframe(reason_view, use_container_width=True, hide_index=True, height=220)

    with tab_portfolio:
        holdings_df, cash_inr, total_inr = portfolio_snapshot(price_map)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Cash (INR)", f"{cash_inr:,.2f}")
        c2.metric("Portfolio Value (INR)", f"{total_inr:,.2f}")
        c3.metric("Risk Budget", f"{risk_profile['risk_budget_pct'] * 100:.1f}% of cash")
        c4.metric("Holdings Count", int(holdings_df.shape[0]))

        left, right = st.columns([1.35, 1.0])
        with left:
            st.subheader("Current Holdings")
            if holdings_df.empty:
                st.info("No holdings in the mock portfolio yet.")
            else:
                st.dataframe(holdings_df, use_container_width=True, hide_index=True)
                st.line_chart(holdings_df.set_index("stock")[["value_inr"]], use_container_width=True)

                sector_rows = []
                for _, row in holdings_df.iterrows():
                    stock = str(row["stock"])
                    sector_rows.append({"sector": STOCK_CATALOG.get(stock, {}).get("sector", "Unknown"), "value_inr": float(row["value_inr"])})
                sector_df = pd.DataFrame(sector_rows).groupby("sector", as_index=False)["value_inr"].sum()
                if not sector_df.empty:
                    st.subheader("Sector Exposure")
                    st.line_chart(sector_df.set_index("sector")[["value_inr"]], use_container_width=True)

        with right:
            st.subheader("Dynamic Risk Adjustment")
            st.write(
                f"Risk level is **{risk_profile['level']}** (score `{risk_profile['risk_score']}`) using "
                f"sentiment `{risk_profile['market_sentiment']:+.3f}` and volatility `{risk_profile['avg_volatility']:.2f}%`."
            )
            risk_hist_df = pd.DataFrame(st.session_state.risk_history)
            if not risk_hist_df.empty:
                st.line_chart(risk_hist_df.set_index("time")[["risk_score"]], use_container_width=True)
            risk_bar = min(1.0, max(0.0, float(risk_profile["risk_score"]) / 100.0))
            st.progress(risk_bar, text=f"Current risk posture: {risk_profile['level']}")

    with tab_orders:
        left, right = st.columns([1.28, 1.06])
        actionable_df = orders_df[orders_df["action"] != "HOLD"] if not orders_df.empty else pd.DataFrame()

        with left:
            st.subheader("Autonomous Calls")
            st.caption("Each call has its own execute button.")
            if actionable_df.empty:
                st.info("No actionable orders in this refresh cycle.")
            else:
                for idx, row in actionable_df.reset_index(drop=True).iterrows():
                    with st.container(border=True):
                        stock = str(row["stock"])
                        action = str(row["action"]).upper()
                        qty = int(row["qty"])
                        price = float(row["price"])
                        reason = str(row["reason"])
                        risk_level = str(row["risk_level"])

                        c1, c2, c3 = st.columns([1.2, 1.0, 1.3])
                        c1.markdown(f"**{stock} · {action}**")
                        c1.caption(f"Qty: {qty} @ INR {price:,.2f}")
                        c2.metric("Risk", risk_level)
                        c3.caption(reason)

                        btn_key = f"exec_{idx}_{stock}_{action}_{qty}_{int(price * 100)}"
                        btn_label = f"Execute {action} {stock}"
                        if st.button(btn_label, key=btn_key, use_container_width=True):
                            ok, msg, call_id = execute_single_order(row.to_dict())
                            if ok:
                                st.session_state.active_call_id = call_id
                                st.success(msg)
                                st.rerun()
                            else:
                                st.error(msg)

            hold_df = orders_df[orders_df["action"] == "HOLD"] if not orders_df.empty else pd.DataFrame()
            if not hold_df.empty:
                with st.expander("Hold calls (no execution)", expanded=False):
                    st.dataframe(hold_df, use_container_width=True, hide_index=True)

        with right:
            st.subheader("Live Trade Monitor")
            live_df = live_calls_summary_df()
            if live_df.empty:
                st.info("Execute a BUY/SELL call to open the live stock chart and PnL tracker.")
            else:
                monitor_cols = ["stock", "action", "qty", "entry_price", "current_price", "pnl_inr", "pnl_pct", "status"]
                st.dataframe(live_df[monitor_cols], use_container_width=True, hide_index=True, height=200)

                call_options = live_df["call_id"].astype(str).tolist()
                label_map = {
                    str(row["call_id"]): f"{row['stock']} {row['action']} · Qty {int(row['qty'])}"
                    for _, row in live_df.iterrows()
                }
                active_id = st.session_state.active_call_id if st.session_state.active_call_id in call_options else call_options[0]
                selected_call_id = st.selectbox(
                    "Select call to monitor",
                    options=call_options,
                    index=call_options.index(active_id),
                    format_func=lambda cid: label_map.get(cid, cid),
                    key="selected_live_call",
                )
                st.session_state.active_call_id = selected_call_id

                selected_call = next((c for c in st.session_state.live_calls if c.get("call_id") == selected_call_id), None)
                if selected_call:
                    m1, m2, m3 = st.columns(3)
                    m1.metric("Entry", f"INR {float(selected_call.get('entry_price', 0.0)):,.2f}")
                    m2.metric("Live", f"INR {float(selected_call.get('current_price', 0.0)):,.2f}")
                    pnl_value = float(selected_call.get("pnl_inr", 0.0))
                    pnl_pct = float(selected_call.get("pnl_pct", 0.0))
                    m3.metric("Live PnL", f"INR {pnl_value:,.2f}", delta=f"{pnl_pct:+.2f}%")

                    hist_df = pd.DataFrame(selected_call.get("price_history", []))
                    if not hist_df.empty:
                        hist_df["step"] = range(1, len(hist_df) + 1)
                        st.line_chart(hist_df.set_index("step")[["price"]], use_container_width=True)
                    st.caption(f"Entry time: {selected_call.get('entry_time', '')}")

                st.markdown("#### Live PnL For Each Call")
                for call in st.session_state.live_calls[:8]:
                    call_label = f"{call.get('stock')} {call.get('action')} · Qty {int(call.get('qty', 0))}"
                    with st.expander(call_label, expanded=(call.get("call_id") == selected_call_id)):
                        k1, k2 = st.columns(2)
                        k1.metric("Current Price", f"INR {float(call.get('current_price', 0.0)):,.2f}")
                        k2.metric("PnL", f"INR {float(call.get('pnl_inr', 0.0)):,.2f}", delta=f"{float(call.get('pnl_pct', 0.0)):+.2f}%")
                        series = pd.DataFrame(call.get("price_history", []))
                        if not series.empty:
                            series["step"] = range(1, len(series) + 1)
                            st.line_chart(series.set_index("step")[["price"]], use_container_width=True)

        st.subheader("Order History")
        hist_df = pd.DataFrame(st.session_state.order_history)
        if hist_df.empty:
            st.info("No executed orders yet.")
        else:
            st.dataframe(hist_df.iloc[::-1], use_container_width=True, hide_index=True, height=260)

    with tab_docs:
        st.markdown(
            """
### Problem Overview
An agent monitors mock social media and mock news sentiment to autonomously adjust a mock portfolio risk level and draft buy/sell orders.

### Key Objectives
- Automate sentiment-driven trading decisions
- Monitor market sentiment in near real-time
- Adjust portfolio risk dynamically
- Demonstrate agentic trading logic

### Requirements Coverage
- News and social media sentiment analysis: done via BeautifulSoup parsing + sentiment engine
- Mock portfolio management: done in Streamlit session state
- Risk level adjustment algorithm: implemented in `derive_risk_profile`
- Buy/sell order drafting logic: implemented in `draft_orders`

### Deliverables
1. Sentiment trading agent
2. Portfolio management dashboard
3. Demo with mock market data
4. Sentiment analysis documentation
            """
        )


if __name__ == "__main__":
    main()
