import pandas as pd
import streamlit as st
from datetime import datetime, timezone
from typing import Any
from config import STOCK_CATALOG

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
