#!/usr/bin/env python3
"""
StockMarketExpert
Modularized sentiment-driven Indian stock trading agent with Streamlit UI.
"""
from __future__ import annotations

import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import streamlit as st

# Import new modules
from config import STOCK_CATALOG
from ui_components import inject_ui_styles, render_kpi_card
from data_pipeline import (
    generate_mock_market_data,
    fetch_live_news_items,
    build_mock_news_html,
    build_mock_social_html,
    parse_mock_html_items
)
from llm_engine import (
    list_ollama_models,
    compute_sentiment_table
)
from portfolio_agent import (
    init_portfolio,
    update_live_calls,
    derive_risk_profile,
    draft_orders,
    portfolio_snapshot,
    execute_single_order,
    live_calls_summary_df
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
        # Groq/Gemini toggles as requested by user
        llm_provider = st.radio("LLM Provider", ["Ollama", "Groq", "Gemini"], index=0)
        
        use_llm = False
        llm_model = None
        
        if llm_provider == "Ollama":
            use_llm = st.checkbox("Use Ollama LLM", value=bool(installed_models))
            llm_model = st.selectbox(
                "LLM model",
                options=installed_models if installed_models else ["No local model found"],
                index=model_index if installed_models else 0,
                disabled=not bool(installed_models),
            )
            if not installed_models:
                st.info("No local Ollama model found. Running lexicon fallback mode.")
        elif llm_provider == "Groq":
            st.info("Groq integration is currently in placeholder mode.")
            use_llm = True
            llm_model = "groq-placeholder"
        elif llm_provider == "Gemini":
            st.info("Gemini integration is currently in placeholder mode.")
            use_llm = True
            llm_model = "gemini-placeholder"

        llm_limit = st.slider(
            "LLM analyses per refresh",
            min_value=0,
            max_value=len(watchlist),
            value=min(4, len(watchlist)),
        )

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
        mode = f"{llm_provider} + Lexicon" if use_llm else "Lexicon Fallback"
        st.caption(f"Mode: {mode}")
        st.caption(f"Tracked universe: {len(watchlist)} stocks")
        st.caption(f"UTC: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')}")

    # Data Pipeline and LLM Engine execution
    seed = int(datetime.now(timezone.utc).timestamp() // 60) + int(st.session_state.manual_nonce)
    market_df = generate_mock_market_data(watchlist, seed)
    price_map = {row["stock"]: float(row["price_inr"]) for _, row in market_df.iterrows()}
    update_live_calls(price_map)

    live_news_df = fetch_live_news_items(watchlist, max_items=30)
    if live_news_df.empty:
        news_html = build_mock_news_html(watchlist, seed)
        live_news_df = pd.DataFrame(parse_mock_html_items(news_html, "news"))

    social_html = build_mock_social_html(watchlist, seed)
    social_df = pd.DataFrame(parse_mock_html_items(social_html, "social"))
    feed_df = pd.concat([live_news_df, social_df], ignore_index=True)

    sentiment_df, llm_used = compute_sentiment_table(feed_df, llm_model, use_llm, llm_limit)
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

    # UI Rendering
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

### Deliverables
1. Sentiment trading agent
2. Portfolio management dashboard
3. Demo with mock market data
4. Sentiment analysis documentation
            """
        )

if __name__ == "__main__":
    main()
