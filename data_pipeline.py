import random
import re
import urllib.request
import urllib.error
import pandas as pd
from bs4 import BeautifulSoup
from config import STOCK_CATALOG, LIVE_NEWS_FEEDS

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
