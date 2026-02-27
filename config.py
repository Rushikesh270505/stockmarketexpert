from typing import Any

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

LIVE_NEWS_FEEDS: list[tuple[str, str]] = [
    ("Economic Times", "https://economictimes.indiatimes.com/markets/rssfeeds/1977021501.cms"),
    ("Business Standard", "https://www.business-standard.com/rss/markets-106.rss"),
    ("Moneycontrol", "https://www.moneycontrol.com/rss/business.xml"),
    ("Mint", "https://www.livemint.com/rss/markets"),
]
