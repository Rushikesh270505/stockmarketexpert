# StockMarketExpert

We track all the Indian stocks.

## Tech Stack

- Python
- BeautifulSoup
- LLM (Ollama local models)
- Streamlit

## Expected Skills

- Sentiment analysis
- Finance

## Problem Overview

An agent that monitors social media (mock) and news sentiment to autonomously adjust a mock portfolio's risk level and draft buy/sell orders based on sentiment analysis.

## Key Objectives

- Automate sentiment-driven trading decisions
- Monitor market sentiment in near real-time
- Adjust portfolio risk dynamically
- Demonstrate agentic trading logic

## Requirements

- News and social media sentiment analysis
- Mock portfolio management
- Risk level adjustment algorithm
- Buy/sell order drafting logic

## Deliverables

1. Sentiment trading agent
2. Portfolio management dashboard
3. Demo with mock market data
4. Sentiment analysis documentation

## Quickstart

### 1) Install dependencies

```bash
pip install -r requirements.txt
```

### 2) Use existing LLMs, or download project models if needed

- If you already have a local Ollama model (recommended), the app will use it.
- If no models are installed, run:

```bash
make setup-models
```

Prerequisite:
- Install Ollama: https://ollama.com/

### 3) Run dashboard

```bash
make run
```

Open: [http://localhost:8501](http://localhost:8501)

## Notes

- App entrypoint: `app.py`
