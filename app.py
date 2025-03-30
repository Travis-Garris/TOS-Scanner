from flask import Flask, render_template, request
import requests
import pandas as pd
# import pandas_ta as ta
from datetime import datetime, timedelta
import json
import urllib.parse
import os
import sys

# sys.path.insert(0, './pandas_ta')
app = Flask(__name__, template_folder="templates")
API_KEY = "FSb95RQbEj22ehQKXRWk0eLFwwXWDo4Z"

def get_minute_data(symbol, multiplier=1, timespan="minute", limit=1000):
    end_date = datetime.now()
    start_date = end_date - timedelta(days=2)
    url = f"https://api.polygon.io/v2/aggs/ticker/{symbol}/range/{multiplier}/{timespan}/{start_date.date()}/{end_date.date()}"
    params = {"adjusted": "true", "apiKey": API_KEY}
    res = requests.get(url, params=params)
    data = res.json()

    if "results" not in data:
        return []
    return data["results"]

def build_range_bars(candles, range_size):
    bars = []
    current_bar = []
    if not candles:
        return []
    open_price = candles[0]['o']
    for c in candles:
        current_bar.append(c)
        high = max(x['h'] for x in current_bar)
        low = min(x['l'] for x in current_bar)
        if abs(high - low) >= range_size:
            close_price = current_bar[-1]['c']
            bars.append({
                "open": open_price,
                "high": high,
                "low": low,
                "close": close_price
            })
            current_bar = []
            open_price = close_price
    return bars

def compute_sve_stoch_rsi(closes, rsi_len=14, stoch_len=5, avg_len=8):
    close_series = pd.Series(closes)
    delta = close_series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(rsi_len).mean()
    avg_loss = loss.rolling(rsi_len).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))

    hi_rsi = rsi.rolling(stoch_len).max()
    lo_rsi = rsi.rolling(stoch_len).min()

    num = (rsi - lo_rsi).rolling(avg_len).mean()
    denom = (hi_rsi - lo_rsi).rolling(avg_len).mean() + 0.1

    stoch_rsi = (num / denom) * 100
    return stoch_rsi.fillna(0)

@app.route("/", methods=["GET", "POST"])
def index():
    try:
        results = []
        if request.method == "POST":
            tickers_input = request.form['tickers'].strip()
            if tickers_input:
                tickers = tickers_input.upper().split(',')
            else:
                tickers = ["AAPL", "MSFT", "GOOGL", "TSLA", "NVDA", "AMD", "META"]
            min_price = float(request.form.get("min_price", 0))
            max_price = float(request.form.get("max_price", 9999))

            for ticker in tickers:
                try:
                    candles = get_minute_data(ticker.strip())
                    if not candles:
                        continue

                    current_price = candles[-1]['c']
                    print(f"\nTicker: {ticker.strip()}, Price: {current_price}")
                    if not (min_price <= current_price <= max_price):
                        continue

                    match = False
                    chart_stoch = []

                    for i in range(1, 4):
                        range_pct = float(request.form.get(f'range_pct_{i}', 0))
                        print(f"    - Checking Config {i} | Range %: {range_pct}")
                        if range_pct == 0:
                            continue

                        rsi_len = int(request.form.get(f'rsi_length_{i}', 14))
                        stoch_len = int(request.form.get(f'stoch_length_{i}', 5))
                        avg_len = int(request.form.get(f'avg_length_{i}', 8))
                        overbought = float(request.form.get(f'overbought_{i}', 92))
                        oversold = float(request.form.get(f'oversold_{i}', 8))
                        condition = request.form.get(f'scan_type_{i}', 'less_than')

                        range_size = current_price * (range_pct / 100)
                        bars = build_range_bars(candles, range_size)
                        closes = [bar['close'] for bar in bars]

                        if len(closes) < rsi_len + stoch_len + avg_len + 2:
                            continue

                        stoch_rsi_series = compute_sve_stoch_rsi(closes, rsi_len, stoch_len, avg_len)
                        latest_value = stoch_rsi_series.iloc[-1]
                        prev_value = stoch_rsi_series.iloc[-2]

                        print(f"  Config {i} | StochRSI: {latest_value:.2f} | Prev: {prev_value:.2f} | Condition: {condition}")

                        if (
                            (condition == "less_than" and latest_value < oversold) or
                            (condition == "greater_than" and latest_value > overbought) or
                            (condition == "crosses_above" and prev_value < oversold and latest_value > oversold) or
                            (condition == "crosses_below" and prev_value > overbought and latest_value < overbought) or
                            (condition == "rising" and latest_value > prev_value) or
                            (condition == "falling" and latest_value < prev_value)
                        ):
                            chart_stoch = list(stoch_rsi_series[-20:])
                            match = True
                            break

                    if match:
                        chart_data = {
                            "type": "line",
                            "data": {
                                "labels": list(range(len(chart_stoch))),
                                "datasets": [{
                                    "label": "SVEStochRSI",
                                    "data": chart_stoch,
                                    "borderColor": "blue",
                                    "fill": False
                                }]
                            },
                            "options": {
                                "scales": {
                                    "x": {"display": False},
                                    "y": {"min": 0, "max": 100, "display": False}
                                },
                                "plugins": {
                                    "legend": {"display": False}
                                }
                            }
                        }
                        chart_url = urllib.parse.quote(json.dumps(chart_data))

                        results.append({
                            "ticker": ticker.strip(),
                            "price": current_price,
                            "stoch_rsi": latest_value,
                            "chart_url": chart_url
                        })
                        print(f"  --> MATCH! Added {ticker.strip()} to results\n")

                except Exception as e:
                    print(f"Error with {ticker}: {e}")
                    continue

        return render_template("index.html", results=results)
    except Exception as e:
        import traceback
        return f"<pre>{traceback.format_exc()}</pre>"

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)