
import os, json
from datetime import datetime, timezone
from pathlib import Path

import httpx
import numpy as np
import pandas as pd
import yfinance as yf

from fastapi import FastAPI, Query, Form
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse


ALPACA_API_KEY = os.getenv("ALPACA_API_KEY", "")
ALPACA_SECRET_KEY = os.getenv("ALPACA_SECRET_KEY", "")
ALPACA_BASE_URL = os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")

AUTO_TRADE = os.getenv("AUTO_TRADE", "false").lower() == "true"
AUTO_TRADE_RUNTIME = {"enabled": AUTO_TRADE}

AUTO_REFRESH_SECONDS = int(os.getenv("AUTO_REFRESH_SECONDS", "60"))
MAX_TRADES_PER_DAY = int(os.getenv("MAX_TRADES_PER_DAY", "10"))
MAX_OPEN_POSITIONS = int(os.getenv("MAX_OPEN_POSITIONS", "4"))
MIN_CONFIDENCE = float(os.getenv("MIN_CONFIDENCE", "62"))

RISK_PER_TRADE_PCT = float(os.getenv("RISK_PER_TRADE_PCT", "0.005"))  # 0.5%
RR_RATIO = float(os.getenv("RR_RATIO", "1.5"))
MAX_SPREAD_PCT = float(os.getenv("MAX_SPREAD_PCT", "0.20"))
NEWS_BLACKOUT = os.getenv("NEWS_BLACKOUT", "false").lower() == "true"

STATE_FILE = Path(os.getenv("STATE_FILE", "trade_state.json"))

app = FastAPI(title="Paper Scalping Bot - Alpaca By Abbas")


ASSETS = {
    "GOLD": {"name": "Gold ETF", "yf": "GLD", "alpaca": "GLD", "fallback_qty": 1},
    "SILVER": {"name": "Silver ETF", "yf": "SLV", "alpaca": "SLV", "fallback_qty": 1},
    "WTI": {"name": "WTI Crude Oil ETF", "yf": "USO", "alpaca": "USO", "fallback_qty": 1},
    "BTC": {"name": "Bitcoin", "yf": "BTC-USD", "alpaca": "BTCUSD", "fallback_qty": 0.001},
}

INTERVALS = {
    "1M": {"interval": "1m", "period": "2d"},
    "5M": {"interval": "5m", "period": "5d"},
    "15M": {"interval": "15m", "period": "10d"},
    "30M": {"interval": "30m", "period": "30d"},
}


def load_state():
    try:
        if STATE_FILE.exists():
            return json.loads(STATE_FILE.read_text())
    except Exception:
        pass
    return {"date": datetime.now().strftime("%Y-%m-%d"), "trades": [], "virtual_exits": []}


def save_state(state):
    try:
        STATE_FILE.write_text(json.dumps(state, indent=2))
    except Exception:
        pass


def reset_state_if_new_day(state):
    today = datetime.now().strftime("%Y-%m-%d")
    if state.get("date") != today:
        state["date"] = today
        state["trades"] = []
        state["virtual_exits"] = []
    return state


def today_trade_count():
    state = reset_state_if_new_day(load_state())
    save_state(state)
    return len(state.get("trades", []))


def record_trade(row):
    state = reset_state_if_new_day(load_state())
    state.setdefault("trades", []).append(row)
    save_state(state)


def record_virtual_exit(row):
    state = reset_state_if_new_day(load_state())
    exits = state.setdefault("virtual_exits", [])
    exits.append(row)
    state["virtual_exits"] = exits[-50:]
    save_state(state)


def safe_float(x, default=0.0):
    try:
        if x is None or pd.isna(x):
            return default
        return float(x)
    except Exception:
        return default


def load_candles(asset_key, tf):
    meta = ASSETS[asset_key]
    cfg = INTERVALS[tf]

    df = yf.download(
        meta["yf"],
        period=cfg["period"],
        interval=cfg["interval"],
        progress=False,
        auto_adjust=False,
        threads=False,
    )

    if df.empty:
        return pd.DataFrame()

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df.reset_index()
    df.columns = [str(c).lower().replace(" ", "_") for c in df.columns]

    time_col = "datetime" if "datetime" in df.columns else "date"
    df = df.rename(columns={time_col: "time"})

    df = df[["time", "open", "high", "low", "close", "volume"]].dropna()
    df = df[df["close"] > 0]

    return df.tail(500)


def add_indicators(df):
    d = df.copy()

    d["ema9"] = d["close"].ewm(span=9, adjust=False).mean()
    d["ema21"] = d["close"].ewm(span=21, adjust=False).mean()

    delta = d["close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(alpha=1 / 14, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / 14, adjust=False).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    d["rsi"] = (100 - (100 / (1 + rs))).fillna(50)

    tr1 = d["high"] - d["low"]
    tr2 = (d["high"] - d["close"].shift()).abs()
    tr3 = (d["low"] - d["close"].shift()).abs()

    d["tr"] = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    d["atr"] = d["tr"].ewm(alpha=1 / 14, adjust=False).mean()

    d["swing_low"] = d["low"].rolling(12).min()
    d["swing_high"] = d["high"].rolling(12).max()

    d["support"] = d["low"].rolling(50).min()
    d["resistance"] = d["high"].rolling(50).max()

    d["vol_ma"] = d["volume"].rolling(20).mean()

    return d


def generate_scalping_signal(df):
    d = add_indicators(df)

    if len(d) < 80:
        return {
            "signal": "HOLD",
            "confidence": 0,
            "score": 0,
            "price": 0,
            "atr": 0,
            "entry": 0,
            "take_profit": 0,
            "stop_loss": 0,
            "risk_per_unit": 0,
            "reason": "Not enough candle data",
        }

    last = d.iloc[-1]
    prev = d.iloc[-2]

    entry = safe_float(last.close)
    atr = max(safe_float(last.atr), entry * 0.001)

    uptrend = last.ema9 > last.ema21 and last.close > last.ema21
    downtrend = last.ema9 < last.ema21 and last.close < last.ema21

    pullback_long = last.low <= last.ema9 or last.low <= last.ema21
    pullback_short = last.high >= last.ema9 or last.high >= last.ema21

    rsi_cross_up = prev.rsi <= 50 and last.rsi > 50
    rsi_cross_down = prev.rsi >= 50 and last.rsi < 50

    volume_ok = last.volume >= safe_float(last.vol_ma, last.volume)

    swing_low = safe_float(last.swing_low, entry - atr)
    swing_high = safe_float(last.swing_high, entry + atr)
    support = safe_float(last.support, entry - atr * 2)
    resistance = safe_float(last.resistance, entry + atr * 2)

    signal = "HOLD"
    score = 0
    reason = "No valid EMA pullback + RSI confirmation"

    if uptrend:
        score += 25
    if downtrend:
        score -= 25
    if pullback_long:
        score += 18
    if pullback_short:
        score -= 18
    if rsi_cross_up:
        score += 28
    if rsi_cross_down:
        score -= 28
    if volume_ok and score > 0:
        score += 8
    elif volume_ok and score < 0:
        score -= 8

    if uptrend and pullback_long and rsi_cross_up:
        signal = "BUY"
        reason = "LONG: uptrend confirmed, pullback to EMA9/EMA21, RSI crossed above 50"

        atr_stop = entry - (1.5 * atr)
        swing_stop = swing_low - (0.10 * atr)
        stop_loss = max(atr_stop, swing_stop)

        risk = max(entry - stop_loss, atr * 0.50)
        rr_target = entry + (risk * RR_RATIO)

        if resistance > entry:
            take_profit = min(rr_target, resistance)
            if take_profit <= entry:
                take_profit = rr_target
        else:
            take_profit = rr_target

    elif downtrend and pullback_short and rsi_cross_down:
        signal = "SELL"
        reason = "SHORT: downtrend confirmed, rally to EMA9/EMA21, RSI crossed below 50"

        atr_stop = entry + (1.5 * atr)
        swing_stop = swing_high + (0.10 * atr)
        stop_loss = min(atr_stop, swing_stop)

        risk = max(stop_loss - entry, atr * 0.50)
        rr_target = entry - (risk * RR_RATIO)

        if support < entry:
            take_profit = max(rr_target, support)
            if take_profit >= entry:
                take_profit = rr_target
        else:
            take_profit = rr_target

    else:
        stop_loss = 0
        take_profit = 0
        risk = 0

    confidence = min(95, max(0, abs(score) * 1.20))

    if signal in ["BUY", "SELL"] and confidence < MIN_CONFIDENCE:
        signal = "HOLD"
        reason = "Setup detected but confidence below minimum threshold"

    if NEWS_BLACKOUT:
        signal = "HOLD"
        reason = "News blackout enabled. Trading blocked."

    return {
        "signal": signal,
        "confidence": round(confidence, 2),
        "score": round(score, 2),
        "price": round(entry, 4),
        "entry": round(entry, 4),
        "take_profit": round(take_profit, 4),
        "stop_loss": round(stop_loss, 4),
        "risk_per_unit": round(risk, 4),
        "atr": round(atr, 4),
        "rsi": round(float(last.rsi), 2),
        "ema9": round(float(last.ema9), 4),
        "ema21": round(float(last.ema21), 4),
        "support": round(support, 4),
        "resistance": round(resistance, 4),
        "reason": reason,
        "rules": {
            "uptrend": bool(uptrend),
            "downtrend": bool(downtrend),
            "pullback_long": bool(pullback_long),
            "pullback_short": bool(pullback_short),
            "rsi_cross_up": bool(rsi_cross_up),
            "rsi_cross_down": bool(rsi_cross_down),
            "volume_ok": bool(volume_ok),
            "news_blackout": NEWS_BLACKOUT,
        },
    }


def alpaca_headers():
    return {
        "APCA-API-KEY-ID": ALPACA_API_KEY,
        "APCA-API-SECRET-KEY": ALPACA_SECRET_KEY,
        "Content-Type": "application/json",
        "accept": "application/json",
    }


async def alpaca_get(path):
    try:
        async with httpx.AsyncClient(timeout=15) as client:
            r = await client.get(ALPACA_BASE_URL.rstrip("/") + path, headers=alpaca_headers())
        try:
            return r.status_code, r.json()
        except Exception:
            return r.status_code, {"raw": r.text}
    except Exception as e:
        return 500, {"error": str(e)}


async def alpaca_post(path, payload):
    try:
        async with httpx.AsyncClient(timeout=15) as client:
            r = await client.post(
                ALPACA_BASE_URL.rstrip("/") + path,
                headers=alpaca_headers(),
                json=payload,
            )
        try:
            return r.status_code, r.json()
        except Exception:
            return r.status_code, {"raw": r.text}
    except Exception as e:
        return 500, {"error": str(e)}


async def get_account_equity():
    code, data = await alpaca_get("/v2/account")
    if code == 200:
        return safe_float(data.get("equity"), 0)
    return 0


async def get_alpaca_quote(symbol):
    try:
        symbol_clean = symbol.replace("/", "").upper()

        if symbol_clean in ["BTCUSD", "ETHUSD"]:
            url = f"https://data.alpaca.markets/v1beta3/crypto/us/latest/trades?symbols={symbol_clean}"
            async with httpx.AsyncClient(timeout=15) as client:
                r = await client.get(url, headers=alpaca_headers())
            data = r.json()
            trade = data.get("trades", {}).get(symbol_clean)
            price = safe_float(trade.get("p"), 0) if trade else 0
            return {"price": price, "bid": price, "ask": price, "spread_pct": 0}

        url = f"https://data.alpaca.markets/v2/stocks/{symbol_clean}/quotes/latest"
        async with httpx.AsyncClient(timeout=15) as client:
            r = await client.get(url, headers=alpaca_headers())

        data = r.json()
        quote = data.get("quote", {})

        bid = safe_float(quote.get("bp"), 0)
        ask = safe_float(quote.get("ap"), 0)
        price = round((bid + ask) / 2, 4) if bid > 0 and ask > 0 else max(bid, ask)

        spread_pct = 0
        if price > 0 and bid > 0 and ask > 0:
            spread_pct = ((ask - bid) / price) * 100

        return {"price": price, "bid": bid, "ask": ask, "spread_pct": spread_pct}
    except Exception:
        return {"price": 0, "bid": 0, "ask": 0, "spread_pct": 999}


async def get_open_positions():
    code, data = await alpaca_get("/v2/positions")
    if code != 200:
        return []
    return data if isinstance(data, list) else []


async def trade_guard(asset_key, spread_pct):
    if not AUTO_TRADE_RUNTIME["enabled"]:
        return False, "Auto Trade OFF"

    if not ALPACA_API_KEY or not ALPACA_SECRET_KEY:
        return False, "Missing Alpaca API keys"

    if NEWS_BLACKOUT:
        return False, "News blackout active"

    if spread_pct > MAX_SPREAD_PCT:
        return False, f"Spread too wide: {round(spread_pct, 4)}%"

    if today_trade_count() >= MAX_TRADES_PER_DAY:
        return False, "Daily trade limit reached"

    positions = await get_open_positions()

    if len(positions) >= MAX_OPEN_POSITIONS:
        return False, "Maximum open positions reached"

    alpaca_symbol = ASSETS[asset_key]["alpaca"].replace("/", "").upper()

    for p in positions:
        pos_symbol = str(p.get("symbol", "")).replace("/", "").upper()
        if pos_symbol == alpaca_symbol:
            return False, "Position already open for this asset"

    return True, "OK"


async def calculate_qty(asset_key, entry, stop_loss):
    meta = ASSETS[asset_key]
    symbol = meta["alpaca"].replace("/", "").upper()

    equity = await get_account_equity()
    if equity <= 0:
        return meta["fallback_qty"], 0

    risk_capital = equity * RISK_PER_TRADE_PCT
    risk_per_unit = abs(entry - stop_loss)

    if risk_per_unit <= 0:
        return meta["fallback_qty"], risk_capital

    raw_qty = risk_capital / risk_per_unit

    if symbol in ["BTCUSD", "ETHUSD"]:
        qty = max(0.0001, round(raw_qty, 6))
    else:
        qty = max(1, int(raw_qty))

    return qty, round(risk_capital, 2)


async def close_position_market(symbol, qty, side):
    payload = {"symbol": symbol, "side": side, "type": "market", "qty": str(qty), "time_in_force": "gtc"}
    code, data = await alpaca_post("/v2/orders", payload)
    return {"status_code": code, "payload": payload, "response": data}


async def monitor_virtual_exits():
    state = reset_state_if_new_day(load_state())
    exits = state.get("virtual_exits", [])

    actions, remaining = [], []

    for item in exits:
        if item.get("status") != "open":
            continue

        quote = await get_alpaca_quote(item["symbol"])
        current = quote["price"]

        if current <= 0:
            remaining.append(item)
            continue

        side = item["side"]
        qty = item["qty"]
        target = item["take_profit"]
        stop = item["stop_loss"]

        hit_target = (side == "buy" and current >= target) or (side == "sell" and current <= target)
        hit_stop = (side == "buy" and current <= stop) or (side == "sell" and current >= stop)

        if hit_target or hit_stop:
            close_side = "sell" if side == "buy" else "buy"
            result = await close_position_market(item["symbol"], qty, close_side)

            item["status"] = "closed"
            item["closed_at"] = datetime.now(timezone.utc).isoformat()
            item["close_price"] = current
            item["close_reason"] = "TAKE_PROFIT" if hit_target else "STOP_LOSS"
            item["close_result"] = result

            actions.append(item)
        else:
            remaining.append(item)

    state["virtual_exits"] = remaining
    save_state(state)

    return actions


async def place_scalp_order(asset_key, side, signal):
    meta = ASSETS[asset_key]
    symbol = meta["alpaca"].replace("/", "").upper()
    is_crypto = symbol in ["BTCUSD", "ETHUSD"]

    quote = await get_alpaca_quote(symbol)
    entry = quote["price"] if quote["price"] > 0 else signal["entry"]

    stop_loss = signal["stop_loss"]
    take_profit = signal["take_profit"]

    if side == "buy":
        take_profit = max(take_profit, entry + 0.02)
        stop_loss = min(stop_loss, entry - 0.02)
    else:
        take_profit = min(take_profit, entry - 0.02)
        stop_loss = max(stop_loss, entry + 0.02)

    qty, risk_capital = await calculate_qty(asset_key, entry, stop_loss)

    payload = {
        "symbol": symbol,
        "side": side,
        "type": "market",
        "qty": str(qty),
        "time_in_force": "gtc" if is_crypto else "day",
    }

    if not is_crypto:
        payload["order_class"] = "bracket"
        payload["take_profit"] = {"limit_price": str(round(take_profit, 2))}
        payload["stop_loss"] = {"stop_price": str(round(stop_loss, 2))}

    code, data = await alpaca_post("/v2/orders", payload)

    if is_crypto and code in [200, 201]:
        record_virtual_exit({
            "status": "open",
            "time": datetime.now(timezone.utc).isoformat(),
            "asset": asset_key,
            "symbol": symbol,
            "side": side,
            "qty": qty,
            "entry": round(entry, 4),
            "take_profit": round(take_profit, 4),
            "stop_loss": round(stop_loss, 4),
        })

    result = {
        "status_code": code,
        "payload": payload,
        "response": data,
        "entry_price_used": round(entry, 4),
        "take_profit": round(take_profit, 4),
        "stop_loss": round(stop_loss, 4),
        "qty": qty,
        "risk_capital": risk_capital,
        "spread_pct": round(quote["spread_pct"], 4),
        "crypto_virtual_tp_sl": is_crypto,
    }

    record_trade({
        "time": datetime.now(timezone.utc).isoformat(),
        "asset": asset_key,
        "symbol": symbol,
        "side": side,
        **result,
    })

    return result


@app.get("/api/account")
async def api_account():
    code, data = await alpaca_get("/v2/account")
    return JSONResponse({"status_code": code, "data": data})


@app.get("/api/positions")
async def api_positions():
    code, data = await alpaca_get("/v2/positions")
    return JSONResponse({"status_code": code, "data": data})


@app.get("/api/orders")
async def api_orders():
    code, data = await alpaca_get("/v2/orders?status=all&limit=20")
    return JSONResponse({"status_code": code, "data": data})


@app.post("/toggle-auto-trade")
async def toggle_auto_trade(asset: str = Form("WTI"), tf: str = Form("1M")):
    AUTO_TRADE_RUNTIME["enabled"] = not AUTO_TRADE_RUNTIME["enabled"]
    return RedirectResponse(url=f"/?asset={asset}&tf={tf}&execute=true", status_code=303)


@app.post("/run-now")
async def run_now(asset: str = Form("WTI"), tf: str = Form("1M"), execute: str = Form("true")):
    return RedirectResponse(url=f"/?asset={asset}&tf={tf}&execute={execute}", status_code=303)


@app.get("/api/run-signal")
async def api_run_signal(asset: str = Query("GOLD"), tf: str = Query("1M"), execute: bool = Query(False)):
    return await run_signal_core(asset, tf, execute)


async def run_signal_core(asset: str, tf: str, execute: bool):
    asset = asset.upper()
    tf = tf.upper()

    virtual_exit_actions = await monitor_virtual_exits()

    if asset not in ASSETS:
        return {"error": "Invalid asset"}

    if tf not in INTERVALS:
        return {"error": "Invalid timeframe"}

    df = load_candles(asset, tf)

    if df.empty:
        return {"error": "No market data returned"}

    signal = generate_scalping_signal(df)

    symbol = ASSETS[asset]["alpaca"].replace("/", "").upper()
    quote = await get_alpaca_quote(symbol)

    allowed, guard_reason = await trade_guard(asset, quote["spread_pct"])

    trade_result = None

    should_execute = (
        execute
        and AUTO_TRADE_RUNTIME["enabled"]
        and allowed
        and signal["signal"] in ["BUY", "SELL"]
        and signal["confidence"] >= MIN_CONFIDENCE
    )

    if should_execute:
        side = "buy" if signal["signal"] == "BUY" else "sell"
        trade_result = await place_scalp_order(asset, side, signal)

    return {
        "asset": asset,
        "name": ASSETS[asset]["name"],
        "trade_symbol": ASSETS[asset]["alpaca"],
        "data_symbol": ASSETS[asset]["yf"],
        "timeframe": tf,
        "auto_trade_enabled": AUTO_TRADE_RUNTIME["enabled"],
        "auto_refresh_seconds": AUTO_REFRESH_SECONDS,
        "execute_requested": execute,
        "guard_allowed": allowed,
        "guard_reason": guard_reason,
        "daily_trade_count": today_trade_count(),
        "risk_per_trade_pct": RISK_PER_TRADE_PCT,
        "rr_ratio": RR_RATIO,
        "max_spread_pct": MAX_SPREAD_PCT,
        "quote": quote,
        "signal": signal,
        "scalping_possible": should_execute,
        "trade_result": trade_result,
        "virtual_exit_actions": virtual_exit_actions,
        "updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }


def html_options(current, values):
    return "".join(
        f'<option value="{v}" {"selected" if v == current else ""}>{v}</option>'
        for v in values
    )


@app.get("/", response_class=HTMLResponse)
async def home(asset: str = Query("WTI"), tf: str = Query("1M"), execute: bool = Query(True)):
    asset = asset.upper()
    tf = tf.upper()

    data = await run_signal_core(asset, tf, execute)

    signal_text = data.get("signal", {}).get("signal", "ERROR")
    signal_class = "buy" if signal_text == "BUY" else "sell" if signal_text == "SELL" else "hold"

    auto_class = "auto-on" if AUTO_TRADE_RUNTIME["enabled"] else "auto-off"
    auto_text = "AUTO TRADE ON" if AUTO_TRADE_RUNTIME["enabled"] else "AUTO TRADE OFF"

    return HTMLResponse(f"""
<!doctype html>
<html>
<head>
<title>Paper Scalping Bot By Abbas</title>
<meta http-equiv="refresh" content="{AUTO_REFRESH_SECONDS}; url=/?asset={asset}&tf={tf}&execute=true">
<style>
body{{background:#050812;color:white;font-family:Arial;margin:0;padding:28px 20px 20px 20px}}
#progress-container{{position:fixed;top:0;left:0;width:100%;height:8px;background:#111827;z-index:9999}}
#progress-bar{{width:100%;height:8px;background:#ff0000;animation:shrink {AUTO_REFRESH_SECONDS}s linear infinite}}
@keyframes shrink{{from{{width:100%}}to{{width:0%}}}}
.card{{background:#111827;border:1px solid #263244;border-radius:16px;padding:16px;margin-bottom:14px}}
button,select{{padding:10px;border-radius:10px;margin:5px;font-weight:bold}}
.buy{{color:#22c55e}}.sell{{color:#ef4444}}.hold{{color:#f59e0b}}
pre{{white-space:pre-wrap;background:#020617;padding:12px;border-radius:12px}}
.small{{color:#94a3b8;font-size:13px}}.name{{color:#ff4444;font-size:16px;margin-left:8px}}
.auto-on{{background:#16a34a;color:white;border:none}}.auto-off{{background:#dc2626;color:white;border:none}}
</style>
</head>
<body>
<div id="progress-container"><div id="progress-bar"></div></div>

<h2>Paper Scalping Bot — Alpaca Paper API <span class="name">By: Abbas</span></h2>
<div class="small">EMA 9/21 Pullback + RSI 50 Cross + ATR Stop + RR Target</div>

<div class="card">
<form method="post" action="/toggle-auto-trade" style="display:inline;">
<input type="hidden" name="asset" value="{asset}">
<input type="hidden" name="tf" value="{tf}">
<button class="{auto_class}" type="submit">{auto_text}</button>
</form>

<form method="post" action="/run-now" style="display:inline;">
<select name="asset">{html_options(asset, ASSETS.keys())}</select>
<select name="tf">{html_options(tf, INTERVALS.keys())}</select>
<button type="submit" name="execute" value="false">Check Signal Only</button>
<button type="submit" name="execute" value="true">Run + Place Paper Trade</button>
</form>

<form method="get" action="/api/account" style="display:inline;"><button type="submit">Account</button></form>
<form method="get" action="/api/positions" style="display:inline;"><button type="submit">Positions</button></form>
<form method="get" action="/api/orders" style="display:inline;"><button type="submit">Orders</button></form>
</div>

<div class="card">
<h3 class="{signal_class}">{signal_text}</h3>
<p>
Asset: {data.get("asset")}<br>
Trade Symbol: {data.get("trade_symbol")}<br>
Price: {data.get("signal", {}).get("price")}<br>
Entry: {data.get("signal", {}).get("entry")}<br>
Take Profit: {data.get("signal", {}).get("take_profit")}<br>
Stop Loss: {data.get("signal", {}).get("stop_loss")}<br>
Confidence: {data.get("signal", {}).get("confidence")}%<br>
Reason: {data.get("signal", {}).get("reason")}<br>
Guard: {data.get("guard_reason")}<br>
Auto Trade Enabled: {data.get("auto_trade_enabled")}<br>
Daily Trades: {data.get("daily_trade_count")}<br>
Updated: {data.get("updated")}
</p>
</div>

<div class="card">
<pre>{json.dumps(data, indent=2)}</pre>
</div>

</body>
</html>
""")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", "8000")),
        reload=True,
    )
