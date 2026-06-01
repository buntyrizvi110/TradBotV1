import os
import json
from datetime import datetime, timezone
from pathlib import Path

import httpx
import numpy as np
import pandas as pd

from fastapi import FastAPI, Query, Form
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse


CAPITAL_API_KEY = os.getenv("CAPITAL_API_KEY", "")
CAPITAL_IDENTIFIER = os.getenv("CAPITAL_IDENTIFIER", "")
CAPITAL_PASSWORD = os.getenv("CAPITAL_PASSWORD", "")

CAPITAL_BASE_URL = os.getenv(
    "CAPITAL_BASE_URL",
    "https://demo-api-capital.backend-capital.com/api/v1"
)

AUTO_TRADE = os.getenv("AUTO_TRADE", "true").lower() == "true"
AUTO_TRADE_RUNTIME = {"enabled": AUTO_TRADE}

AUTO_REFRESH_SECONDS = int(os.getenv("AUTO_REFRESH_SECONDS", "60"))
MAX_TRADES_PER_DAY = int(os.getenv("MAX_TRADES_PER_DAY", "100"))
MIN_CONFIDENCE = float(os.getenv("MIN_CONFIDENCE", "60"))

RISK_PER_TRADE_PCT = float(os.getenv("RISK_PER_TRADE_PCT", "0.005"))
RR_RATIO = float(os.getenv("RR_RATIO", "1.5"))
MAX_SPREAD_PCT = float(os.getenv("MAX_SPREAD_PCT", "100.00"))
NEWS_BLACKOUT = os.getenv("NEWS_BLACKOUT", "false").lower() == "true"

STATE_FILE = Path(os.getenv("STATE_FILE", "trade_state.json"))

app = FastAPI(title="Capital.com Scalping Bot By Abbas")

CAPITAL_SESSION = {
    "cst": "",
    "x_security_token": "",
    "last_login": None,
}

ASSETS = {
    "GOLD": {"name": "Gold Spot CFD", "capital_epic": "GOLD", "fallback_qty": 1},
    "SILVER": {"name": "Silver Spot CFD", "capital_epic": "SILVER", "fallback_qty": 1},
    "WTI": {"name": "WTI Crude Oil CFD", "capital_epic": "OIL_CRUDE", "fallback_qty": 1},
    "BRENT": {"name": "Brent Crude Oil CFD", "capital_epic": "OIL_BRENT", "fallback_qty": 1},
    "BTC": {"name": "Bitcoin CFD", "capital_epic": "BTCUSD", "fallback_qty": 0.01},
    "ETH": {"name": "Ethereum CFD", "capital_epic": "ETHUSD", "fallback_qty": 0.1},
    "USTECH100": {"name": "US Tech 100 CFD", "capital_epic": "US100", "fallback_qty": 1},
}

INTERVALS = {
    "1M": "MINUTE",
    "5M": "MINUTE_5",
    "15M": "MINUTE_15",
    "30M": "MINUTE_30",
}


def safe_float(x, default=0.0):
    try:
        if x is None or pd.isna(x):
            return default
        return float(x)
    except Exception:
        return default


def load_state():
    try:
        if STATE_FILE.exists():
            return json.loads(STATE_FILE.read_text())
    except Exception:
        pass
    return {"date": datetime.now().strftime("%Y-%m-%d"), "trades": []}


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
    return state


def today_trade_count():
    state = reset_state_if_new_day(load_state())
    save_state(state)
    return len(state.get("trades", []))


def record_trade(row):
    state = reset_state_if_new_day(load_state())
    state.setdefault("trades", []).append(row)
    save_state(state)


def capital_login_headers():
    return {
        "X-CAP-API-KEY": CAPITAL_API_KEY.strip(),
        "Content-Type": "application/json",
        "Accept": "application/json",
    }


def capital_auth_headers():
    return {
        "X-CAP-API-KEY": CAPITAL_API_KEY.strip(),
        "CST": CAPITAL_SESSION["cst"],
        "X-SECURITY-TOKEN": CAPITAL_SESSION["x_security_token"],
        "Content-Type": "application/json",
        "Accept": "application/json",
    }


async def capital_login():
    if not CAPITAL_API_KEY or not CAPITAL_IDENTIFIER or not CAPITAL_PASSWORD:
        return {"ok": False, "error": "Missing Capital.com API environment variables"}

    payload = {
        "identifier": CAPITAL_IDENTIFIER.strip(),
        "password": CAPITAL_PASSWORD.strip(),
        "encryptedPassword": False,
    }

    try:
        async with httpx.AsyncClient(timeout=20) as client:
            r = await client.post(
                f"{CAPITAL_BASE_URL.rstrip('/')}/session",
                headers=capital_login_headers(),
                json=payload,
            )

        cst = r.headers.get("CST", "")
        xsec = r.headers.get("X-SECURITY-TOKEN", "")

        try:
            body = r.json()
        except Exception:
            body = {"raw": r.text}

        if r.status_code in [200, 201] and cst and xsec:
            CAPITAL_SESSION["cst"] = cst
            CAPITAL_SESSION["x_security_token"] = xsec
            CAPITAL_SESSION["last_login"] = datetime.now(timezone.utc).isoformat()
            return {"ok": True, "status_code": r.status_code, "message": "Capital.com login OK"}

        return {"ok": False, "status_code": r.status_code, "error": body}

    except Exception as e:
        return {"ok": False, "error": str(e)}


async def ensure_capital_session():
    if CAPITAL_SESSION["cst"] and CAPITAL_SESSION["x_security_token"]:
        return {"ok": True}
    return await capital_login()


async def capital_get(path):
    session = await ensure_capital_session()
    if not session.get("ok"):
        return 401, session

    async with httpx.AsyncClient(timeout=20) as client:
        r = await client.get(
            CAPITAL_BASE_URL.rstrip("/") + path,
            headers=capital_auth_headers(),
        )

    if r.status_code == 401:
        await capital_login()
        async with httpx.AsyncClient(timeout=20) as client:
            r = await client.get(
                CAPITAL_BASE_URL.rstrip("/") + path,
                headers=capital_auth_headers(),
            )

    try:
        return r.status_code, r.json()
    except Exception:
        return r.status_code, {"raw": r.text}


async def capital_post(path, payload):
    session = await ensure_capital_session()
    if not session.get("ok"):
        return 401, session

    async with httpx.AsyncClient(timeout=20) as client:
        r = await client.post(
            CAPITAL_BASE_URL.rstrip("/") + path,
            headers=capital_auth_headers(),
            json=payload,
        )

    if r.status_code == 401:
        await capital_login()
        async with httpx.AsyncClient(timeout=20) as client:
            r = await client.post(
                CAPITAL_BASE_URL.rstrip("/") + path,
                headers=capital_auth_headers(),
                json=payload,
            )

    try:
        return r.status_code, r.json()
    except Exception:
        return r.status_code, {"raw": r.text}


async def load_capital_candles(asset_key, tf):
    epic = ASSETS[asset_key]["capital_epic"]
    resolution = INTERVALS.get(tf, "MINUTE")

    code, data = await capital_get(f"/prices/{epic}?resolution={resolution}&max=300")

    if code != 200:
        print("Capital candle error:", code, data)
        return pd.DataFrame()

    rows = []

    for p in data.get("prices", []):
        try:
            op = p.get("openPrice", {})
            hp = p.get("highPrice", {})
            lp = p.get("lowPrice", {})
            cp = p.get("closePrice", {})

            rows.append({
                "time": p.get("snapshotTimeUTC") or p.get("snapshotTime"),
                "open": safe_float(op.get("bid") or op.get("ask") or op.get("lastTraded")),
                "high": safe_float(hp.get("bid") or hp.get("ask") or hp.get("lastTraded")),
                "low": safe_float(lp.get("bid") or lp.get("ask") or lp.get("lastTraded")),
                "close": safe_float(cp.get("bid") or cp.get("ask") or cp.get("lastTraded")),
                "volume": safe_float(p.get("lastTradedVolume"), 0),
            })
        except Exception:
            continue

    df = pd.DataFrame(rows)

    if df.empty:
        return df

    df["time"] = pd.to_datetime(df["time"], errors="coerce")
    df = df.dropna()
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

    d["support"] = d["low"].rolling(50).min()
    d["resistance"] = d["high"].rolling(50).max()
    d["vol_ma"] = d["volume"].rolling(20).mean()

    return d


def generate_scalping_signal(df):
    d = add_indicators(df)

    if len(d) < 30:
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
            "reason": "Not enough Capital.com candle data",
            "engine": "V4_STRONG_CAPITAL_SIGNAL",
        }

    last = d.iloc[-1]
    prev = d.iloc[-2]
    prev2 = d.iloc[-3]

    entry = safe_float(last.close)
    atr = max(safe_float(last.atr), entry * 0.0008)

    one_candle_move = ((last.close - prev.close) / max(prev.close, 0.0001)) * 100
    three_candle_move = ((last.close - prev2.close) / max(prev2.close, 0.0001)) * 100

    score = 0
    reasons = []

    if last.close > prev.close:
        score += 12
        reasons.append("last candle bullish")
    elif last.close < prev.close:
        score -= 12
        reasons.append("last candle bearish")

    if three_candle_move > 0:
        score += 12
        reasons.append("3-candle move up")
    elif three_candle_move < 0:
        score -= 12
        reasons.append("3-candle move down")

    if three_candle_move >= 0.03:
        score += 12
        reasons.append("bullish momentum")
    elif three_candle_move <= -0.03:
        score -= 12
        reasons.append("bearish momentum")

    if last.ema9 > last.ema21:
        score += 12
        reasons.append("EMA bullish")
    elif last.ema9 < last.ema21:
        score -= 12
        reasons.append("EMA bearish")

    if last.ema9 > prev.ema9:
        score += 10
        reasons.append("EMA slope up")
    elif last.ema9 < prev.ema9:
        score -= 10
        reasons.append("EMA slope down")

    candle_range = max(last.high - last.low, entry * 0.0001)
    close_position = (last.close - last.low) / candle_range

    if close_position >= 0.60:
        score += 10
        reasons.append("close near candle high")
    elif close_position <= 0.40:
        score -= 10
        reasons.append("close near candle low")

    rsi = safe_float(last.rsi, 50)
    prev_rsi = safe_float(prev.rsi, 50)

    if rsi >= prev_rsi and rsi >= 52:
        score += 8
        reasons.append("RSI strong bullish")
    elif rsi <= prev_rsi and rsi <= 48:
        score -= 8
        reasons.append("RSI strong bearish")

    recent_high = d["high"].tail(6).iloc[:-1].max()
    recent_low = d["low"].tail(6).iloc[:-1].min()

    if last.close > recent_high:
        score += 10
        reasons.append("micro breakout up")
    elif last.close < recent_low:
        score -= 10
        reasons.append("micro breakout down")

    vol_ma = safe_float(last.vol_ma, last.volume)

    if vol_ma > 0 and last.volume >= vol_ma * 0.50:
        if score > 0:
            score += 6
            reasons.append("volume supports buy")
        elif score < 0:
            score -= 6
            reasons.append("volume supports sell")

    signal = "HOLD"
    stop_loss = 0
    take_profit = 0
    risk = 0

    RAW_TRIGGER = 35

    trend_buy = (
        last.close > last.ema9
        and last.ema9 > last.ema21
        and last.ema9 > prev.ema9
        and rsi >= 52
    )

    trend_sell = (
        last.close < last.ema9
        and last.ema9 < last.ema21
        and last.ema9 < prev.ema9
        and rsi <= 48
    )

    momentum_buy = (
        three_candle_move >= 0.03
        and close_position >= 0.60
    )

    momentum_sell = (
        three_candle_move <= -0.03
        and close_position <= 0.40
    )

    if score >= RAW_TRIGGER and trend_buy and momentum_buy:
        signal = "BUY"
        stop_loss = entry - atr
        risk = entry - stop_loss
        take_profit = entry + risk * RR_RATIO

    elif score <= -RAW_TRIGGER and trend_sell and momentum_sell:
        signal = "SELL"
        stop_loss = entry + atr
        risk = stop_loss - entry
        take_profit = entry - risk * RR_RATIO

    if signal in ["BUY", "SELL"]:
        confidence = min(95, max(60, abs(score) * 1.8))
    else:
        confidence = 0

    if NEWS_BLACKOUT:
        signal = "HOLD"
        confidence = 0
        reasons.append("News blackout enabled")

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
        "rsi": round(rsi, 2),
        "ema9": round(float(last.ema9), 4),
        "ema21": round(float(last.ema21), 4),
        "support": round(safe_float(last.support, entry - atr * 2), 4),
        "resistance": round(safe_float(last.resistance, entry + atr * 2), 4),
        "reason": ", ".join(reasons) if reasons else "No strong setup",
        "engine": "V4_STRONG_CAPITAL_SIGNAL",
        "rules": {
            "one_candle_move_pct": round(one_candle_move, 4),
            "three_candle_move_pct": round(three_candle_move, 4),
            "score": round(score, 2),
            "trend_buy": bool(trend_buy),
            "trend_sell": bool(trend_sell),
            "momentum_buy": bool(momentum_buy),
            "momentum_sell": bool(momentum_sell),
            "news_blackout": NEWS_BLACKOUT,
        },
    }


async def get_capital_account_equity():
    code, data = await capital_get("/accounts")
    if code != 200:
        return 0

    accounts = data.get("accounts", [])
    if not accounts:
        return 0

    balance = accounts[0].get("balance", {})
    return (
        safe_float(balance.get("balance"), 0)
        or safe_float(balance.get("available"), 0)
        or safe_float(balance.get("deposit"), 0)
    )


async def get_capital_market(asset_key):
    epic = ASSETS[asset_key]["capital_epic"]
    return await capital_get(f"/markets/{epic}")


async def get_capital_quote(asset_key):
    code, data = await get_capital_market(asset_key)

    if code != 200:
        return {
            "price": 0,
            "bid": 0,
            "ask": 0,
            "spread_pct": 999,
            "source": "capital_market_failed",
            "status_code": code,
            "raw": data,
        }

    snapshot = data.get("snapshot", {})

    bid = safe_float(snapshot.get("bid") or snapshot.get("bidPrice") or snapshot.get("sell"), 0)
    ask = safe_float(snapshot.get("offer") or snapshot.get("ask") or snapshot.get("offerPrice") or snapshot.get("buy"), 0)

    if bid > 0 and ask > 0:
        price = round((bid + ask) / 2, 4)
        spread_pct = ((ask - bid) / price) * 100
        return {
            "price": price,
            "bid": bid,
            "ask": ask,
            "spread_pct": round(spread_pct, 4),
            "source": "capital_market_quote",
            "epic": ASSETS[asset_key]["capital_epic"],
        }

    return {
        "price": 0,
        "bid": bid,
        "ask": ask,
        "spread_pct": 999,
        "source": "capital_quote_no_bid_ask",
        "raw": data,
    }


async def trade_guard(asset_key, spread_pct, execute):
    if not execute:
        return False, "Execution disabled"

    if not AUTO_TRADE_RUNTIME["enabled"]:
        return False, "Auto Trade OFF"

    if not CAPITAL_API_KEY or not CAPITAL_IDENTIFIER or not CAPITAL_PASSWORD:
        return False, "Missing Capital.com API keys"

    if NEWS_BLACKOUT:
        return False, "News blackout active"

    if today_trade_count() >= MAX_TRADES_PER_DAY:
        return False, f"Daily trade limit reached: {MAX_TRADES_PER_DAY}"

    if spread_pct >= 999:
        return True, "OK - quote unavailable, allowed for demo"

    if spread_pct > MAX_SPREAD_PCT:
        return True, f"OK - spread ignored for demo trade: {round(spread_pct, 4)}%"

    return True, "OK - auto order allowed"


async def calculate_qty(asset_key, entry, stop_loss):
    meta = ASSETS[asset_key]
    equity = await get_capital_account_equity()

    if equity <= 0:
        return meta["fallback_qty"], 0

    risk_capital = equity * RISK_PER_TRADE_PCT
    risk_per_unit = abs(entry - stop_loss)

    if risk_per_unit <= 0:
        return meta["fallback_qty"], risk_capital

    raw_qty = risk_capital / risk_per_unit

    if asset_key in ["BTC", "ETH"]:
        qty = max(meta["fallback_qty"], round(raw_qty, 4))
    else:
        qty = max(meta["fallback_qty"], round(raw_qty, 2))

    return qty, round(risk_capital, 2)


async def place_capital_order(asset_key, side, signal):
    quote = await get_capital_quote(asset_key)

    entry = quote["price"] if quote["price"] > 0 else signal["entry"]
    spread_buffer = max(entry * 0.0003, 0.03)

    atr = safe_float(signal.get("atr"), entry * 0.001)
    risk_distance = max(atr, spread_buffer * 2)

    side = side.lower()

    if side == "buy":
        stop_loss = entry - risk_distance
        take_profit = entry + risk_distance * RR_RATIO
        direction = "BUY"
    else:
        stop_loss = entry + risk_distance
        take_profit = entry - risk_distance * RR_RATIO
        direction = "SELL"

    qty, risk_capital = await calculate_qty(asset_key, entry, stop_loss)

    payload = {
        "epic": ASSETS[asset_key]["capital_epic"],
        "direction": direction,
        "size": qty,
        "orderType": "MARKET",
        "currencyCode": "USD",
        "forceOpen": True,
        "guaranteedStop": False,
        "stopLevel": round(stop_loss, 4),
        "profitLevel": round(take_profit, 4),
    }

    code, data = await capital_post("/positions", payload)

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
    }

    record_trade({
        "time": datetime.now(timezone.utc).isoformat(),
        "asset": asset_key,
        "epic": ASSETS[asset_key]["capital_epic"],
        "side": side,
        **result,
    })

    return result


@app.get("/api/login")
async def api_login():
    return JSONResponse(await capital_login())


@app.get("/api/account")
async def api_account():
    code, data = await capital_get("/accounts")
    return JSONResponse({"status_code": code, "data": data})


@app.get("/api/positions")
async def api_positions():
    code, data = await capital_get("/positions")
    return JSONResponse({"status_code": code, "data": data})


@app.get("/api/orders")
async def api_orders():
    code, data = await capital_get("/workingorders")
    return JSONResponse({"status_code": code, "data": data})


@app.post("/toggle-auto-trade")
async def toggle_auto_trade(asset: str = Form("WTI"), tf: str = Form("1M")):
    AUTO_TRADE_RUNTIME["enabled"] = not AUTO_TRADE_RUNTIME["enabled"]
    return RedirectResponse(url=f"/?asset={asset}&tf={tf}&execute=true", status_code=303)


@app.post("/run-now")
async def run_now(asset: str = Form("WTI"), tf: str = Form("1M"), execute: str = Form("true")):
    return RedirectResponse(url=f"/?asset={asset}&tf={tf}&execute={execute}", status_code=303)


@app.get("/api/run-signal")
async def api_run_signal(asset: str = Query("WTI"), tf: str = Query("1M"), execute: bool = Query(False)):
    return await run_signal_core(asset, tf, execute)


async def run_signal_core(asset: str, tf: str, execute: bool):
    asset = asset.upper()
    tf = tf.upper()

    if asset not in ASSETS:
        return {"error": "Invalid asset"}

    if tf not in INTERVALS:
        return {"error": "Invalid timeframe"}

    df = await load_capital_candles(asset, tf)

    if df.empty:
        return {"error": "No Capital.com market data returned"}

    signal = generate_scalping_signal(df)
    quote = await get_capital_quote(asset)

    allowed, guard_reason = await trade_guard(asset, quote["spread_pct"], execute)

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
        trade_result = await place_capital_order(asset, side, signal)

    return {
        "asset": asset,
        "name": ASSETS[asset]["name"],
        "trade_symbol": ASSETS[asset]["capital_epic"],
        "data_symbol": ASSETS[asset]["capital_epic"],
        "data_source": "Capital.com candles",
        "timeframe": tf,
        "broker": "Capital.com",
        "auto_trade_enabled": AUTO_TRADE_RUNTIME["enabled"],
        "auto_refresh_seconds": AUTO_REFRESH_SECONDS,
        "execute_requested": execute,
        "guard_allowed": allowed,
        "guard_reason": guard_reason,
        "daily_trade_count": today_trade_count(),
        "max_trades_per_day": MAX_TRADES_PER_DAY,
        "risk_per_trade_pct": RISK_PER_TRADE_PCT,
        "rr_ratio": RR_RATIO,
        "max_spread_pct": MAX_SPREAD_PCT,
        "quote": quote,
        "signal": signal,
        "scalping_possible": should_execute,
        "trade_result": trade_result,
        "capital_session_active": bool(CAPITAL_SESSION["cst"] and CAPITAL_SESSION["x_security_token"]),
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
<title>Capital.com Scalping Bot By Abbas</title>
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

<h2>Capital.com Strong Signal Scalping Bot <span class="name">By: Abbas</span></h2>
<div class="small">V4 Strong Signal: Capital.com candles + Capital.com execution</div>

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
<button type="submit" name="execute" value="true">Run + Place Capital Trade</button>
</form>

<form method="get" action="/api/login" style="display:inline;"><button type="submit">Login</button></form>
<form method="get" action="/api/account" style="display:inline;"><button type="submit">Account</button></form>
<form method="get" action="/api/positions" style="display:inline;"><button type="submit">Positions</button></form>
<form method="get" action="/api/orders" style="display:inline;"><button type="submit">Orders</button></form>
</div>

<div class="card">
<h3 class="{signal_class}">{signal_text}</h3>
<p>
Asset: {data.get("asset")}<br>
Broker: Capital.com<br>
Trade Symbol / EPIC: {data.get("trade_symbol")}<br>
Data Source: {data.get("data_source")}<br>
Price: {data.get("signal", {}).get("price")}<br>
Entry: {data.get("signal", {}).get("entry")}<br>
Take Profit: {data.get("signal", {}).get("take_profit")}<br>
Stop Loss: {data.get("signal", {}).get("stop_loss")}<br>
Confidence: {data.get("signal", {}).get("confidence")}%<br>
Engine: {data.get("signal", {}).get("engine")}<br>
Reason: {data.get("signal", {}).get("reason")}<br>
Guard: {data.get("guard_reason")}<br>
Auto Trade Enabled: {data.get("auto_trade_enabled")}<br>
Daily Trades: {data.get("daily_trade_count")} / {data.get("max_trades_per_day")}<br>
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
        reload=False,
    )
