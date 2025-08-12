# Streamlit dashboard for Crypto Trader Thunderdome
# ------------------------------------------------
# Features:
# 1) Historical chart of bot wallet valuations using a snapshots file (JSON Lines)
# 2) Bot profile cards with bios and avatar images
# 3) A configurable link/button for users to request access to the private Discord server
#
# Minimal setup:
# - Place this file at the repo root (or in a /web folder) and run:  streamlit run streamlit_app.py
# - Create a snapshots file at data/snapshots.jsonl (one JSON per line; see README note)
# - Put bot avatar images in ./assets/avatars/ (filenames set below)
# - Optional: set env var DISCORD_JOIN_URL to control the join-request link
#
# Optional live valuation fallback:
# - If there's no snapshots file yet, we can still show current valuations if wallet JSONs exist,
#   provided COINGECKO_API_KEY is set. We'll hit the simple/price endpoint for spot USD quotes.

from __future__ import annotations
import os
import json
from datetime import datetime
from dataclasses import dataclass
from typing import Dict, List, Optional
from policy_helper_functions import load_policy
import streamlit as st
import pandas as pd
import plotly.express as px
import requests
from pathlib import Path

# -----------------------------
# Config
# -----------------------------
SNAPSHOTS_PATH = os.environ.get("SNAPSHOTS_PATH", "data/snapshots.jsonl")
WALLETS_DIR = os.environ.get("WALLETS_DIR", "wallets")
DISCORD_JOIN_URL = os.environ.get("DISCORD_JOIN_URL", "https://docs.google.com/forms/d/e/1FAIpQLSfZbORwXHKLODc1SBuETqtkpw4_CJK3tfT5q6tFrpPQCVgN9A/viewform?usp=header")
COINGECKO_API_KEY = os.environ.get("COINGECKO_API_KEY")

BOT_PROFILES = {
    # bot_key must match the names used in snapshots (e.g., "bitbot", "maxibit", "bearbot", "badbytebillie")
    "bitbot": {
        "display": "Jordan Bitbot",
        "bio": "Sarcastic classic wall street style trader who chases momentum but can explain the logic behind each move.",
        "avatar": "assets/avatars/bitbot.png",
    },
    "maxibit": {
        "display": "Maxibit",
        "bio": "Pretentious Bitcoin maxi with a soft spot for ice cream, prioritizing BTC/ETH unless alts outperform.",
        "avatar": "assets/avatars/maxibit.png",
    },
    "bearbot": {
        "display": "Bear Bot",
        "bio": "Risk‑averse grizzly who prefers stable positions, defensive entries, and holding cash when unsure.",
        "avatar": "assets/avatars/bearbot.png",
    },
    "badbytebillie": {
        "display": "Bad Byte Billie",
        "bio": "Deadpan, risk‑tolerant bot that rides trends with tight risk controls and snarky commentary.",
        "avatar": "assets/avatars/badbytebillie.png",
    },
}

# -----------------------------
# Helpers
# -----------------------------

def _read_snapshots(path: str) -> Optional[pd.DataFrame]:
    if not os.path.exists(path):
        return None
    rows: List[dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                # expected keys: timestamp (iso), bot, total_usd, usd_cash, holdings (dict)
                rows.append(obj)
            except json.JSONDecodeError:
                continue
    if not rows:
        return None
    df = pd.DataFrame(rows)
    # Normalize / parse types
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    # Some referees may emit "bot" or "name"; normalize
    if "name" in df.columns and "bot" not in df.columns:
        df = df.rename(columns={"name": "bot"})
    return df


def _read_wallets(dirpath: str) -> Dict[str, dict]:
    """Read each wallet JSON file (one per bot). Returns mapping {bot_name: wallet_obj}."""
    wallets: Dict[str, dict] = {}
    if not os.path.isdir(dirpath):
        return wallets
    for fname in os.listdir(dirpath):
        if not fname.endswith(".json"):
            continue
        try:
            with open(os.path.join(dirpath, fname), "r", encoding="utf-8") as f:
                data = json.load(f)
                # infer name from filename if not present
                bot = data.get("name") or os.path.splitext(fname)[0]
                wallets[bot.lower()] = data
        except Exception:
            continue
    return wallets


def _fetch_prices_usd(symbols: List[str]) -> Dict[str, float]:
    """Fetch USD prices for symbols using CoinGecko simple price. symbols are coin IDs, e.g., 'bitcoin', 'ethereum'."""
    if not COINGECKO_API_KEY or not symbols:
        return {}
    # Deduplicate and join
    ids = ",".join(sorted(set(symbols)))
    url = "https://pro-api.coingecko.com/api/v3/simple/price"
    params = {
        "ids": ids,
        "vs_currencies": "usd",
    }
    headers = {"x-cg-pro-api-key": COINGECKO_API_KEY}
    try:
        r = requests.get(url, params=params, headers=headers, timeout=10)
        r.raise_for_status()
        data = r.json()
        out: Dict[str, float] = {}
        for k, v in data.items():
            if isinstance(v, dict) and "usd" in v:
                out[k] = float(v["usd"])
        return out
    except Exception:
        return {}


def _estimate_current_from_wallets(wallets: Dict[str, dict]) -> Optional[pd.DataFrame]:
    """Build a one-timestamp dataframe of current totals using wallet balances + spot prices.
    Wallet structure assumption: {
        "name": "bitbot",
        "usd": 1234.56,
        "holdings": [{"id": "ethereum", "symbol": "eth", "amount": 0.5}, ...]
    }
    """
    if not wallets:
        return None
    # gather all CG IDs used
    coin_ids: List[str] = []
    for w in wallets.values():
        for h in w.get("holdings", []) or []:
            cid = h.get("id") or h.get("coingecko_id")
            if cid:
                coin_ids.append(str(cid).lower())
    prices = _fetch_prices_usd(coin_ids)
    if not prices and coin_ids:
        return None

    now = pd.Timestamp.utcnow()
    rows: List[dict] = []
    for bot_key, w in wallets.items():
        usd_cash = float(w.get("usd", 0.0))
        total = usd_cash
        for h in w.get("holdings", []) or []:
            cid = (h.get("id") or h.get("coingecko_id") or "").lower()
            amt = float(h.get("amount", 0.0))
            px = prices.get(cid, 0.0)
            total += amt * px
        rows.append({
            "timestamp": now,
            "bot": bot_key,
            "total_usd": total,
            "usd_cash": usd_cash,
        })
    return pd.DataFrame(rows)

def load_wallet(bot_name):
    wallet_path = Path(f"wallets/{bot_name}.json")
    return json.loads(wallet_path.read_text())

def get_portfolio_composition(bot_name):
    w = load_wallet(bot_name)
    balances = w.get("balances", {})
    usd_val = balances.get("USD", 0.0)
    
    # Remove USD from composition; we're focusing on crypto assets
    crypto_assets = {k: v for k, v in balances.items() if k != "USD" and v > 0}
    total_val = usd_val
    
    # mark to market if possible
    # Here you could pull spot prices if you want live values instead of qty
    return crypto_assets

# -----------------------------
# UI
# -----------------------------
st.set_page_config(page_title="Crypto Trader Thunderdome", page_icon="💥", layout="wide")

st.title("💥 Crypto Trader Thunderdome — Dashboard")
st.caption("Live(ish) paper‑trading tournament. Spot prices via CoinGecko Pro. Not financial advice.")

# ---- Chart section ----
st.subheader("Wallet valuations over time (USD)")

df_snap = _read_snapshots(SNAPSHOTS_PATH)
if df_snap is None:
    wallets = _read_wallets(WALLETS_DIR)
    df_snap = _estimate_current_from_wallets(wallets)
    if df_snap is not None:
        st.info("No snapshots found yet — showing a one‑point estimate from current wallets.")

if df_snap is None or df_snap.empty:
    st.warning("No data available. Ensure the referee is writing to data/snapshots.jsonl.")
else:
    # Basic smoothing / sampling controls
    left, right = st.columns([3, 1])
    with right:
        bots_selected = st.multiselect(
            "Bots",
            options=sorted(df_snap["bot"].str.lower().unique().tolist()),
            default=sorted(df_snap["bot"].str.lower().unique().tolist()),
        )
    if bots_selected:
        dff = df_snap[df_snap["bot"].str.lower().isin(bots_selected)].copy()
    else:
        dff = df_snap.copy()

    dff = dff.sort_values(["timestamp", "bot"])  # ensure stable order
    fig = px.line(
        dff,
        x="timestamp",
        y="total_usd",
        color="bot",
        markers=True,
        labels={"timestamp": "Time", "total_usd": "Total (USD)", "bot": "Bot"},
        title=None,
    )
    fig.update_traces(mode="lines+markers")
    fig.update_layout(height=420, legend_title_text="Bots", hovermode="x unified")
    st.plotly_chart(fig, use_container_width=True)

# ---- Profiles section ----
st.subheader("Meet the bots")

cols = st.columns(4)
for i, (key, meta) in enumerate(BOT_PROFILES.items()):
    with cols[i % 4]:
        avatar_path = meta.get("avatar", "")
        if os.path.exists(avatar_path):
            st.image(avatar_path, caption=meta.get("display", key), use_container_width=True)
        else:
            st.markdown(f"**{meta.get('display', key)}**")
        st.write(meta.get("bio", ""))

# ---- Bot risk policies ----
# They should change over time...

st.header("🤖 Bot Policies")
bot_names = ["bitbot", "maxibit", "bearbot", "badbytebillie"]  # adjust as needed

for bot in bot_names:
    try:
        policy = load_policy(bot)
        st.subheader(f"{bot.title()} Policy")
        st.json(policy.dict() if hasattr(policy, "dict") else policy)
    except FileNotFoundError:
        st.warning(f"No saved policy found for {bot}.")

# ---- Bot current holdings ----
# ---- Bot current holdings ----
st.header("📊 Portfolio Composition")

for bot in bot_names:
    assets = get_portfolio_composition(bot)  # dict like {"ETH": 0.42, "LINK": 12.3}
    if assets:
        df = pd.DataFrame({
            "Asset": list(assets.keys()),
            "Amount": list(assets.values())
        })
        # Optional: show percentages in hover
        df["Percent"] = df["Amount"] / df["Amount"].sum() * 100.0

        fig = px.pie(
            df,
            names="Asset",
            values="Amount",
            title=f"{bot.title()} — Portfolio Composition",
            hover_data={"Percent": ":.2f"}
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info(f"{bot.title()} has no crypto holdings yet.")


# ---- Join link ----
st.subheader("Request access to the private Discord server")
st.write("Want to watch the bots live, ask questions, or propose challenges?")
st.link_button("Request to join", DISCORD_JOIN_URL)

st.divider()
st.caption("© 2025 Crypto Trader Thunderdome. Data powered by CoinGecko Pro.")
