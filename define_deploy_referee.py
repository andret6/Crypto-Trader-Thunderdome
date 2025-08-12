import os
import json
import time
import math
import glob
import asyncio
import requests
import discord
import subprocess
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
SNAP_PATH = Path("data") / "snapshots.jsonl"
SNAP_PATH.parent.mkdir(parents=True, exist_ok=True)


# ========== Config ==========
REF_TOKEN   = os.getenv("DISCORD_BOT_TOKEN_5")
CG_KEY      = os.getenv("COINGECKO_API_KEY")  # CoinGecko Pro key
WALLET_DIR  = os.getenv("WALLET_DIR", os.path.join(os.getcwd(), "wallets"))
POST_EVERY  = int(os.getenv("REFEREE_INTERVAL_SECS", "900"))  # 900s = 15 min

if not REF_TOKEN:
    raise RuntimeError("DISCORD_BOT_TOKEN_5 env var is required")
if not CG_KEY:
    raise RuntimeError("COINGECKO_API_KEY env var is required for pricing")

CG_BASE = "https://pro-api.coingecko.com/api/v3"

def _cg_headers():
    return {"x-cg-pro-api-key": CG_KEY}

# Common canonical mappings (extend if needed)
CANONICAL_IDS = {
    "BTC": "bitcoin",
    "XBT": "bitcoin",
    "ETH": "ethereum",
    "SOL": "solana",
    "ADA": "cardano",
    "DOGE": "dogecoin",
    "BNB": "binancecoin",
    "XRP": "ripple",
}

# Simple in-memory cache for id resolution
_COINS_CACHE = {}  # symbol/name upper -> id
_COINS_CACHE_TIME = 0.0
_COINS_CACHE_TTL = 60 * 60

def _refresh_coins_cache():
    global _COINS_CACHE, _COINS_CACHE_TIME
    if _COINS_CACHE and (time.time() - _COINS_CACHE_TIME) < _COINS_CACHE_TTL:
        return
    r = requests.get(f"{CG_BASE}/coins/list", headers=_cg_headers(), timeout=30)
    r.raise_for_status()
    _COINS_CACHE = {}
    for row in r.json():
        cid = row.get("id")
        sym = (row.get("symbol") or "").upper()
        name = (row.get("name") or "").upper()
        if not cid:
            continue
        if sym:
            _COINS_CACHE.setdefault(sym, cid)
            if sym == "BTC":
                _COINS_CACHE.setdefault("XBT", cid)
        if name:
            _COINS_CACHE.setdefault(name, cid)
    _COINS_CACHE_TIME = time.time()

def _search_id(query: str):
    # fallback search for odd tickers
    try:
        r = requests.get(f"{CG_BASE}/search", headers=_cg_headers(), params={"query": query}, timeout=15)
        r.raise_for_status()
        coins = r.json().get("coins", [])
        if not coins:
            return None
        # prefer exact symbol match
        up = query.upper()
        for c in coins:
            if (c.get("symbol") or "").upper() == up:
                return c.get("id")
        return coins[0].get("id")
    except Exception:
        return None

def resolve_id(symbol: str):
    sym = symbol.upper().strip()
    if sym in CANONICAL_IDS:
        return CANONICAL_IDS[sym]
    _refresh_coins_cache()
    if sym in _COINS_CACHE:
        return _COINS_CACHE[sym]
    cid = _search_id(sym)
    if cid:
        _COINS_CACHE[sym] = cid
    return cid

def pretty_money(x: float) -> str:
    if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
        return "N/A"
    if x >= 1000:
        return f"${x:,.0f}"
    if x >= 1:
        return f"${x:,.2f}"
    return f"${x:.6f}"

def load_wallets():
    """
    Return dict: {bot_name: {'USD':float, 'SYM1':qty, ...}} from wallet JSON files.
    Supports two schemas:

    A) balances dict:
       {"USD": 5000.0, "ETH": 0.5, "LINK": 100}

    B) holdings list + usd:
       {"name":"bitbot","usd":5000.0,"holdings":[{"id":"ethereum","symbol":"eth","amount":0.5}, ...]}
    """
    wallets = {}
    pattern = os.path.join(WALLET_DIR, "*.json")
    for path in glob.glob(pattern):
        bot = os.path.splitext(os.path.basename(path))[0]  # e.g., 'bitbot'
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)

            # Schema A: direct balances dict
            balances = data.get("balances")
            if isinstance(balances, dict):
                # normalize keys to upper
                norm = {k.upper(): float(v or 0.0) for k, v in balances.items()}
                wallets[bot] = norm
                continue

            # Schema B: usd + holdings list
            usd_cash = float(data.get("usd", data.get("USD", 0.0)) or 0.0)
            holdings = data.get("holdings") or []
            bal = {"USD": usd_cash}
            for h in holdings:
                sym = (h.get("symbol") or h.get("SYM") or "").upper()
                amt = float(h.get("amount", 0.0) or 0.0)
                if sym and amt > 0:
                    bal[sym] = bal.get(sym, 0.0) + amt
            wallets[bot] = bal

        except Exception as e:
            print(f"⚠️ Failed to read wallet {path}: {e}")
    return wallets

def fetch_prices(ids: list[str], vs_currency="usd"):
    """Return dict: id -> price"""
    if not ids:
        return {}
    # Use coins/markets (batch) for spot prices
    params = {
        "vs_currency": vs_currency,
        "ids": ",".join(ids),
        "order": "market_cap_desc",
        "per_page": len(ids),
        "page": 1
    }
    r = requests.get(f"{CG_BASE}/coins/markets", headers=_cg_headers(), params=params, timeout=30)
    r.raise_for_status()
    out = {}
    for row in r.json():
        out[row.get("id")] = row.get("current_price")
    return out

async def compute_leaderboard():
    wallets = load_wallets()
    # Gather symbols to price
    symbols = set()
    for bal in wallets.values():
        for k, v in bal.items():
            if k.upper() == "USD":
                continue
            if isinstance(v, (int, float)) and v > 0:
                symbols.add(k.upper())

    # Resolve ids
    id_by_sym = {}
    for sym in symbols:
        cid = resolve_id(sym)
        if cid:
            id_by_sym[sym] = cid

    prices = fetch_prices(list(set(id_by_sym.values()))) if id_by_sym else {}

    # Compute values
    rows = []
    for bot, bal in wallets.items():
        usd = float(bal.get("USD", 0.0) or 0.0)
        detail = []
        total = usd
        for sym, qty in bal.items():
            if sym.upper() == "USD":
                continue
            try:
                q = float(qty or 0.0)
            except Exception:
                q = 0.0
            if q <= 0:
                continue
            cid = id_by_sym.get(sym.upper())
            px  = prices.get(cid, 0.0)
            val = q * (px or 0.0)
            total += val
            detail.append((sym.upper(), q, px, val))
        rows.append((bot, total, usd, detail))

    # Rank
    rows.sort(key=lambda r: r[1], reverse=True)
    return rows

def format_report(rows):
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    if not rows:
        return f"**The Referee Report — {ts}**\nNo wallets found."

    lines = [f"**The Referee Report — {ts}**"]
    winner = rows[0][0]
    loser  = rows[-1][0]
    lines.append(f"🏆 Highest total: **{winner}** | 🫠 Lowest total: **{loser}**")

    for bot, total, usd, detail in rows:
        lines.append(f"\n**{bot}** — Total: {pretty_money(total)}  \nUSD - Cash: {pretty_money(usd)}")
        if detail:
            for sym, q, px, val in detail:
                lines.append(f"• {sym}: {q:.6f} × {pretty_money(px)} = {pretty_money(val)}")
        else:
            lines.append("• No crypto positions")

    return "\n".join(lines)
  
# function to push updates to the repo, to keep the streamlit app updated
def git_push_updates():
    try:
        # Stage changes
        subprocess.run(["git", "add", "data/snapshots.jsonl", "policies/"], check=True)
        
        # Check if anything is staged (git diff-index returns non-zero if staged changes)
        diff = subprocess.run(["git", "diff", "--cached", "--quiet"])
        if diff.returncode == 0:
            print("[referee] No changes to commit.")
            return
        
        # Commit
        ts = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
        subprocess.run(["git", "commit", "-m", f"[referee] update snapshots & policies at {ts}"], check=True)
        
        # Push
        subprocess.run(["git", "push"], check=True)
        print("[referee] Changes pushed to GitHub.")
    except subprocess.CalledProcessError as e:
        print(f"[referee] Git push failed: {e}")

# ========== Discord Bot ==========
intents = discord.Intents.default()
client = discord.Client(intents=intents)

# Add a shared sender like your other bots
async def send_to_default_channel(content: str):
    for guild in client.guilds:
        for channel in guild.text_channels:
            if channel.permissions_for(guild.me).send_messages:
                await channel.send(content)
                return

@client.event
async def on_ready():
    print(f"[referee] Logged in as {client.user}")
    rows = await compute_leaderboard()
    await send_to_default_channel(format_report(rows))

async def referee_loop():
    await client.wait_until_ready()
    while not client.is_closed():
        try:
            rows = await compute_leaderboard()
            await send_to_default_channel(format_report(rows))

            # --- snapshots writer (append one line per bot) ---
            now_iso = datetime.now(timezone.utc).isoformat()
            with SNAP_PATH.open("a", encoding="utf-8") as w:
                for bot, total, usd, _detail in rows:
                    rec = {
                        "timestamp": now_iso,
                        "bot": bot.lower(),
                        "total_usd": float(total),
                        "usd_cash": float(usd),
                    }
                    w.write(json.dumps(rec) + "\n")
            
            git_push_updates()

        except Exception as e:
            print(f"⚠️ Referee loop error: {e}")

        await asyncio.sleep(POST_EVERY)

@client.event
async def on_message(message):
    # Ignore messages from the bot itself
    if message.author == client.user:
        return

    # Check if the bot is mentioned
    if client.user.mentioned_in(message):
        try:
            rows = await compute_leaderboard()
            report = format_report(rows)
            await message.channel.send(report)
        except Exception as e:
            await message.channel.send(f"⚠️ Could not fetch leaderboard: {e}")


@client.event
async def setup_hook():
    client.loop.create_task(referee_loop())

if __name__ == "__main__":
    client.run(REF_TOKEN)
