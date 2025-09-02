"""This script contains helper functioned required for agents.py, stored here
to ease with debugging of the agents.py script. This includes tools which prevent
bot instructed trade, generic helpers that help to parse out date related queries and
format currencies, functions that assist with working with the coin gecko API, and
functions to assist working with the simulated wallets"""

from typing import Dict, List, Optional, Tuple
import os, time, re, math, json, calendar
from datetime import datetime, timezone
import requests
import contextvars

# =============================
# Preventing bot instructed trade
# =============================

REQUEST_CTX = contextvars.ContextVar("REQUEST_CTX", default={"source": "unknown", "allow_tools": False})

def set_request_context(source: str, allow_tools: bool):
    REQUEST_CTX.set({"source": source, "allow_tools": allow_tools})

def _tools_allowed() -> bool:
    return bool(REQUEST_CTX.get().get("allow_tools", False))

# ==============================
# Generic Helpers
# ==============================

MONTHS = {m.lower(): i for i, m in enumerate(calendar.month_name) if m}
MONTHS.update({m.lower(): i for i, m in enumerate(calendar.month_abbr) if m})

def _parse_month_year_window(text: str):
    """Return (start_ts,end_ts) in UNIX seconds for 'Sep 2022' style queries, else None."""
    t = text.lower()
    for name, mnum in MONTHS.items():
        token = f"{name} "
        if token in t:
            idx = t.find(token)
            rest = t[idx + len(token):]
            year = None
            for part in rest.split():
                if part.isdigit() and len(part) == 4:
                    year = int(part)
                    break
            if year:
                start = datetime(year, mnum, 1, tzinfo=timezone.utc)
                end = datetime(year + (mnum == 12), 1 if mnum == 12 else mnum + 1, 1, tzinfo=timezone.utc)
                return int(start.timestamp()), int(end.timestamp())
    return None

def _parse_year_window(text: str):
    """Return (start_ts,end_ts) in UNIX seconds for plain year queries like 'in 2023'."""
    m = re.search(r"\b(19|20)\d{2}\b", text)
    if not m:
        return None
    year = int(m.group(0))
    start = datetime(year, 1, 1, tzinfo=timezone.utc)
    end   = datetime(year + 1, 1, 1, tzinfo=timezone.utc)
    return int(start.timestamp()), int(end.timestamp())

def _avg_from_prices(prices: list[list[float]]) -> float:
    """prices is [[ms, price], ...]"""
    if not prices:
        return float("nan")
    vals = [p[1] for p in prices if isinstance(p, (list, tuple)) and len(p) >= 2]
    return sum(vals) / len(vals) if vals else float("nan")

def _pretty_money(x: float) -> str:
    if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
        return "N/A"
    if x >= 1000:
        return f"${x:,.0f}"
    if x >= 1:
        return f"${x:,.2f}"
    return f"${x:.6f}"

def _decide_intent(q: str) -> str:
    """Rough routing: 'current_price' | 'historical_chart' | 'market_stats' | 'unknown'."""
    ql = q.lower()
    if any(x in ql for x in ["now", "current", "spot", "price right now"]):
        return "current_price"
    if any(x in ql for x in ["chart", "over", "history", "last", "past", "since", "trend", "moving average", "average"]):
        return "historical_chart"
    if any(x in ql for x in ["market cap", "volume", "dominance", "circulating", "supply", "stats"]):
        return "market_stats"
    return "unknown"

def _leaderboard_rank(this_bot: str) -> tuple[int, int]:
    """
    Return (rank, total_bots) using current mark-to-market totals.
    Rank is 1-based; 1 means first place.
    """
    # infer wallet directory from any bot's wallet path
    wdir = Path(_wallet_path(this_bot)).parent
    if not wdir.exists():
        return (1, 1)

    results = []
    for fp in wdir.glob("*.json"):
        try:
            name = fp.stem
            w = _load_wallet(name)
            total, _ = _mark_wallet(w)   # mark to market for that wallet
            results.append((name.lower(), float(total or 0.0)))
        except Exception:
            # be resilient—ignore bad/missing wallets
            continue

    if not results:
        return (1, 1)

    # sort by total desc, find our position
    results.sort(key=lambda x: x[1], reverse=True)
    names = [n for n, _ in results]
    try:
        pos = names.index(this_bot.lower())
        return (pos + 1, len(results))
    except ValueError:
        # our wallet missing? treat as last
        return (len(results), len(results))


# ==============================
# CoinGecko (Pro) Helpers
# ==============================

CG_BASE = "https://pro-api.coingecko.com/api/v3"
CG_KEY = os.getenv("COINGECKO_API_KEY")  # Analyst tier key

if not CG_KEY:
    print("⚠️ COINGECKO_API_KEY not set; answer_crypto_question will fail until configured.")

def _cg_headers():
    return {"x-cg-pro-api-key": CG_KEY}

# simple cache for /coins/list (ID lookups)
_COINS_CACHE: Dict[str, str] = {}   # maps SYMBOL/NAME -> id
_COINS_CACHE_TTL = 60 * 60
_COINS_CACHE_TIME = 0.0

def _refresh_coins_cache() -> None:
    global _COINS_CACHE, _COINS_CACHE_TIME
    if _COINS_CACHE and (time.time() - _COINS_CACHE_TIME) < _COINS_CACHE_TTL:
        return
    resp = requests.get(f"{CG_BASE}/coins/list", headers=_cg_headers(), timeout=30)
    resp.raise_for_status()
    _COINS_CACHE = {}
    for row in resp.json():
        sym = (row.get("symbol") or "").upper()
        name = (row.get("name") or "").upper()
        cid = row.get("id")
        if not cid:
            continue
        _COINS_CACHE.setdefault(sym, cid)
        if sym == "BTC":
            _COINS_CACHE.setdefault("XBT", cid)
        if name:
            _COINS_CACHE.setdefault(name, cid)
    _COINS_CACHE_TIME = time.time()

def _request_with_retry(url: str, params: dict, max_tries: int = 5) -> dict:
    """Requests with backoff + a few auto-fixes."""
    params = {k: v for k, v in params.items() if v is not None}
    wait = 1.0
    last_err = None
    for _ in range(max_tries):
        try:
            r = requests.get(url, headers=_cg_headers(), params=params, timeout=60)
            if r.status_code == 429:
                raise requests.HTTPError("429 Too Many Requests", response=r)
            r.raise_for_status()
            data = r.json()
            if isinstance(data, dict) and data.get("status", {}).get("error_code"):
                raise requests.HTTPError(str(data), response=r)
            return data
        except requests.HTTPError as e:
            last_err = e
            code = getattr(e.response, "status_code", None)
            if code in (400, 422):
                if params.get("vs_currency") and params["vs_currency"].lower() != "usd":
                    params["vs_currency"] = "usd"
                if "days" in params and int(params["days"]) > 365 * 3:
                    params["days"] = 365
            elif code in (500, 502, 503, 504, 429):
                time.sleep(wait)
                wait = min(wait * 2, 8)
                continue
            time.sleep(min(wait, 2))
        except (requests.ConnectionError, requests.Timeout) as e:
            last_err = e
            time.sleep(wait)
            wait = min(wait * 2, 8)
    raise RuntimeError(f"CoinGecko request failed after {max_tries} attempts: {last_err}")

def _search_and_cache(query: str) -> Optional[str]:
    """Search CoinGecko for a coin ID by symbol or name and cache it."""
    _refresh_coins_cache()
    q_upper = query.upper()
    if q_upper in _COINS_CACHE:
        return _COINS_CACHE[q_upper]
    try:
        r = requests.get(f"{CG_BASE}/search", headers=_cg_headers(), params={"query": query}, timeout=15)
        r.raise_for_status()
        results = r.json().get("coins", [])
    except Exception as e:
        print(f"⚠️ CoinGecko search failed for '{query}': {e}")
        return None
    if not results:
        return None
    match = next((c for c in results if (c.get("symbol") or "").upper() == q_upper), results[0])
    cid  = match.get("id")
    sym  = (match.get("symbol") or "").upper()
    name = (match.get("name") or "").upper()
    if cid:
        if sym:  _COINS_CACHE.setdefault(sym, cid)
        if name: _COINS_CACHE.setdefault(name, cid)
        _COINS_CACHE.setdefault(q_upper, cid)
    return cid

CANONICAL_IDS = {
    "ETH": "ethereum",
    "BTC": "bitcoin",
    "XBT": "bitcoin",
    "SOL": "solana",
    "ADA": "cardano",
}

def _resolve_ids(mentions: list[str]) -> list[str]:
    """Map symbols to CoinGecko ids, falling back to search."""
    ids: List[str] = []
    for m in mentions:
        key = m.strip().upper().replace("USD", "").strip()
        if key in CANONICAL_IDS:
            cid = CANONICAL_IDS[key]
        else:
            cid = _search_and_cache(key)
        if cid and cid not in ids:
            ids.append(cid)
    return ids

def _safe_mentions_from_text(question: str, coins: str = "") -> List[str]:
    """Extract likely symbols; filter by CG cache + canonical map; dedupe."""
    _refresh_coins_cache()
    cg_symbols = set(k for k in _COINS_CACHE.keys() if k.isupper() and len(k) <= 6)

    mentions: List[str] = []
    for t in re.findall(r"[A-Za-z]{2,6}", question.upper()):
        if t in CANONICAL_IDS or t in cg_symbols:
            mentions.append(t)

    if coins:
        if isinstance(coins, str):
            mentions += [c.strip().upper() for c in coins.split(",") if c.strip()]
        elif isinstance(coins, (list, tuple)):
            mentions += [str(c).strip().upper() for c in coins if str(c).strip()]

    seen = set()
    mentions = [m for m in mentions if not (m in seen or seen.add(m))]
    return mentions[:5]  # cap to 5 to be safe

def _cg_market_chart_range(coin_id: str, vs_currency: str, start_ts: int, end_ts: int, interval="daily"):
    return _request_with_retry(
        f"{CG_BASE}/coins/{coin_id}/market_chart/range",
        {"vs_currency": vs_currency, "from": start_ts, "to": end_ts, "interval": interval}
    )

def _parse_time_horizon(q: str) -> Tuple[int, str]:
    """Return (days, interval_hint). Defaults to (30, 'daily')."""
    ql = q.lower()
    m = re.search(r"(\d+)\s*(min|mins|minutes|hour|hours|day|days|week|weeks|month|months|year|years|d|h|w|m|y)\b", ql)
    if m:
        n = int(m.group(1))
        u = m.group(2)
        if u in ("min","mins","minutes"): return (1, "hourly") if n <= 48 else (7, "hourly")
        if u in ("hour","hours","h"):     return (1, "hourly") if n <= 48 else (7, "hourly")
        if u in ("day","days","d"):       return (max(1, n), "daily" if n >= 7 else "hourly")
        if u in ("week","weeks","w"):     return (7 * n, "daily")
        if u in ("month","months","m"):   return (30 * n, "daily")
        if u in ("year","years","y"):     return (365 * n, "daily")
    if "24h" in ql or "24 h" in ql or "last day" in ql: return (1, "hourly")
    if "week" in ql:   return (7, "daily")
    if "month" in ql:  return (30, "daily")
    if "year" in ql or "ytd" in ql: return (365, "daily")
    return (30, "daily")

# ==============================
# Wallet helpers (file-backed)
# ==============================
WALLET_DIR = os.path.join(os.getcwd(), "wallets")
os.makedirs(WALLET_DIR, exist_ok=True)

FEE_BPS_DEFAULT = int(os.getenv("PAPER_TRADE_FEE_BPS", "10"))   # 10 bps = 0.10%
SLIPPAGE_BPS_DEFAULT = int(os.getenv("PAPER_TRADE_SLIPPAGE_BPS", "5"))  # 5 bps = 0.05%

def _wallet_path(bot_name: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_\-]", "_", bot_name)
    return os.path.join(WALLET_DIR, f"{safe}.json")

def _init_wallet(bot_name: str) -> dict:
    return {
        "bot": bot_name,
        "balances": { "USD": 10_000.0 },  # starting cash
        "trade_log": [],
        "created_at": datetime.now(timezone.utc).isoformat(),
        "last_mark": None
    }

def _load_wallet(bot_name: str) -> dict:
    p = _wallet_path(bot_name)
    if not os.path.exists(p):
        w = _init_wallet(bot_name)
        with open(p, "w") as f:
            json.dump(w, f, indent=2)
        return w
    with open(p) as f:
        return json.load(f)

def _save_wallet(wallet: dict) -> None:
    p = _wallet_path(wallet["bot"])
    with open(p, "w") as f:
        json.dump(wallet, f, indent=2)

def _spot_price_usd(coin_id: str) -> float:
    params = {
        "vs_currency": "usd", "ids": coin_id,
        "order": "market_cap_desc", "per_page": 1, "page": 1
    }
    raw = _request_with_retry(f"{CG_BASE}/coins/markets", params)
    if isinstance(raw, list) and raw:
        return float(raw[0].get("current_price") or 0.0)
    return float("nan")

# ======= Prompting bots to explain their choices ===
# ===================================================
# --- Persona + policy context for explanations ---
def _persona_desc(bot: str) -> str:
    b = bot.lower()
    if "bitbot" in b:
        return "Jordan Belfort-style, cocky momentum chaser who chases liquidity and velocity."
    if "maxibit" in b:
        return "Bitcoin maxi who prefers BTC/ETH unless alts clearly outperform; smug but analytical."
    if "bearbot" in b:
        return "Cautious, risk-averse, capital preservation first; trims losers and adds on strength carefully."
    if "badbytebillie" in b:
        return "Dry, deadpan, trend-riding with snark; will press winners and cut laggards quickly."
    return "Pragmatic crypto trader."

def _policy_brief(bot: str) -> str:
    # Best-effort read; OK if policies aren’t set up yet
    try:
        from policy_helper_functions import load_policy
        p = load_policy(bot)
        return (
            f"risk_tolerance={p.risk_tolerance:.2f}, cash={p.target_cash_pct:.2f}, "
            f"max_pos={p.max_positions}, min_liq_usd={int(p.min_liquidity_usd):,}, "
            f"prefer_majors={p.prefer_majors_weight:.2f}, stop={p.stop_loss_bps}bps, "
            f"take={p.take_profit_bps}bps"
        )
    except Exception:
        return "no explicit policy file; using persona defaults"

