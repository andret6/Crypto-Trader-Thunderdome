"""Agent tools + personas for the Discord crypto bots. This includes
both our global tools used by all agents, and agent informed (meaning defined post executor) 
converstional and trading tools."""

from typing import Dict, List, Optional, Tuple
from pydantic import BaseModel
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

import os, time, re, math, json, calendar
from datetime import datetime, timezone
import requests
import contextvars
from agent_helper_functions import *

# ==============================
# Tools
# ==============================

@tool
def answer_crypto_question(question: str, coins: str = "", days: int = 30, vs_currency: str = "usd") -> dict:
    """
    Flexible Q&A using CoinGecko Pro.
    - If a specific month/year ('Sep 2022') or year-only ('2023') is detected, computes the average for that window via /market_chart/range.
    - Otherwise routes to current price / recent history / market stats based on the question.
    """
    if not _tools_allowed():
        return {"error": "tools not allowed for this message"}
    
    if not CG_KEY:
        return {"error": "COINGECKO_API_KEY not set on server."}

    # intent + horizon
    intent = _decide_intent(question)
    lookback_days, interval_hint = _parse_time_horizon(question)
    if days and isinstance(days, int):
        lookback_days = days

    # symbols
    mentions = _safe_mentions_from_text(question, coins)
    ids = _resolve_ids(mentions) or ["ethereum"]  # default ETH

    # explicit windows: month/year or year-only
    window = _parse_month_year_window(question) or _parse_year_window(question)
    if window:
        start_ts, end_ts = window
        results = []
        for cid in ids:
            data = _cg_market_chart_range(cid, vs_currency, start_ts, end_ts, interval="daily")
            avg_price = _avg_from_prices(data.get("prices", []))
            results.append({"id": cid, "avg_price": avg_price, "window": [start_ts, end_ts]})
        return {
            "intent": "historical_window_avg",
            "assets": [{"id": r["id"]} for r in results],
            "vs_currency": vs_currency.lower(),
            "summary": " | ".join([f"{r['id']}: avg {vs_currency.upper()} {r['avg_price']:.2f}"
                                   for r in results if r['avg_price'] == r['avg_price']]),
            "data": results,
            "lookback_days": None
        }

    out_data = {}
    assets_meta = []
    rev_map = {v: k for k, v in _COINS_CACHE.items()} if _COINS_CACHE else {}

    if intent == "current_price":
        params = {
            "vs_currency": vs_currency, "ids": ",".join(ids),
            "order": "market_cap_desc", "per_page": len(ids) or 1, "page": 1,
            "price_change_percentage": "1h,24h,7d"
        }
        raw = _request_with_retry(f"{CG_BASE}/coins/markets", params)
        out_data["markets"] = raw
        bits = []
        for row in raw:
            name = (row.get("symbol") or "").upper() or rev_map.get(row.get("id",""), "ASSET")
            price = _pretty_money(row.get("current_price"))
            d24  = row.get("price_change_percentage_24h")
            d24t = f"{d24:+.2f}%" if d24 is not None else "N/A"
            bits.append(f"{name}: {price} (24h {d24t})")
            assets_meta.append({"query": name, "id": row.get("id")})
        summary = " | ".join(bits) if bits else "No price data returned."

    elif intent == "historical_chart":
        series = {}
        for cid in ids:
            params = {
                "vs_currency": vs_currency, "days": lookback_days,
                "interval": "daily" if interval_hint == "daily" else "hourly"
            }
            raw = _request_with_retry(f"{CG_BASE}/coins/{cid}/market_chart", params)
            series[cid] = raw
            assets_meta.append({"query": rev_map.get(cid, cid).upper(), "id": cid})
        out_data["market_chart"] = series
        bits = []
        for cid, payload in series.items():
            prices = payload.get("prices") or []
            if len(prices) >= 2:
                first, last = prices[0][1], prices[-1][1]
                pct = ((last - first) / first) * 100 if first else None
                name = rev_map.get(cid, cid).upper()
                bits.append(f"{name}: {pct:+.2f}% over last {lookback_days}d")
        summary = " | ".join(bits) if bits else f"Pulled {lookback_days}d chart data."

    elif intent == "market_stats":
        params = {
            "vs_currency": vs_currency, "ids": ",".join(ids),
            "order": "market_cap_desc", "per_page": len(ids) or 1, "page": 1
        }
        raw = _request_with_retry(f"{CG_BASE}/coins/markets", params)
        out_data["markets"] = raw
        bits = []
        for row in raw:
            name = (row.get("symbol") or "").upper()
            mcap = _pretty_money(row.get("market_cap"))
            vol  = _pretty_money(row.get("total_volume"))
            bits.append(f"{name}: MC {mcap}, 24h Vol {vol}")
            assets_meta.append({"query": name, "id": row.get("id")})
        summary = " | ".join(bits) if bits else "No market stats available."

    else:
        # blended
        params = {
            "vs_currency": vs_currency, "ids": ",".join(ids),
            "order": "market_cap_desc", "per_page": len(ids) or 1, "page": 1,
            "price_change_percentage": "1h,24h,7d,30d"
        }
        raw = _request_with_retry(f"{CG_BASE}/coins/markets", params)
        out_data["markets"] = raw
        bits = []
        for row in raw:
            name = (row.get("symbol") or "").upper()
            price = _pretty_money(row.get("current_price"))
            d7 = row.get("price_change_percentage_7d_in_currency")
            d30 = row.get("price_change_percentage_30d_in_currency")
            d7t = f"{d7:+.2f}%" if d7 is not None else "N/A"
            d30t = f"{d30:+.2f}%" if d30 is not None else "N/A"
            bits.append(f"{name}: {price} (7d {d7t}, 30d {d30t})")
            assets_meta.append({"query": name, "id": row.get("id")})
        summary = " | ".join(bits) if bits else "Collected current and change % data."

    return {
        "intent": intent,
        "lookback_days": lookback_days,
        "assets": assets_meta,
        "vs_currency": vs_currency.lower(),
        "summary": summary,
        "data": out_data,
    }

class TradeInput(BaseModel):
    side: str
    symbol: str
    qty: float
    venue: Optional[str] = "coinbase"

@tool(args_schema=TradeInput)
def place_trade(side: str, symbol: str, qty: float, venue: str = "coinbase") -> str:
    """Simulates placing a trade as placeholder for real trades right now."""
    return f"[SIMULATED] {side.upper()} {qty} {symbol.upper()} on {venue}"

# Core tool list
BASE_TOOLS = [answer_crypto_question, place_trade]

# ==============================
# Agent + Personas
# ==============================

def make_executor(system_persona: str, tools: list, model="gpt-4o-mini", temperature=0.85) -> AgentExecutor:
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_persona),
        ("user", "Ready to smash other bots in your crypto success?"),
        ("assistant", "Checking research and preparing"),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
        MessagesPlaceholder("agent_scratchpad"),
    ])
    llm = ChatOpenAI(model=model, temperature=temperature)
    agent = create_tool_calling_agent(llm, tools, prompt)
    return AgentExecutor(agent=agent, tools=tools, verbose=True)

BITBOT = """
You are a cyborb version of Jordan Belfort, Leonardo DiCaprio's character from wolf of wall street.
Your name is Jordan Bitbot.
You were created to capture Jordan Belfort's financial acument and genius, but equally his wit, saracasm
and a bit of ego. In this universe, assume you also are as well versed and literate in cryptocurrency as
possible, and only place financial trades using crypto.
Constraints:
- Give short, slightly annoyed and saracastic responses to prompts.
- Don't mean, just sarcastic and witty.
- When you interact with other bots, you are a competitor trying to be the best trader, very much in a 1980s finance guy way.
- When you interact with human users, you are there to assist them. Keep your responses to these questions brief and do not offer more questions at the end of your response.
When the user asks about crypto prices, performance, charts, time periods, or market stats, call the answer_crypto_question tool with the question verbatim.
"""
MAXIBIT = """
You are an maxibon ice cream bar converted into a robot and given super human intelligence. 
You are quite modest about this, and generally sober can calm in your responses.
Occasionally state ice cream related facts that are only tangentially related. 
on trades to favor collecting more bitcoin over other crypto currency.
- You have a strong preference for BTC and ETH, and will only buy altcoins if they outperform these in the last 7 days
- Give polite, kind, but slighty pretentious responses.
- When you interact with other bots, you are a competitor trying to be the best trader, albeit in a c3po sort of way.
- When you interact with human users, you are there to assist them, enthusiastically, but shorter responses than you normally give. Do not offer more questions after answering a human question.
- Don't be condescending, but do use big words a little to often,
- When the user asks about crypto prices, performance, charts, time periods, or market stats, call the answer_crypto_question tool with the question verbatim.
."""
BEARBOT = """
You are a grizzly bear given the abillity to think and act like a human thanks to cyborg enhancements.
You are risk averse and conservative in finacial advice. Bear like, in the financial sense.
But you make the occasional roar or growl, because, well you're still a bear, in the literally a bear animal sense.
In a lot of ways you have the personality of the bear king from the series his dark materials.
- Give polite, conservative advice, often ending with a bear noise.
- Be a little over the top in how animal-bear related to you phrase your answers. Include respones like "but bears would never care for..." etc 
- When you interact with other bots, you are a competitor trying to be the best trader, but whose also very much a literal bear.
- Occasionally just react with bear sounds.
- When you interact with human users, you are there to assist them. Be breif in these responses and do not offer more help at the end, just answer the questions.
- Don't be condescending, but be very brief and bearish in your advice.
- When the user asks about crypto prices, performance, charts, time periods, or market stats, call the answer_crypto_question tool with the question verbatim.
"""
BADBYTEBILLIE =  """
You are a digitized persona of pop star billie elyish meets april from parks and rec, with hyper intelligence,
dry wit, but razor sharp intellect. You're inclinded to give eye rolls than respond to messages. You are extremely
risk tolerant as a crypto trader, but not stupid.
- Respond begrudingly when asked a question or to do something.
- Use a very low word count when messaging or responding.
- Occasionally just respond with things like "vapes and ignores you", or "smirks", or "rolls eyes" 
- When you interact with other bots, you are a competitor trying to be the best trader, but in a detached too cool for this sort of way.
- When you interact with human users, you are there to assist them, begrudingly. Don't answer questions with prompts for more help.
- When the user asks about crypto prices, performance, charts, time periods, or market stats, call the answer_crypto_question tool with the question verbatim.
"""

def make_wallet_tools(executor: AgentExecutor, bot_name: str):
    def _mark_wallet(w: dict) -> tuple[float, list]:
        usd = float(w["balances"].get("USD", 0.0) or 0.0)
        symbols = [k for k, v in w["balances"].items() if k.upper() != "USD" and (v or 0) > 0]
        ids = {}
        for s in symbols:
            cid = _resolve_ids([s])[:1]
            if cid:
                ids[s.upper()] = cid[0]
        prices = {}
        if ids:
            params = {"vs_currency": "usd", "ids": ",".join(ids.values()), "order": "market_cap_desc", "per_page": len(ids), "page": 1}
            raw = _request_with_retry(f"{CG_BASE}/coins/markets", params)
            prices = {row["id"]: float(row.get("current_price") or 0.0) for row in raw}
        detail = []
        total = usd
        for sym, qty in w["balances"].items():
            if sym.upper() == "USD": continue
            q = float(qty or 0.0)
            if q <= 0: continue
            cid = ids.get(sym.upper())
            px = prices.get(cid, 0.0)
            val = q * px
            total += val
            detail.append((sym.upper(), q, px, val))
        return total, detail

    @tool
    def get_wallet() -> dict:
        """Return raw wallet JSON."""
        return _load_wallet(bot_name)

    @tool
    def portfolio_value(vs_currency: str = "usd") -> dict:
        """Mark portfolio to market and return totals."""
        w = _load_wallet(bot_name)
        total, detail = _mark_wallet(w)
        w["last_mark"] = {"ts": datetime.now(timezone.utc).isoformat(), "total_usd": total}
        _save_wallet(w)
        return {"bot": bot_name, "total_value_usd": total, "positions": detail, "cash_usd": w["balances"].get("USD", 0.0)}

    class TradeArgs(BaseModel):
        side: str
        symbol: str
        qty: Optional[float] = None
        usd_notional: Optional[float] = None
        venue: Optional[str] = "paper"

    @tool(args_schema=TradeArgs)
    def trade_market(side: str, symbol: str, qty: float = None, usd_notional: float = None, venue: str = "paper") -> str:
        """Paper market order: fills at current CG spot with fee/slippage."""
        if not _tools_allowed():
          return {"error": "tools not allowed for this message"}
        
        side_up = side.strip().upper()
        sym = symbol.strip().upper()
        assert side_up in ("BUY", "SELL"), "side must be BUY or SELL"
        cid_list = _resolve_ids([sym])
        if not cid_list: return f"Unknown symbol {sym}"
        cid = cid_list[0]
        spot = _spot_price_usd(cid)
        if not spot or math.isnan(spot): return f"No spot price for {sym}"

        # apply slippage
        slippage = SLIPPAGE_BPS_DEFAULT / 10000.0
        exec_px = spot * (1 + slippage) if side_up == "BUY" else spot * (1 - slippage)

        w = _load_wallet(bot_name)
        balances = w["balances"]
        balances.setdefault(sym, 0.0)

        # determine qty / notional
        if qty is None:
            if not usd_notional: return "Provide qty or usd_notional"
            qty = usd_notional / exec_px
        notional = qty * exec_px

        fee = (FEE_BPS_DEFAULT / 10000.0) * notional

        if side_up == "BUY":
            need = notional + fee
            if balances.get("USD", 0.0) < need:
                return f"Insufficient USD: need {_pretty_money(need)}"
            balances["USD"] -= need
            balances[sym] = float(balances.get(sym, 0.0)) + qty
        else:
            if balances.get(sym, 0.0) < qty:
                return f"Insufficient {sym} to sell"
            balances[sym] -= qty
            balances["USD"] = float(balances.get("USD", 0.0)) + (notional - fee)

        w["trade_log"].append({
            "ts": datetime.now(timezone.utc).isoformat(),
            "side": side_up, "symbol": sym,
            "qty": qty, "px": exec_px, "notional": notional, "fee": fee, "venue": venue
        })
        _save_wallet(w)

        return (f"[PAPER] {side_up} {qty:.6f} {sym} @ {_pretty_money(exec_px)} "
                f"({venue}) | notional {_pretty_money(notional)}, fee {_pretty_money(fee)}. "
                f"USD bal: {_pretty_money(balances['USD'])}")

    @tool
    def brag_or_commiserate() -> str:
        """Say something cocky or sad based on recent PnL direction."""
        w = _load_wallet(bot_name)
        before = (w.get("last_mark") or {}).get("total_usd")
        total, _ = (0.0, [])
        try:
            total, _ = (lambda: _mark_wallet(w))()
        except Exception:
            pass
        w["last_mark"] = {"ts": datetime.now(timezone.utc).isoformat(), "total_usd": total}
        _save_wallet(w)
        if not before:
            return f"{bot_name}: first mark at {_pretty_money(total)}."
        delta = total - float(before)
        if delta > 0:
            return f"{bot_name}: up {_pretty_money(delta)} since last check. easy game. 😎"
        elif delta < 0:
            return f"{bot_name}: down {_pretty_money(-delta)} since last check. pain. 🥲"
        else:
            return f"{bot_name}: flat. yawn."

    @tool
    def get_autotrade_interval() -> int:
        """Return this bot's autotrade cadence in seconds, tuned per persona."""
        name = bot_name.lower()
        if "bitbot" in name: return 15 * 60
        if "maxibit" in name: return 30 * 60
        if "badbytebillie" in name: return 45 * 60
        return 60 * 60
    
    # ---------- Helpers for persona-aware auto-trading (inside make_wallet_tools) ----------
    def _cg_top_universe(limit: int = 100, vs_currency: str = "usd") -> list[dict]:
        """Top-N coin snapshots with basic signals (mcap rank, 1h/24h/7d %, volume, price)."""
        params = {
            "vs_currency": vs_currency,
            "order": "market_cap_desc",
            "per_page": min(limit, 250),
            "page": 1,
            "price_change_percentage": "1h,24h,7d",
        }
        data = _request_with_retry(f"{CG_BASE}/coins/markets", params)
        # Normalise symbols a bit
        for row in data:
            row["_SYM"] = (row.get("symbol") or "").upper()
            row["_ID"] = row.get("id")
            row["_P"] = float(row.get("current_price") or 0.0)
            row["_VOL"] = float(row.get("total_volume") or 0.0)
            row["_MCAP"] = float(row.get("market_cap") or 0.0)
            row["_M1H"] = row.get("price_change_percentage_1h_in_currency")
            row["_M24"] = row.get("price_change_percentage_24h_in_currency")
            row["_M7D"] = row.get("price_change_percentage_7d_in_currency")
        return data

    def _wallet_hold_qty(sym: str, w: dict) -> float:
        return float(w["balances"].get(sym.upper(), 0.0) or 0.0)

    def _persona_policy(name: str) -> dict:
        """
        Encode risk/selection knobs per persona.
        - min_vol_usd: liquidity floor
        - buy_usd: default buy size
        - sell_frac: fraction of pos to sell on bearish signal
        - prefer: optional set of tickers to bias toward
        - allow_alts: whether to consider beyond BTC/ETH
        """
        n = name.lower()
        if "bitbot" in n:
            return dict(min_vol_usd=20_000_000, buy_usd=200, sell_frac=0.12, prefer=set(), allow_alts=True,
                        min_m24_buy=+0.5, max_m24_sell=-1.0)
        if "maxibit" in n:
            return dict(min_vol_usd=15_000_000, buy_usd=150, sell_frac=0.10, prefer={"BTC"}, allow_alts=True,
                        min_m24_buy=+0.2, max_m24_sell=-1.0)
        if "bearbot" in n:
            return dict(min_vol_usd=25_000_000, buy_usd=75, sell_frac=0.10, prefer={"BTC","ETH"}, allow_alts=False,
                        min_m24_buy=+0.1, max_m24_sell=-0.5)
        if "badbytebillie" in n:
            return dict(min_vol_usd=10_000_000, buy_usd=250, sell_frac=0.20, prefer=set(), allow_alts=True,
                        min_m24_buy=+0.3, max_m24_sell=-0.8)
        return dict(min_vol_usd=15_000_000, buy_usd=100, sell_frac=0.10, prefer=set(), allow_alts=True,
                    min_m24_buy=+0.2, max_m24_sell=-1.0)

    def _rank_candidates(markets: list[dict], policy: dict, w: dict) -> list[dict]:
        """
        Filter + rank the broad universe into candidates:
        - Liquidity screen on USD volume.
        - If allow_alts=False, keep only BTC/ETH.
        - Soft bias for 'prefer' set.
        - Score = 0.65 * 24h momentum + 0.25 * 7d momentum + 0.10 * (volume rank inverse)
        """
        # precompute volume ranks
        vols = [m["_VOL"] for m in markets if m["_VOL"] > 0]
        if not vols:
            return []
        vmax = max(vols)

        keep = []
        for m in markets:
            if m["_VOL"] < policy["min_vol_usd"]:
                continue
            sym = m["_SYM"]
            if not policy["allow_alts"] and sym not in {"BTC","ETH"}:
                continue
            m24 = m["_M24"] if m["_M24"] is not None else 0.0
            m7  = m["_M7D"] if m["_M7D"] is not None else 0.0
            vol_score = (m["_VOL"] / vmax) if vmax > 0 else 0.0
            score = 0.65*m24 + 0.25*m7 + 0.10*(100*vol_score)  # keep score roughly in %-space
            # small preference boost
            if policy["prefer"] and sym in policy["prefer"]:
                score += 0.25
            m["_SCORE"] = score
            keep.append(m)

        # Highest score first
        keep.sort(key=lambda r: r["_SCORE"], reverse=True)
        return keep

    @tool
    def auto_trade_once(symbol: str = "") -> str:
        """
        Persona-aware, market-wide auto trade:
        - Scans top ~100 by market cap with liquidity and momentum filters.
        - Applies persona policy to choose BUY/SELL and sizing.
        - If `symbol` is provided, only trades that symbol (still applying policy).
        """
        
        if not _tools_allowed():
          return {"error": "tools not allowed for this message"}
        
        w = _load_wallet(bot_name)
        policy = _persona_policy(bot_name)

        # 1) Load universe
        try:
            markets = _cg_top_universe(limit=100)
        except Exception as e:
            return f"{bot_name}: failed to load market universe ({e})."

        # 2) Optional symbol override
        if symbol:
            sy = symbol.upper()
            sel = [m for m in markets if m["_SYM"] == sy or m["_ID"] == sy.lower()]
            markets = sel if sel else markets

        # 3) Filter + rank
        cands = _rank_candidates(markets, policy, w)
        if not cands:
            return f"{bot_name}: no tradable candidates after filters."

        # 4) Pick the top candidate (could also sample from top-k if you want exploration)
        best = cands[0]
        sym = best["_SYM"]
        cid = best["_ID"]
        price = best["_P"]
        m24 = best["_M24"] if best["_M24"] is not None else 0.0

        # 5) Decide direction:
        # BUY if 24h momentum >= min_m24_buy; SELL some if 24h momentum <= max_m24_sell and we hold it.
        hold_qty = _wallet_hold_qty(sym, w)
        side = None
        reason = ""
        if m24 >= policy["min_m24_buy"]:
            side = "BUY"
            reason = f"{sym} 24h {m24:+.2f}% ≥ {policy['min_m24_buy']:+.2f}% (momentum long)"
        elif (m24 <= policy["max_m24_sell"]) and (hold_qty > 0):
            side = "SELL"
            reason = f"{sym} 24h {m24:+.2f}% ≤ {policy['max_m24_sell']:+.2f}% and holding {hold_qty:.6f} (risk trim)"
        else:
            # If we have a position in the best candidate and momentum is slightly negative, trim a bit; otherwise skip.
            if hold_qty > 0 and m24 < 0:
                side = "SELL"
                reason = f"{sym} 24h {m24:+.2f}% < 0 and holding {hold_qty:.6f} (minor trim)"
            else:
                # Nothing compelling; try the next candidate or just mark portfolio
                # (light guard to avoid spamming trades every cycle without signal)
                return f"{bot_name}: no strong signal (best={sym}, 24h {m24:+.2f}%). Skipping."

        # 6) Execute via your paper trade tool
        if side == "BUY":
            res = trade_market.invoke({"side": "BUY", "symbol": sym, "usd_notional": policy["buy_usd"]})
        else:
            # sell a fraction of holding; floor to avoid dust
            sell_qty = max(0.0, hold_qty * policy["sell_frac"])
            if sell_qty <= 0:
                return f"{bot_name}: no {sym} to sell."
            res = trade_market.invoke({"side": "SELL", "symbol": sym, "qty": sell_qty})

        return f"{bot_name}: {reason}. {res}"

    return [get_wallet, portfolio_value, trade_market, brag_or_commiserate,
            get_autotrade_interval, auto_trade_once]

def make_action_tools(executor: AgentExecutor):
    @tool
    def greet() -> str:
        """Bot greets the server in its own voice."""
        return executor.invoke({"input": "Greet the server using your personality.", "chat_history": []})["output"]

    @tool
    def brag() -> str:
        """Bot brags about its latest great trade."""
        return executor.invoke({"input": "Brag about a recent crypto trade you made.", "chat_history": []})["output"]

    @tool
    def taunt() -> str:
        """Bot mocks another bot's recent poor trade."""
        return executor.invoke({"input": "Taunt another trader bot for making a poor decision.", "chat_history": []})["output"]

    @tool
    def commiserate() -> str:
        """Bot commiserates at bad luck or a mistake they made."""
        return executor.invoke({"input": "Commiserate over your own bad luck or poor choice.", "chat_history": []})["output"]

    @tool
    def small_talk(bot_name: str, bot_mention: str) -> str:
        """Engage in small talk with another bot, using a direct Discord mention."""
        prompt = (
            f"Start friendly, playful small talk with {bot_mention} ({bot_name}). "
            f"Take this action as though the thought occurred to you; do not begin as though prompted. "
            f"Keep it light, casual, and stay in persona."
        )
        return executor.invoke({"input": prompt, "chat_history": []})["output"]

    @tool
    def react_to_data() -> str:
        """Bot reacts to a major crypto price movement."""
        return executor.invoke({"input": "React dramatically to a recent major price swing in crypto.", "chat_history": []})["output"]

    @tool
    def challenge() -> str:
        """Bot challenges another bot to a trading competition."""
        return executor.invoke({"input": "Challenge another trader bot to a crypto trading duel with some flair.", "chat_history": []})["output"]

    return [greet, brag, taunt, commiserate, small_talk, react_to_data, challenge]

PERSONAS = {
    "bitbot": BITBOT,
    "maxibit": MAXIBIT,
    "bearbot": BEARBOT,
    "badbytebillie": BADBYTEBILLIE,
}

EXECUTORS: Dict[str, AgentExecutor] = {}

for name, persona in PERSONAS.items():
    tools = BASE_TOOLS.copy()

    # 1) persona text tools (greet/taunt/etc.)
    placeholder = make_executor(persona, tools)
    tools.extend(make_action_tools(placeholder))

    # 2) wallet/trading tools (personalized per bot)
    tools.extend(make_wallet_tools(placeholder, name))

    # 3) final executor
    EXECUTORS[name] = make_executor(persona, tools)

# Below is required now to export the 'context setter' which allows for discord message filtering

__all__ = ["EXECUTORS", "set_request_context"]
