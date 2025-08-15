# policy_adjuster.py
from typing import Dict, Any
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from policies import BotPolicy
from policy_helper_functions import load_policy, save_policy
from pydantic import ValidationError
import json

_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)

SYSTEM = """You are a trading policy tuner for a specific bot.
Only output JSON matching the provided schema. Do not include prose.
You may tweak parameters within reasonable bounds to fit persona and market.
"""

TEMPLATE = ChatPromptTemplate.from_messages([
    ("system", SYSTEM),
    ("user", """Persona:
{persona}

Current policy (JSON):
{current_policy}

Recent PnL (last N trades):
{pnl_summary}

Market snapshot (top coins with % change, volume, dominance hints):
{market_snapshot}

Instructions:
- Update parameters to reflect persona and market.
- Keep values within human-like ranges; avoid extreme flips unless justified.
- Prefer sparse changes; include a short 'updated_reason'.
- Output JSON ONLY with the same keys as current_policy; omit metadata fields if present.

JSON schema (keys/types, not a validator):
{schema_hint}
""")
])

SCHEMA_HINT = {
  "risk_tolerance": "float 0..1",
  "target_cash_pct": "float 0..1",
  "max_positions": "int 1..20",
  "min_liquidity_usd": "float >=0",
  "momentum_window_days": "int 1..365",
  "prefer_majors_weight": "float 0..1",
  "exploration_rate": "float 0..1",
  "stop_loss_bps": "int 0..5000",
  "take_profit_bps": "int 0..10000",
  "token_biases": "list of {symbol?:str,id?:str,weight:0..1}",  # <- fixed
  "min_m24_buy": "float -20..20",
  "max_m24_sell": "float -20..20",
  "buy_usd": "int 5..10000",
  "sell_frac": "float 0..1",
  "min_trade_usd": "int 1..500",
  "updated_reason": "str <= 280 chars"
}

ALLOWED = set(BotPolicy.model_fields.keys()) - {"updated_at", "updated_reason"}

def _sanitize(proposed: dict) -> dict:
    out = {k: v for k, v in (proposed or {}).items() if k in ALLOWED}
    def clamp(x, lo, hi):
        try: return max(lo, min(hi, float(x)))
        except: return None

    if "risk_tolerance" in out:   out["risk_tolerance"]   = clamp(out["risk_tolerance"], 0, 1)
    if "target_cash_pct" in out:  out["target_cash_pct"]  = clamp(out["target_cash_pct"], 0, 1)
    if "sell_frac" in out:        out["sell_frac"]        = clamp(out["sell_frac"], 0, 1)
    if "buy_usd" in out:          out["buy_usd"]          = int(clamp(out["buy_usd"], 5, 10_000) or 150)
    if "min_trade_usd" in out:    out["min_trade_usd"]    = int(clamp(out["min_trade_usd"], 1, 500) or 25)
    if "min_m24_buy" in out:      out["min_m24_buy"]      = clamp(out["min_m24_buy"], -20, 20)
    if "max_m24_sell" in out:     out["max_m24_sell"]     = clamp(out["max_m24_sell"], -20, 20)
    if "token_biases" in out and isinstance(out["token_biases"], list):
        cleaned = []
        seen = set()
        for tb in out["token_biases"][:12]:
            if not isinstance(tb, dict): continue
            sym = (tb.get("symbol") or "").upper() or None
            cid = (tb.get("id") or "").lower() or None
            key = sym or cid
            if not key or key in seen: continue
            seen.add(key)
            w = clamp(tb.get("weight", 0.5), 0, 1)
            if w is not None:
                cleaned.append({"symbol": sym, "id": cid, "weight": float(w)})
        out["token_biases"] = cleaned
    return out

def adjust_policy(bot_name: str, persona: str, market_snapshot: Dict[str, Any], pnl_summary: str) -> BotPolicy:
    # 1) load current
    current = load_policy(bot_name)
    # 2) ask LLM for updated JSON
    msg = TEMPLATE.format_messages(
        persona=persona,
        current_policy=current.model_dump(exclude={"updated_at", "updated_reason"}),
        pnl_summary=pnl_summary,
        market_snapshot=market_snapshot,
        schema_hint=SCHEMA_HINT,
    )
    raw = _llm.invoke(msg).content
    # 3) parse+validate (clamp by Pydantic)
    try:
        proposed = json.loads(raw)
        safe = _sanitize(proposed if isinstance(proposed, dict) else {})
    except Exception:
        # fallback: keep current, stamp reason
        save_policy(bot_name, current, reason="parse_error")
        return current

    try:
        updated = BotPolicy(**{**current.model_dump(exclude={"updated_at","updated_reason"}), **safe})
    except Exception:
        # invalid fields -> keep current
        save_policy(bot_name, current, reason="validation_error")
        return current

    # 4) save and return
    reason = (proposed.get("updated_reason") or "self-tune") if isinstance(proposed, dict) else "self-tune"
    save_policy(bot_name, updated, reason=reason)
    return updated
