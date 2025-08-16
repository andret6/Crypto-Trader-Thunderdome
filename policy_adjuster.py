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
    if "prefer_majors_weight" in out: out["prefer_majors_weight"] = clamp(out["prefer_majors_weight"], 0, 1)
    if "exploration_rate" in out:     out["exploration_rate"]     = clamp(out["exploration_rate"], 0, 1)
    if "stop_loss_bps" in out:        out["stop_loss_bps"]        = int(clamp(out["stop_loss_bps"], 0, 5000) or 0)
    if "take_profit_bps" in out:      out["take_profit_bps"]      = int(clamp(out["take_profit_bps"], 0, 10000) or 0)
    if "momentum_window_days" in out: out["momentum_window_days"] = int(clamp(out["momentum_window_days"], 1, 365) or 30)
    if "max_positions" in out:        out["max_positions"]        = int(clamp(out["max_positions"], 1, 20) or 8)
    if "min_liquidity_usd" in out:    out["min_liquidity_usd"]    = max(0.0, float(out["min_liquidity_usd"] or 0.0))
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
    current = load_policy(bot_name)

    def _clean_json_string(s: str) -> str:
        s = (s or "").strip()
        # strip code fences if present
        if s.startswith("```"):
            parts = s.split("```")
            if len(parts) >= 2:
                s = parts[1].strip()
        # if the model wrapped JSON in prose, try to extract the first {...} block
        if not s.startswith("{"):
            try:
                start = s.index("{"); end = s.rindex("}") + 1
                s = s[start:end]
            except Exception:
                pass
        return s

    def _format_messages(prev_error: str | None):
        sys = SYSTEM
        if prev_error:
            sys += (
                "\nIMPORTANT:\n"
                "- Your last output failed (see error below). Respond with VALID JSON ONLY.\n"
                "- Do NOT include markdown fences, comments, or prose.\n"
                f"- ERROR: {prev_error}\n"
            )
        return ChatPromptTemplate.from_messages([
            ("system", sys),
            ("user", f"""Persona:
{persona}

Current policy (JSON):
{current.model_dump(exclude={{"updated_at", "updated_reason"}})}

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
{SCHEMA_HINT}
""")
        ]).format_messages()

    MAX_RETRIES = 5
    prev_error = None
    for attempt in range(1, MAX_RETRIES + 1):
        # 1) ask LLM (with feedback if retry)
        msgs = _format_messages(prev_error)
        raw = _llm.invoke(msgs).content

        # 2) parse -> sanitize -> validate
        try:
            raw_s = _clean_json_string(raw)
            proposed = json.loads(raw_s)
        except Exception as e:
            prev_error = f"json decode error: {str(e)[:180]}"
            continue

        try:
            safe = _sanitize(proposed if isinstance(proposed, dict) else {})
        except Exception as e:
            prev_error = f"sanitize error: {str(e)[:180]}"
            continue

        try:
            updated = BotPolicy(**{
                **current.model_dump(exclude={"updated_at", "updated_reason"}),
                **safe
            })
        except ValidationError as e:
            # capture the first pydantic error for feedback
            first = e.errors()[0] if e.errors() else {"msg": str(e)}
            prev_error = f"validation error: {first.get('msg','unknown')[:180]}"
            continue
        except Exception as e:
            prev_error = f"unexpected validate error: {str(e)[:180]}"
            continue

        # 3) success → save and return (only successful writes update reason/timestamp)
        reason = (proposed.get("updated_reason") if isinstance(proposed, dict) else None) or "self-tune"
        reason = reason.strip()[:280] if isinstance(reason, str) else "self-tune"
        save_policy(bot_name, updated, reason=reason if reason else "self-tune")
        return updated

    # All retries failed → keep current policy, do not overwrite reason/timestamp
    print(f"[{bot_name}] policy update failed after {MAX_RETRIES} attempts: {prev_error}")
    return current
