# policy_adjuster.py
from typing import Dict, Any
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from policies import BotPolicy
from policy_helper_functions import load_policy, save_policy

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
  "token_biases": "list of {symbol?:str,id?:str,weight:0..1}",
  "updated_reason": "str <= 280 chars"
}

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
    import json
    try:
        proposed = json.loads(raw)
    except Exception:
        # fallback: keep current, stamp reason
        save_policy(bot_name, current, reason="parse_error")
        return current

    try:
        updated = BotPolicy(**{**current.model_dump(exclude={"updated_at","updated_reason"}), **proposed})
    except Exception:
        # invalid fields -> keep current
        save_policy(bot_name, current, reason="validation_error")
        return current

    # 4) save and return
    reason = (proposed.get("updated_reason") or "self-tune") if isinstance(proposed, dict) else "self-tune"
    save_policy(bot_name, updated, reason=reason)
    return updated
