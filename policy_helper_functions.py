"""These are helper functions that support bot policy adjustment"""

import os, json
from datetime import datetime, timezone
from policies import BotPolicy

POLICY_DIR = os.getenv("POLICY_DIR", "policies")

def policy_path(bot_name: str) -> str:
    os.makedirs(POLICY_DIR, exist_ok=True)
    return os.path.join(POLICY_DIR, f"{bot_name.lower()}.json")

def load_policy(bot_name: str) -> BotPolicy:
    p = policy_path(bot_name)
    if not os.path.exists(p):
        return BotPolicy()
    with open(p, "r", encoding="utf-8") as f:
        data = json.load(f)
    return BotPolicy(**data)

def save_policy(bot_name: str, policy: BotPolicy, reason: str = "") -> None:
    obj = policy.model_dump()
    obj["updated_at"] = datetime.now(timezone.utc).isoformat()
    obj["updated_reason"] = (reason or obj.get("updated_reason") or "")[:280]
    with open(policy_path(bot_name), "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)
