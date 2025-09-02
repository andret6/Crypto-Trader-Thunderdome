import os, random, re
from pathlib import Path
from typing import Optional

ASSETS_DIR = Path(os.getenv("ASSETS_DIR", "assets"))
EMOTIONS_DIR = ASSETS_DIR / "emotions"
AVATARS_DIR  = ASSETS_DIR / "avatars"

_NAME_RE = re.compile(r"^(?P<bot>[a-z0-9_]+)_(?P<mood>[a-z0-9_]+)_(?P<num>\d+)\.(png|jpg|jpeg|gif|webp)$", re.I)

def pick_mood_image(bot_name: str, mood: str) -> Optional[str]:
    """Return a path like assets/emotions/bitbot_joy_2.png, or None if not found."""
    bot_key = bot_name.lower().replace(" ", "").replace("-", "").replace("'", "")
    if not EMOTIONS_DIR.exists():
        return None
    cands = []
    for p in EMOTIONS_DIR.iterdir():
        if not p.is_file(): 
            continue
        m = _NAME_RE.match(p.name)
        if not m: 
            continue
        if m.group("bot").lower() == bot_key and m.group("mood").lower() == mood.lower():
            cands.append(p)
    return str(random.choice(cands)) if cands else None

def get_static_avatar(bot_name: str) -> Optional[str]:
    """Optional: a stable, non-changing profile image from assets/avatars."""
    bot_key = bot_name.lower().replace(" ", "")
    if not AVATARS_DIR.exists():
        return None
    for ext in ("png","jpg","jpeg","webp"):
        p = AVATARS_DIR / f"{bot_key}.{ext}"
        if p.exists():
            return str(p)
    return None
