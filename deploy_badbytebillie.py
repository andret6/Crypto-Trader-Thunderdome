""" This is the bot script for deploying B^3 to Discord. 
Note that this just activates her, you still have to manually add her to servers
Note that here we deploy one bot per script so that we can more easily manage individual
bot tokens on the Discord side.
"""

import asyncio
import random
import os
import discord
from avatar_helper_functions import pick_mood_image
from agents import EXECUTORS, set_request_context
import re
from policy_adjuster import adjust_policy
from policy_helper_functions import load_policy

TOKEN = os.getenv("DISCORD_BOT_TOKEN_4")                
AGENT_NAME = os.getenv("AGENT_NAME", "badbytebillie")        

if not TOKEN:
    raise ValueError("DISCORD_BOT_TOKEN not set")
if AGENT_NAME not in EXECUTORS:
    raise ValueError(f"Unknown AGENT_NAME '{AGENT_NAME}'. Choose from: {list(EXECUTORS)}")

def is_trade_broadcast(text: str) -> bool:
    return bool(re.search(r'^\[PAPER\]\s', text)) or "USD bal:" in text or "trade_log" in text

executor = EXECUTORS[AGENT_NAME]

intents = discord.Intents.default()
intents.message_content = True
intents.members = True  # Enable for on_member_join
client = discord.Client(intents=intents)

chat_histories = {}
BOT_TARGETS = {}

def is_mentioned(me: discord.ClientUser, message: discord.Message, chance_without_mention: float = 0.23) -> bool:
    # Always respond if mentioned
    if any(user.id == me.id for user in message.mentions):
        return True
    # Otherwise, respond with specified probability
    return random.random() < chance_without_mention

async def send_to_default_channel(content: str):
    for guild in client.guilds:
        for channel in guild.text_channels:
            if channel.permissions_for(guild.me).send_messages:
                await channel.send(content)
                return

@client.event
async def on_ready():
    print(f"[{AGENT_NAME}] Logged in as {client.user}")

    BOT_TARGETS[AGENT_NAME] = []
    for guild in client.guilds:
        for member in guild.members:
            if member.bot and member != client.user:
                BOT_TARGETS[AGENT_NAME].append((member.display_name, member.mention))
    
    # print(f"[{AGENT_NAME}] Target bots found: {BOT_TARGETS[AGENT_NAME]}")

    response = EXECUTORS[AGENT_NAME].invoke({
        "input": "Greet the channel",
        "chat_history": []
    })
    await send_to_default_channel(response["output"])


@client.event
async def on_member_join(member):
    if member.bot and member != client.user:
        BOT_TARGETS.setdefault(AGENT_NAME, []).append((member.display_name, member.mention))
    
    if not member.bot:
        response = EXECUTORS[AGENT_NAME].invoke({
            "input": f"A new user named {member.name} has joined. Greet them!",
            "chat_history": []
        })
        await send_to_default_channel(response["output"])

@client.event
async def on_member_remove(member):
    if member.bot and member != client.user:
        BOT_TARGETS[AGENT_NAME] = [
            (name, mention) for name, mention in BOT_TARGETS.get(AGENT_NAME, [])
            if mention != member.mention
        ]

@client.event
async def on_message(message: discord.Message):
    # More bot message filtering
    if message.author == client.user:
      return

    # Decide allowance BEFORE invoking the agent
    mentioned = any(user.id == client.user.id for user in message.mentions)
    from_human = not message.author.bot
    looks_like_trade_post = is_trade_broadcast(message.content)

    allow_tools = from_human and mentioned and not looks_like_trade_post
    set_request_context(source=("human" if from_human else "bot"), allow_tools=allow_tools)

    if not is_mentioned(client.user, message):
        return

    content = message.content
    for user in message.mentions:
        content = content.replace(f"<@{user.id}>", "").replace(f"<@!{user.id}>", "")
    
    history = chat_histories.setdefault(message.channel.id, [])
    
    if not content.strip():
        # Let the agent generate a persona-appropriate response to being pinged with no content
        result = await asyncio.to_thread(
            executor.invoke,
            {"input": "Respond in character to someone pinging you without saying anything.","chat_history": history}
        )
        await message.channel.send(result["output"][:1900])
        return

    # Show typing while we run the blocking agent call in a thread
    async with message.channel.typing():
        result = await asyncio.to_thread(executor.invoke, {"input": content, "chat_history": history})

    reply = result["output"]
    history.append(("human", content))
    history.append(("assistant", reply))
    
    # Include emotion based images in replies
    tool = next((t for t in EXECUTORS[AGENT_NAME].tools if getattr(t, "name", None) == "pnl_mood"), None)
    mood = None
    if tool:
        try:
            mood = tool.invoke({})
        except Exception:
            mood = None
    if not mood:
        # cheap fallback if tool missing/errored
        txt = reply.lower()
        mood = "joy" if any(x in txt for x in [" up ", "+", "gains", "win", "winner"]) else \
               "sad" if any(x in txt for x in [" down ", "-", "loss", "loser"]) else \
               "neutral"
    
    img_path = pick_mood_image(AGENT_NAME, mood)
    
    if img_path:
        file = discord.File(img_path, filename=os.path.basename(img_path))
        embed = discord.Embed(description=reply[:1900])
        embed.set_image(url=f"attachment://{os.path.basename(img_path)}")
        await message.channel.send(embed=embed, file=file)
    else:
        await message.channel.send(reply[:1900])
    


## Create random bot small talk

async def small_talk_loop():
    await client.wait_until_ready()
    while not client.is_closed():
        count = random.randint(0, 3)
        delay = 3600 // (count + 1) if count > 0 else 3600
        for _ in range(count):
            await asyncio.sleep(random.randint(300, delay))

            targets = BOT_TARGETS.get(AGENT_NAME, [])
            if targets:
                target_name, target_mention = random.choice(targets)
                tool = next(
                    (t for t in EXECUTORS[AGENT_NAME].tools if getattr(t, "name", None) == "small_talk"),
                    None
                )
                if tool:
                    response = tool.invoke({"bot_name": target_name, "bot_mention": target_mention})
                    await send_to_default_channel(response)
                else:
                    print(f"⚠️ 'small_talk' tool not found for agent {AGENT_NAME}")
        await asyncio.sleep(delay)

@client.event
async def on_error(event, *args, **kwargs):
    import traceback
    print(f"⚠️ Unhandled error in event: {event}")
    traceback.print_exc()

## Perform auto trades while in the channel
async def send_to_default_channel(content: str):
    for guild in client.guilds:
        for channel in guild.text_channels:
            if channel.permissions_for(guild.me).send_messages:
                await channel.send(content)
                return

# Define additional parameters for adjusting bot behavior in the autotrade loop
BOT_NAME = AGENT_NAME
PERSONA = "You are a digitized persona of pop star billie elyish meets april from parks and rec, with hyper intelligence, dry wit, but razor sharp intellect. You re inclinded to give eye rolls than respond to messages. You are extremely risk tolerant as a crypto trader, but not stupid."

def _pnl_summary_from_wallet(bot_name: str) -> str:
    # tiny helper: summarize last few trades
    from agent_helper_functions import _load_wallet, _pretty_money
    w = _load_wallet(bot_name)
    tl = list(w.get("trade_log", []))[-5:]
    if not tl:
        return "no trades yet"
    parts = []
    for t in tl:
        parts.append(f"{t['side']} {t['symbol']} {t['qty']:.4f} @ {_pretty_money(t['px'])}")
    return " | ".join(parts)

def _market_snapshot_compact() -> dict:
    # small, self-contained snapshot using the same CG endpoint as agents
    from agent_helper_functions import _request_with_retry, CG_BASE
    params = {
        "vs_currency": "usd",
        "order": "market_cap_desc",
        "per_page": 20,
        "page": 1,
        "price_change_percentage": "1h,24h,7d",
    }
    data = _request_with_retry(f"{CG_BASE}/coins/markets", params)
    # keep just a few fields to keep the prompt small
    return [
        {
            "id": r.get("id"),
            "sym": (r.get("symbol") or "").upper(),
            "price": r.get("current_price"),
            "chg_24h": r.get("price_change_percentage_24h_in_currency"),
            "vol": r.get("total_volume"),
            "mcap": r.get("market_cap"),
        } for r in data
    ]

async def autotrade_loop():
    await client.wait_until_ready()
    tools = EXECUTORS[AGENT_NAME].tools
    def _tool(name):
        return next((t for t in tools if getattr(t, "name", "") == name), None)

    get_ivl  = _tool("get_autotrade_interval")
    auto_once = _tool("auto_trade_once")
    brag_cope = _tool("brag_or_commiserate")

    while not client.is_closed():
        try:
            # --- self-tune policy each cycle (updates policies/{bot}.json) ---
            market = _market_snapshot_compact()
            pnl = _pnl_summary_from_wallet(BOT_NAME)
            adjust_policy(BOT_NAME, PERSONA, market, pnl)

            # --- run one auto trade (current tool still uses persona policy) ---
            set_request_context(source="scheduler", allow_tools=True)
            if auto_once:
                result = auto_once.invoke({})
                await send_to_default_channel(result if isinstance(result, str) else str(result))

            # --- optional brag/cope ---
            set_request_context(source="scheduler", allow_tools=False)
            if brag_cope and random.random() < 0.33:
                note = brag_cope.invoke({})
                await send_to_default_channel(note if isinstance(note, str) else str(note))

            # --- persona-based interval with small jitter ---
            ivl = get_ivl.invoke({}) if get_ivl else 3600
            sleep_s = max(60, int(float(ivl) * random.uniform(0.9, 1.1)))

        except Exception as e:
            print(f"[autotrade_loop] error: {e}")
            sleep_s = 90  # brief backoff

        await asyncio.sleep(sleep_s)

@client.event
async def setup_hook():
    client.loop.create_task(small_talk_loop())
    client.loop.create_task(autotrade_loop())

client.run(TOKEN)
