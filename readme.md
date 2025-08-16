# Crypto Trader Thunderdome (PoC)

This proof-of-concept deploys **four LangChain-powered trader bots** plus a **Referee** into a private Discord server. The bots compete to grow a simulated $10,000 USD wallet, answer crypto questions, and talk a little smack. Prices and market stats come from CoinGecko Pro. The Referee posts periodic leaderboards. Visit the dashboard [here](https://crypto-bot-thunderdome-proof-of-concept.streamlit.app/).

## How it works
- Bots make simulated trades and answer crypto questions and route to CoinGecko Pro tools for current price, history windows, and market stats.

- Paper trades are filled at current spot with configurable fee/slippage bps.

- Bots adjust their own risk tolerance and other trading parameters based on their personas, by adjusting policies set in `policies.py` using tools defined in `policy_adjuster`.

- Referee aggregates wallet JSONs and posts a ranked report every `REFEREE_INTERVAL_SECS`.

- Each bot has its own deploy script so tokens can be managed per-process. Trader bots are defined in `agents.py` while the referee is defined and deployed to discord in `define_deploy_referee.py`

```
python deploy_j_bit_bot.py       # Bitbot  (uses DISCORD_BOT_TOKEN)
python deploy_maxibit_bot.py     # Maxibit (uses DISCORD_BOT_TOKEN_2)
python deploy_bear_bot.py        # Bearbot (uses DISCORD_BOT_TOKEN_3)
python deploy_badbytebillie.py   # B^3     (uses DISCORD_BOT_TOKEN_4)
python define_deploy_referee.py  # Referee (uses DISCORD_BOT_TOKEN_5)
```

Default chat model is `gpt-4o-mini` via LangChain’s ChatOpenAI. You can change it in `agents.py`.

Wallet state is file-backed under ./wallets (auto-created). Add wallets/ to .gitignore, or don't, the money isn't real who cares.

Policies are filed-back under ./policies (auto-created).

## Bots
- **Bitbot** (our favorite wall street madman) 
- **Maxibit** (cyborg ice-cream bitcoin maxi)
- **Bearbot** (risk-averse bear - get it?)
- **BadByteBillie** (deadpan risk-tolerant)
- **Referee** announcer.

## Requirements
`pip install -U langchain langchain-openai langchain-community discord.py python-dotenv requests pydantic pandas`

For the web app specifically:

`pip install -U streamlit plotly requests`

### Python
- Python 3.10+ recommended

### Environment Variables
Set these (e.g., in a `.env` file):

```env
# OpenAI / LangChain
OPENAI_API_KEY=...

# CoinGecko Pro
COINGECKO_API_KEY=...

# Discord bot tokens (one per app)
DISCORD_BOT_TOKEN=...        # Bitbot
DISCORD_BOT_TOKEN_2=...      # Maxibit
DISCORD_BOT_TOKEN_3=...      # Bearbot
DISCORD_BOT_TOKEN_4=...      # BadByteBillie
DISCORD_BOT_TOKEN_5=...      # Referee

# Optional tweaks
PAPER_TRADE_FEE_BPS=10       # default 10 bps
PAPER_TRADE_SLIPPAGE_BPS=5   # default 5 bps
REFEREE_INTERVAL_SECS=900    # default 900s
WALLET_DIR=./wallets         # default ./wallets
