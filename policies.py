"""This script defines the policies for each trader bot. These are the parameters that govern their
trading, that are called to govern each trade along with the API request and executor personality. 
These parameters will be adjusted based on bot persona and market behavior"""
from pydantic import BaseModel, Field, field_validator
from typing import List, Optional

class TokenBias(BaseModel):
    symbol: Optional[str] = None  # e.g., "ETH"
    id: Optional[str] = None      # e.g., "ethereum"
    weight: float = Field(0.5, ge=0, le=1)

class BotPolicy(BaseModel):
    risk_tolerance: float = Field(0.5, ge=0, le=1)
    target_cash_pct: float = Field(0.2, ge=0, le=1)
    max_positions: int = Field(5, ge=1, le=20)
    min_liquidity_usd: float = Field(5e6, ge=0)
    momentum_window_days: int = Field(30, ge=1, le=365)
    prefer_majors_weight: float = Field(0.5, ge=0, le=1)
    exploration_rate: float = Field(0.15, ge=0, le=1)
    stop_loss_bps: int = Field(150, ge=0, le=5000)
    take_profit_bps: int = Field(300, ge=0, le=10000)
    token_biases: List[TokenBias] = Field(default_factory=list)
    
    # Additional buy sell parameters
    min_m24_buy: float = Field(0.30, ge=-20, le=20)   # momentum threshold to BUY
    max_m24_sell: float = Field(-0.60, ge=-20, le=20) # momentum threshold to SELL/trim
    buy_usd: int = Field(150, ge=5, le=10_000)        # default buy size
    sell_frac: float = Field(0.10, ge=0, le=1)        # fraction to trim on sell signal
    min_trade_usd: int = Field(25, ge=1, le=500)      # floor to avoid dust
    
    # metadata (not modifiable by model)
    updated_at: Optional[str] = None
    updated_reason: Optional[str] = None

    @field_validator("token_biases")
    @classmethod
    def _normalize_ids(cls, v):
        # Enforce max 12 biases, normalize symbols upper
        out = []
        seen = set()
        for tb in v[:12]:
            sym = (tb.symbol or "").upper() or None
            cid = (tb.id or "").lower() or None
            key = sym or cid
            if key and key not in seen:
                seen.add(key)
                out.append(TokenBias(symbol=sym, id=cid, weight=float(tb.weight)))
        return out
