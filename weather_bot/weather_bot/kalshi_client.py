# ============================================================
# kalshi_client.py  -  Minimal Kalshi REST wrapper
# Uses RSA-PSS auth as per Kalshi API docs
# Base: https://api.elections.kalshi.com/trade-api/v2
# ============================================================
import time
import base64
import json
import hashlib
import requests
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding
from cryptography.hazmat.backends import default_backend
from typing import Optional, Dict, Any
import config as cfg


class KalshiClient:
    """Lightweight read + paper-trade wrapper for Kalshi REST API."""

    BASE = cfg.KALSHI_BASE_URL

    def __init__(self):
        self._key_id = cfg.KALSHI_API_KEY_ID
        self._private_key = self._load_key(cfg.KALSHI_PRIVATE_KEY)
        self._session = requests.Session()
        self._session.headers.update({"Content-Type": "application/json"})

    # ------------------------------------------------------------------
    # Auth helpers
    # ------------------------------------------------------------------
    def _load_key(self, path: str):
        """Load RSA private key from PEM file."""
        try:
            with open(path, "rb") as f:
                return serialization.load_pem_private_key(
                    f.read(), password=None, backend=default_backend()
                )
        except Exception:
            return None  # paper-trade still works without key for public endpoints

    def _sign(self, ts_ms: str, method: str, path: str) -> str:
        """RSA-PSS sign the request string."""
        msg = ts_ms + method.upper() + path
        sig = self._private_key.sign(
            msg.encode(), padding.PSS(
                mgf=padding.MGF1(hashes.SHA256()),
                salt_length=padding.PSS.DIGEST_LENGTH,
            ), hashes.SHA256()
        )
        return base64.b64encode(sig).decode()

    def _headers(self, method: str, path: str) -> Dict:
        """Build auth headers for authenticated endpoints."""
        ts = str(int(time.time() * 1000))
        h = {
            "KALSHI-ACCESS-KEY": self._key_id,
            "KALSHI-ACCESS-TIMESTAMP": ts,
        }
        if self._private_key:
            h["KALSHI-ACCESS-SIGNATURE"] = self._sign(ts, method, path)
        return h

    # ------------------------------------------------------------------
    # Public market data (no auth needed)
    # ------------------------------------------------------------------
    def get_markets(self, series_ticker: Optional[str] = None,
                    status: str = "open", limit: int = 100) -> list:
        """GET /markets - returns list of market dicts."""
        params = {"status": status, "limit": limit}
        if series_ticker:
            params["series_ticker"] = series_ticker
        r = self._session.get(f"{self.BASE}/markets", params=params, timeout=10)
        r.raise_for_status()
        return r.json().get("markets", [])

    def get_orderbook(self, ticker: str, depth: int = 1) -> Dict:
        """GET /markets/{ticker}/orderbook - best bid/ask for YES/NO."""
        path = f"/trade-api/v2/markets/{ticker}/orderbook"
        r = self._session.get(
            f"{self.BASE}/markets/{ticker}/orderbook",
            params={"depth": depth}, timeout=10
        )
        r.raise_for_status()
        return r.json().get("orderbook", {})

    def get_market(self, ticker: str) -> Dict:
        """GET /markets/{ticker}"""
        r = self._session.get(f"{self.BASE}/markets/{ticker}", timeout=10)
        r.raise_for_status()
        return r.json().get("market", {})

    # ------------------------------------------------------------------
    # Authenticated portfolio endpoints
    # ------------------------------------------------------------------
    def get_balance(self) -> float:
        """GET /portfolio/balance - returns available cash in dollars."""
        path = "/trade-api/v2/portfolio/balance"
        r = self._session.get(
            f"{self.BASE}/portfolio/balance",
            headers=self._headers("GET", path), timeout=10
        )
        r.raise_for_status()
        data = r.json()
        return data.get("balance", 0) / 100  # Kalshi returns cents

    def get_positions(self) -> list:
        """GET /portfolio/positions"""
        path = "/trade-api/v2/portfolio/positions"
        r = self._session.get(
            f"{self.BASE}/portfolio/positions",
            headers=self._headers("GET", path), timeout=10
        )
        r.raise_for_status()
        return r.json().get("market_positions", [])

    def create_order(self, ticker: str, side: str, count: int,
                     yes_price: int, action: str = "buy") -> Dict:
        """
        POST /portfolio/orders
        ticker:    market ticker e.g. 'KXHIGHNYC-26MAR02-T65'
        side:      'yes' or 'no'
        count:     number of contracts (1 contract = $1 max payout)
        yes_price: limit price in cents (1-99)
        action:    'buy' or 'sell'
        """
        if cfg.PAPER_TRADE:
            # paper mode - don't actually send
            return {"paper": True, "ticker": ticker, "side": side,
                    "count": count, "yes_price": yes_price, "action": action}

        path = "/trade-api/v2/portfolio/orders"
        body = {
            "ticker": ticker,
            "side": side,
            "action": action,
            "type": "limit",
            "count": count,
            "yes_price": yes_price,
            "client_order_id": f"clwd_{int(time.time())}",
        }
        r = self._session.post(
            f"{self.BASE}/portfolio/orders",
            headers=self._headers("POST", path),
            json=body, timeout=10
        )
        r.raise_for_status()
        return r.json().get("order", {})

    # ------------------------------------------------------------------
    # Convenience: extract best bid/ask from orderbook
    # ------------------------------------------------------------------
    @staticmethod
    def parse_best_prices(orderbook: Dict) -> Dict:
        """
        Returns {'yes_ask': float, 'yes_bid': float,
                 'no_ask': float,  'no_bid': float}
        Prices are 0.0-1.0 (divided by 100 from Kalshi cents).
        """
        def best(side_list, which):
            if not side_list:
                return None
            # orderbook levels: [[price_cents, qty], ...]
            return side_list[0][0] / 100 if side_list else None

        yes_asks = orderbook.get("yes", [])
        yes_bids = orderbook.get("no", [])   # NO bids = YES asks complement
        return {
            "yes_ask": best(yes_asks, "ask"),
            "yes_bid": (1 - best(yes_bids, "bid")) if best(yes_bids, "bid") is not None else None,
            "no_ask":  best(yes_bids, "ask"),
            "no_bid":  (1 - best(yes_asks, "bid")) if best(yes_asks, "bid") is not None else None,
        }