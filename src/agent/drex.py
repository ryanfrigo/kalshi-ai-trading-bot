"""
Drex (Nace's decision model) as the forecaster for ``verify --drex``.

Drex serves the same System One request shape as Jev (a ``state`` of facts plus
typed ``questions``), so ``build_jev_request`` is reused unchanged; only the
endpoint, key and model differ. If Drex is unavailable or errors, the floor falls
back to Jev, so ``--drex`` is never looser than ``--jev``.
Docs: https://console.nace.ai/docs/drex/api-reference/systemone
"""

from __future__ import annotations

import logging
import os
from importlib.metadata import PackageNotFoundError, version
from typing import Any, Dict, Optional

from src.agent.jev import jev_yes_pct, parse_jev_yes_pct

DREX_URL = "https://console.nace.ai/v1/systemone"
DREX_MODEL = "drex-max-finance"
# Sent as X-Nace-Client and User-Agent so Nace can attribute Drex traffic to this repo.
try:
    CLIENT = f"kalshi-ai-trading-bot/{version('kalshi-ai-trading-bot')}"
except PackageNotFoundError:  # a clone that skipped `pip install -e .`
    CLIENT = "kalshi-ai-trading-bot/unknown"

logger = logging.getLogger(__name__)


async def drex_yes_pct(request: Dict[str, Any], record: Optional[Dict[str, Any]] = None) -> float:
    """POST one System One request to Drex and return P(YES) percent; Jev answers if Drex fails.

    ``record`` (optional dict) receives ``drex_model`` only when Drex answered.
    """
    import httpx

    try:
        key = os.getenv("NACE_API_KEY")
        if not key:
            raise RuntimeError("NACE_API_KEY not set")
        async with httpx.AsyncClient(timeout=60) as http:
            r = await http.post(
                DREX_URL,
                headers={
                    "Authorization": f"Bearer {key}",
                    "X-Nace-Client": CLIENT,
                    "User-Agent": CLIENT,
                },
                json={**request, "model": DREX_MODEL},
            )
        if r.status_code != 200:
            raise RuntimeError(f"Drex HTTP {r.status_code}: {r.text[:300]}")
        pct = parse_jev_yes_pct(r.json(), "Drex")
    except (RuntimeError, ValueError, httpx.HTTPError) as e:
        logger.warning("Drex forecast failed (%s); falling back to Jev", e)
        return await jev_yes_pct(request)
    if record is not None:
        record["drex_model"] = DREX_MODEL
    return pct
