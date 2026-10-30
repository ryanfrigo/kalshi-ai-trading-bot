"""trade.py legacy fallback must hand db_manager to XAIClient (spend mirror)."""
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.jobs import trade


@pytest.mark.asyncio
async def test_fallback_passes_db_manager_to_xai_client():
    db = MagicMock()
    db.get_eligible_markets = AsyncMock(return_value=[])
    with patch.object(trade, "DatabaseManager", return_value=db), \
         patch.object(trade, "KalshiClient"), \
         patch.object(trade, "XAIClient") as xai:
        await trade._fallback_legacy_trading()
    xai.assert_called_once_with(db_manager=db)
