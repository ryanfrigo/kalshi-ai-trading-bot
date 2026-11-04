"""KalshiClient's request path over httpx.MockTransport: an empty 2xx body, and the
get_markets query string. A throwaway RSA key per test; no real key, no network.
"""
import base64
from urllib.parse import parse_qs, urlsplit

import httpx
import pytest
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding, rsa

from src.clients import kalshi_client as kc
from src.clients.kalshi_client import KalshiClient

DUMMY_KEY_ID = "test-dummy-key-id"  # secret-scan:allow
PSS = padding.PSS(mgf=padding.MGF1(hashes.SHA256()), salt_length=padding.PSS.DIGEST_LENGTH)


@pytest.fixture
async def rig(tmp_path, monkeypatch):
    async def no_sleep(_):
        return None
    monkeypatch.setattr(kc.asyncio, "sleep", no_sleep)
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    path = tmp_path / "throwaway.pem"
    path.write_bytes(key.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8,
                                       serialization.NoEncryption()))
    client = KalshiClient(api_key=DUMMY_KEY_ID, private_key_path=str(path))
    seen = []
    replies = {"status": 200, "body": b'{"markets": []}'}

    def handler(request):
        seen.append(request)
        return httpx.Response(replies["status"], content=replies["body"])

    await client.client.aclose()
    client.client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    yield client, seen, replies, key
    await client.client.aclose()


@pytest.mark.parametrize("status, body", [(200, b""), (201, b""), (201, b"  \n")])
async def test_an_empty_success_body_is_a_result_not_a_json_error(rig, status, body):
    client, _, replies, _ = rig
    replies.update(status=status, body=body)
    assert await client._make_authenticated_request("POST", "/x", json_data={"a": 1}) == {
        "status_code": status, "empty_body": True}


async def test_a_json_body_is_parsed_not_reported_as_empty(rig):
    client, _, replies, _ = rig
    replies["body"] = b'{"ok": 1}'
    assert await client._make_authenticated_request("GET", "/x") == {"ok": 1}


async def test_get_markets_sends_only_the_limit_by_default(rig):
    client, seen, _, _ = rig
    assert await client.get_markets(limit=7) == {"markets": []}
    assert parse_qs(urlsplit(str(seen[0].url)).query) == {"limit": ["7"]}


async def test_get_markets_sends_every_filter_and_joins_tickers_with_commas(rig):
    client, seen, _, _ = rig
    await client.get_markets(limit=5, cursor="c1", event_ticker="E", series_ticker="S",
                             status="open", tickers=["T1", "T2"])
    assert parse_qs(urlsplit(str(seen[0].url)).query) == {
        "limit": ["5"], "cursor": ["c1"], "event_ticker": ["E"], "series_ticker": ["S"],
        "status": ["open"], "tickers": ["T1,T2"]}


async def test_get_markets_signs_the_path_without_the_query_string(rig):
    client, seen, _, key = rig
    await client.get_markets(limit=5, cursor="c1")
    headers = seen[0].headers
    assert urlsplit(str(seen[0].url)).path == "/trade-api/v2/markets"
    assert headers["KALSHI-ACCESS-KEY"] == DUMMY_KEY_ID
    key.public_key().verify(base64.b64decode(headers["KALSHI-ACCESS-SIGNATURE"]),
                            (headers["KALSHI-ACCESS-TIMESTAMP"] + "GET/trade-api/v2/markets").encode(),
                            PSS, hashes.SHA256())
