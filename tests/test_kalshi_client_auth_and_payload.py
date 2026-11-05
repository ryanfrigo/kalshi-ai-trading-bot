"""KalshiClient request signing, key loading and v2 order-payload validation.

A throwaway RSA key is generated per test in tmp_path; no real key is ever read.
No network: the httpx client is built but never used, and is closed after each test.
"""
import base64

import pytest
from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding, rsa

from src.clients.kalshi_client import KalshiAPIError, KalshiClient, build_order_v2_payload

DUMMY_KEY_ID = "test-dummy-key-id"  # secret-scan:allow
PSS = padding.PSS(mgf=padding.MGF1(hashes.SHA256()), salt_length=padding.PSS.DIGEST_LENGTH)


def _key_file(tmp_path):
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    p = tmp_path / "throwaway.pem"
    p.write_bytes(key.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8,
                                    serialization.NoEncryption()))
    return key, str(p)


def test_missing_key_file_raises_kalshi_api_error(tmp_path):
    with pytest.raises(KalshiAPIError, match="Private key file not found"):
        KalshiClient(api_key=DUMMY_KEY_ID, private_key_path=str(tmp_path / "absent.pem"))


def test_unparseable_key_file_raises_kalshi_api_error(tmp_path):
    p = tmp_path / "junk.pem"
    p.write_bytes(b"not a pem key")
    with pytest.raises(KalshiAPIError, match="Failed to load private key"):
        KalshiClient(api_key=DUMMY_KEY_ID, private_key_path=str(p))


async def test_signature_is_rsa_pss_sha256_over_timestamp_upper_method_path(tmp_path):
    key, path = _key_file(tmp_path)
    c = KalshiClient(api_key=DUMMY_KEY_ID, private_key_path=path)
    try:
        sig = c._sign_request("1700000000000", "get", "/trade-api/v2/portfolio/balance")
        key.public_key().verify(base64.b64decode(sig),
                                b"1700000000000GET/trade-api/v2/portfolio/balance", PSS, hashes.SHA256())
    finally:
        await c.client.aclose()


async def test_signature_does_not_verify_for_another_path(tmp_path):
    key, path = _key_file(tmp_path)
    c = KalshiClient(api_key=DUMMY_KEY_ID, private_key_path=path)
    try:
        sig = c._sign_request("1700000000000", "GET", "/trade-api/v2/portfolio/balance")
        with pytest.raises(InvalidSignature):
            key.public_key().verify(base64.b64decode(sig),
                                    b"1700000000000GET/trade-api/v2/portfolio/orders", PSS, hashes.SHA256())
    finally:
        await c.client.aclose()


async def test_sign_failure_raises_kalshi_api_error(tmp_path):
    _, path = _key_file(tmp_path)
    c = KalshiClient(api_key=DUMMY_KEY_ID, private_key_path=path)
    try:
        c.private_key = None
        with pytest.raises(KalshiAPIError, match="Failed to sign request"):
            c._sign_request("1", "GET", "/x")
    finally:
        await c.client.aclose()


@pytest.mark.parametrize("kw, msg", [
    (dict(side="yes", action="hold", yes_price=50), "action must be"),
    (dict(side="maybe", action="buy", yes_price=50), "side must be"),
    (dict(side="no", action="buy"), "no_price required"),
    (dict(side="yes", action="buy"), "a price"),
])
def test_bad_order_spec_is_refused(kw, msg):
    with pytest.raises(ValueError, match=msg):
        build_order_v2_payload("MKT", "c", count=1, type_="limit", **kw)


def test_side_and_action_are_case_insensitive():
    p = build_order_v2_payload("MKT", "c", side="NO", action="BUY", count=1, type_="limit", no_price=96)
    assert p["side"] == "ask" and p["price"] == "0.0400"


def test_reduce_only_forces_ioc_and_drops_expiration():
    p = build_order_v2_payload("MKT", "c", side="yes", action="sell", count=2, type_="limit",
                               yes_price=40, expiration_ts=1800000000, reduce_only=True)
    assert p["time_in_force"] == "immediate_or_cancel"
    assert p["reduce_only"] is True
    assert "expiration_time" not in p


def test_gtc_limit_keeps_expiration_and_omits_reduce_only():
    p = build_order_v2_payload("MKT", "c", side="yes", action="buy", count=2, type_="limit",
                               yes_price=40, expiration_ts=1800000000)
    assert p["time_in_force"] == "good_till_canceled"
    assert p["expiration_time"] == 1800000000
    assert "reduce_only" not in p
