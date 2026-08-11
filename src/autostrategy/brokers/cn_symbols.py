"""Classification helpers for tradable Shanghai/Shenzhen securities."""

from __future__ import annotations

from typing import Literal

CnSecurityType = Literal["stock", "etf"]
CnMarketDataType = Literal["stock", "etf", "index"]

_SH_A_SHARE_PREFIXES = ("600", "601", "603", "605", "688", "689")
_SZ_A_SHARE_PREFIXES = ("000", "001", "002", "003", "300", "301")

# Shanghai ETF allocation uses selected 51x/52x/56x/58x blocks. Keep this
# explicit so index (000xxx), bond (10xxxx/11xxxx) and legacy warrant blocks
# cannot pass a broad numeric-range check. Shenzhen exchange-traded funds use
# the 159xxx block.
_SH_ETF_PREFIXES = (
    "510",
    "511",
    "512",
    "513",
    "515",
    "516",
    "517",
    "518",
    "520",
    "560",
    "561",
    "562",
    "563",
    "588",
    "589",
)


def classify_cn_security(symbol: str) -> CnSecurityType | None:
    """Return the supported FTShare asset type for an exchange-qualified code."""
    normalized = str(symbol).strip().upper()
    if "." not in normalized:
        return None
    code, exchange = normalized.rsplit(".", 1)
    if len(code) != 6 or not code.isdigit():
        return None
    if exchange == "SH":
        if code.startswith(_SH_A_SHARE_PREFIXES):
            return "stock"
        if code.startswith(_SH_ETF_PREFIXES):
            return "etf"
        return None
    if exchange == "SZ":
        if code.startswith(_SZ_A_SHARE_PREFIXES):
            return "stock"
        if code.startswith("159"):
            return "etf"
    return None


def is_supported_cn_security(symbol: str) -> bool:
    """Return whether a symbol can be used for FT client simulation orders."""
    return classify_cn_security(symbol) is not None


def ftshare_asset_type(symbol: str) -> CnSecurityType | None:
    """Return the FTShare daily-bar type for a supported security."""
    return classify_cn_security(symbol)


def ftshare_market_data_type(symbol: str) -> CnMarketDataType | None:
    """Return an FTShare type for tradable securities or mainland benchmarks."""
    security_type = classify_cn_security(symbol)
    if security_type is not None:
        return security_type
    normalized = str(symbol).strip().upper()
    if "." not in normalized:
        return None
    code, exchange = normalized.rsplit(".", 1)
    if len(code) != 6 or not code.isdigit():
        return None
    if (exchange == "SH" and code.startswith("000")) or (
        exchange == "SZ" and code.startswith("399")
    ):
        return "index"
    return None
