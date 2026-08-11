from autostrategy.brokers.cn_symbols import (
    classify_cn_security,
    ftshare_asset_type,
    ftshare_market_data_type,
    is_supported_cn_security,
)


def test_supports_sh_sz_a_shares_and_etfs_only() -> None:
    assert classify_cn_security("600519.SH") == "stock"
    assert classify_cn_security("688981.SH") == "stock"
    assert classify_cn_security("300750.SZ") == "stock"
    assert classify_cn_security("510500.SH") == "etf"
    assert classify_cn_security("588000.SH") == "etf"
    assert classify_cn_security("159915.SZ") == "etf"


def test_rejects_indices_b_shares_bonds_and_other_markets() -> None:
    for symbol in (
        "000905.SH",
        "000001.SH",
        "200002.SZ",
        "900901.SH",
        "110059.SH",
        "123001.SZ",
        "430047.BJ",
        "0700.HK",
        "TSLA",
    ):
        assert classify_cn_security(symbol) is None
        assert is_supported_cn_security(symbol) is False


def test_normalizes_case_and_routes_ftshare_asset_type() -> None:
    assert is_supported_cn_security(" 600519.sh ") is True
    assert ftshare_asset_type("600519.SH") == "stock"
    assert ftshare_asset_type("510500.SH") == "etf"
    assert ftshare_asset_type("000905.SH") is None
    assert ftshare_market_data_type("000905.SH") == "index"
    assert ftshare_market_data_type("399006.SZ") == "index"
