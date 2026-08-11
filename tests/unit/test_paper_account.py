"""Unit tests for the Phase 5C virtual paper account model."""

from __future__ import annotations

from autostrategy.core.paper_account import PaperAccount


def make_account(cash: float = 1_000_000, commission: float = 0.0) -> PaperAccount:
    return PaperAccount(cash=cash, initial_cash=cash, commission=commission)


def test_from_config_defaults():
    account = PaperAccount.from_config({})
    assert account.cash == 1_000_000
    assert account.initial_cash == 1_000_000
    assert account.commission == 0.0


def test_from_config_reads_initial_cash_and_commission():
    account = PaperAccount.from_config({"initial_cash": 500_000, "commission": 0.001})
    assert account.cash == 500_000
    assert account.commission == 0.001


def test_buy_fills_and_reduces_cash():
    account = make_account(100_000)
    event = account.apply({"action": "buy", "symbol": "000001.SZ", "price": 10, "size": 1000})
    assert event["status"] == "filled"
    assert account.cash == 90_000
    assert account.positions["000001.SZ"].quantity == 1000
    assert account.positions["000001.SZ"].avg_price == 10
    assert account.trade_count == 1
    assert event["cash_after"] == 90_000
    assert event["position_after"] == 1000


def test_buy_rejected_on_insufficient_cash():
    account = make_account(5_000)
    event = account.apply({"action": "buy", "symbol": "000001.SZ", "price": 10, "size": 1000})
    assert event["status"] == "rejected"
    assert event["reject_reason"] == "insufficient_cash"
    assert account.cash == 5_000
    assert "000001.SZ" not in account.positions


def test_sell_fills_and_realizes_pnl():
    account = make_account(100_000)
    account.apply({"action": "buy", "symbol": "000001.SZ", "price": 10, "size": 1000})
    event = account.apply({"action": "sell", "symbol": "000001.SZ", "price": 12, "size": 1000})
    assert event["status"] == "filled"
    assert account.cash == 102_000
    assert account.realized_pnl == 2_000
    assert "000001.SZ" not in account.positions
    assert event["position_after"] == 0


def test_sell_rejected_on_insufficient_position():
    account = make_account(100_000)
    account.apply({"action": "buy", "symbol": "000001.SZ", "price": 10, "size": 100})
    event = account.apply({"action": "sell", "symbol": "000001.SZ", "price": 10, "size": 200})
    assert event["status"] == "rejected"
    assert event["reject_reason"] == "insufficient_position"
    assert account.positions["000001.SZ"].quantity == 100


def test_sell_rejected_without_position():
    account = make_account(100_000)
    event = account.apply({"action": "sell", "symbol": "000001.SZ", "price": 10, "size": 100})
    assert event["status"] == "rejected"
    assert account.cash == 100_000


def test_hold_records_without_mutation():
    account = make_account(100_000)
    event = account.apply({"action": "hold", "symbol": "000001.SZ", "price": 10})
    assert event["status"] == "held"
    assert account.cash == 100_000
    assert account.trade_count == 0


def test_unknown_action_treated_as_hold():
    account = make_account(100_000)
    event = account.apply({"action": "wait", "symbol": "000001.SZ", "price": 10})
    assert event["action"] == "hold"
    assert event["status"] == "held"


def test_commission_applies_to_buy_and_sell():
    account = make_account(100_000, commission=0.001)
    account.apply({"action": "buy", "symbol": "000001.SZ", "price": 10, "size": 1000})
    # cost = 10 * 1000 * 1.001 = 10010
    assert account.cash == 89_990
    account.apply({"action": "sell", "symbol": "000001.SZ", "price": 10, "size": 1000})
    # proceeds = 10 * 1000 * 0.999 = 9990
    assert account.cash == 99_980


def test_partial_sell_keeps_position_and_avg_price():
    account = make_account(100_000)
    account.apply({"action": "buy", "symbol": "000001.SZ", "price": 10, "size": 1000})
    account.apply({"action": "sell", "symbol": "000001.SZ", "price": 12, "size": 400})
    position = account.positions["000001.SZ"]
    assert position.quantity == 600
    assert position.avg_price == 10


def test_average_price_updates_on_additional_buy():
    account = make_account(100_000)
    account.apply({"action": "buy", "symbol": "000001.SZ", "price": 10, "size": 1000})
    account.apply({"action": "buy", "symbol": "000001.SZ", "price": 20, "size": 1000})
    assert account.positions["000001.SZ"].avg_price == 15


def test_equity_marks_to_market():
    account = make_account(100_000)
    account.apply({"action": "buy", "symbol": "000001.SZ", "price": 10, "size": 1000})
    account.mark_price("000001.SZ", 12)
    # cash 90000 + 1000 * 12 = 102000
    assert account.equity == 102_000
    assert account.unrealized_pnl == 2_000


def test_snapshot_structure():
    account = make_account(100_000)
    account.apply({"action": "buy", "symbol": "000001.SZ", "price": 10, "size": 1000})
    snapshot = account.snapshot()
    assert snapshot["initial_cash"] == 100_000
    assert snapshot["cash"] == 90_000
    assert snapshot["equity"] == 100_000
    assert snapshot["trade_count"] == 1
    assert snapshot["position_count"] == 1
    assert snapshot["positions"][0]["symbol"] == "000001.SZ"
    assert snapshot["positions"][0]["quantity"] == 1000
    assert snapshot["positions"][0]["market_value"] == 10_000


def test_multi_symbol_positions():
    account = make_account(100_000)
    account.apply({"action": "buy", "symbol": "000001.SZ", "price": 10, "size": 100})
    account.apply({"action": "buy", "symbol": "600519.SH", "price": 100, "size": 10})
    assert len(account.positions) == 2
    assert account.cash == 100_000 - 1_000 - 1_000
