from __future__ import annotations

import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import pytest

from autostrategy.brokers.ft_client import FtClientError
from autostrategy.brokers.models import (
    AccountHealth,
    BrokerEvent,
    BrokerOrder,
    FtAccount,
    FundSnapshot,
    MonitoringMetric,
    MonitoringSnapshot,
    PositionSnapshot,
)
from autostrategy.config import FtClientConfig
from autostrategy.core.strategy import StrategyStatus
from autostrategy.services.client_simulation_service import (
    ClientSimulationError,
    ClientSimulationRequest,
    ClientSimulationService,
    FtshareDailyMarketContextProvider,
    FtshareTenMinuteMarketContextProvider,
)
from autostrategy.services.strategy_service import StrategyService


class FakeBroker:
    def __init__(
        self,
        *,
        account_login: bool = True,
        engine_ready: bool = True,
        submit_error: FtClientError | None = None,
        reconcile_orders: list[BrokerOrder] | None = None,
    ) -> None:
        self.account_login = account_login
        self.engine_ready = engine_ready
        self.submit_error = submit_error
        self.reconcile_orders = reconcile_orders or []
        self.submit_calls = 0
        self.cancelled: list[str] = []
        self.intents = []
        self.get_orders_calls = 0

    def connect(self):
        return [
            FtAccount(
                ft_account="ft-user",
                ft_account_name="FT 模拟",
                broker_id="1",
                broker_name="测试券商",
                trade_account="SIM001",
                nickname="模拟账户",
                login_status=self.account_login,
            )
        ]

    def list_accounts(self):
        return self.connect()

    def disconnect(self):
        return None

    def health(self):
        return {
            "SIM001": AccountHealth(
                trade_account="SIM001",
                login_status=self.account_login,
                order_engine_status=self.engine_ready,
            )
        }

    def get_funds(self, trade_account: str, broker_id: str):
        return FundSnapshot(
            trade_account=trade_account,
            balance=1_000_000,
            asset=1_000_000,
            available=800_000,
            risk_equity=1_000_000,
        )

    def get_positions(self, trade_account: str, broker_id: str):
        return [
            PositionSnapshot(
                trade_account=trade_account,
                stock_code="588000.SH",
                total_volume=1_000,
                available_volume=1_000,
                raw={"market_value": 1_000},
            )
        ]

    def submit_order(self, intent):
        self.submit_calls += 1
        self.intents.append(intent)
        if self.submit_error:
            raise self.submit_error
        return BrokerOrder(
            parent_order_id=str(40 + self.submit_calls),
            external_id=intent.intent_id,
            trade_account=intent.trade_account,
            basket_name=intent.session_id,
            stock_code=intent.broker_symbol,
            order_volume=intent.quantity,
            raw_status=1,
            normalized_status="working",
        )

    def cancel_orders(self, parent_order_ids: list[str]):
        self.cancelled.extend(parent_order_ids)

    def get_orders(self, trade_account: str, broker_id: str):
        self.get_orders_calls += 1
        return self.reconcile_orders

    def get_child_orders(self, parent_order_id: str):
        return []

    def get_monitoring(self, ft_account: str):
        return MonitoringSnapshot(
            ft_account=ft_account,
            trade_accounts=[
                MonitoringMetric(
                    account_id=ft_account,
                    trade_account="SIM001",
                    total_rate=0.5,
                    exposure=0.1,
                )
            ],
        )


def _ft_config(**overrides) -> FtClientConfig:
    values = {
        "enabled": True,
        "confirmed_client_version": "3.11.4",
        "allowed_simulation_accounts": ["SIM001"],
        "allowed_symbols": ["588000.SH", "563300.SH"],
        "symbol_mapping": {
            "588000.SH": "588000.SH",
            "563300.SH": "563300.SH",
        },
        "allowed_algorithms": ["TWAP"],
        "external_id_max_length": 64,
        "external_id_scope_confirmed": True,
    }
    values.update(overrides)
    return FtClientConfig(**values)


def _write_strategy(root: Path, *, quantity: int = 100, side: str = "buy") -> str:
    strategy_service = StrategyService(workspace_root=root)
    strategy = strategy_service.create_strategy("Grid Demo")
    strategy_dir = root / strategy.slug
    (strategy_dir / "strategy.py").write_text(
        "def run_backtest(config):\n"
        "    return {'annual_return': 1, 'max_drawdown': 1, 'sharpe': 1, "
        "'win_rate': 50, 'profit_loss_ratio': 1, 'total_trades': 1}\n\n"
        "def generate_intents(context):\n"
        "    return {\n"
        "        'decisions': [{'symbol': '588000.SH', 'action': '"
        + side
        + "', 'signal_price': 1.0, 'reason': 'grid'}],\n"
        "        'intents': [{\n"
        "            'intent_key': '588000.SH:2026-08-10:grid:1:"
        + side
        + "',\n"
        "            'symbol': '588000.SH', 'side': '"
        + side
        + "',\n"
        f"            'quantity': {quantity}, 'signal_price': 1.0, 'reason': 'grid',\n"
        "            'execution_window': {'start': '093500', 'end': '145000'},\n"
        "        }],\n"
        "        'strategy_state': {'last_grid': 1},\n"
        "    }\n",
        encoding="utf-8",
    )
    (strategy_dir / "config.yaml").write_text(
        "market: A股\nsymbols: ['588000.SH']\n", encoding="utf-8"
    )
    strategy_service.workspace.update_strategy_status(strategy.slug, StrategyStatus.BACKTESTED)
    return strategy.slug


def _write_daily_market_strategy(root: Path) -> str:
    strategy_service = StrategyService(workspace_root=root)
    strategy = strategy_service.create_strategy("Daily Market Demo")
    strategy_dir = root / strategy.slug
    (strategy_dir / "strategy.py").write_text(
        "def generate_intents(context):\n"
        "    completed = context['market']['completed_bar_at']\n"
        "    return {\n"
        "        'intents': [{\n"
        "            'intent_key': f'588000.SH:{completed}:daily:buy',\n"
        "            'symbol': '588000.SH', 'side': 'buy',\n"
        "            'quantity': 100, 'signal_price': 1.0, 'reason': 'daily',\n"
        "        }],\n"
        "        'strategy_state': {'last_completed_bar_at': completed},\n"
        "    }\n",
        encoding="utf-8",
    )
    (strategy_dir / "config.yaml").write_text(
        "market: A股\nsymbols: ['588000.SH']\n", encoding="utf-8"
    )
    strategy_service.workspace.update_strategy_status(strategy.slug, StrategyStatus.BACKTESTED)
    return strategy.slug


def _request(mode: str = "observe") -> ClientSimulationRequest:
    return ClientSimulationRequest(
        trade_account="SIM001",
        execution_mode=mode,
        acknowledge_simulation=True,
    )


def test_preflight_blocks_unverified_client_and_failed_account_login(tmp_path: Path) -> None:
    slug = _write_strategy(tmp_path)
    broker = FakeBroker(account_login=False)
    service = ClientSimulationService(
        workspace_root=tmp_path,
        config=_ft_config(confirmed_client_version=None),
        broker_factory=lambda _: broker,
    )

    result = service.preflight(slug, _request())

    failed_codes = {item.code for item in result.checks if not item.passed}
    assert "client_version_unverified" in failed_codes
    assert "account_login_failed" in failed_codes
    assert result.ready is False


def test_preflight_accepts_a_share_and_etf_execution_universe(tmp_path: Path) -> None:
    slug = _write_strategy(tmp_path)
    (tmp_path / slug / "config.yaml").write_text(
        "market: A股\nsymbols: ['600519.SH', '510500.SH', '300750.SZ']\n",
        encoding="utf-8",
    )
    config = _ft_config(
        allowed_symbols=["600519.SH", "510500.SH", "300750.SZ"],
        symbol_mapping={
            "600519.SH": "600519.SH",
            "510500.SH": "510500.SH",
            "300750.SZ": "300750.SZ",
        },
    )
    service = ClientSimulationService(
        workspace_root=tmp_path,
        config=config,
        broker_factory=lambda _: FakeBroker(),
    )

    result = service.preflight(slug, _request())

    assert result.ready is True


@pytest.mark.parametrize("symbol", ["000905.SH", "200002.SZ", "110059.SH", "430047.BJ"])
def test_preflight_rejects_unsupported_execution_market(
    tmp_path: Path, symbol: str
) -> None:
    slug = _write_strategy(tmp_path)
    (tmp_path / slug / "config.yaml").write_text(
        f"market: A股\nsymbols: ['{symbol}']\n", encoding="utf-8"
    )
    service = ClientSimulationService(
        workspace_root=tmp_path,
        config=_ft_config(
            allowed_symbols=[symbol],
            symbol_mapping={symbol: symbol},
        ),
        broker_factory=lambda _: FakeBroker(),
    )

    result = service.preflight(slug, _request())

    failed_codes = {item.code for item in result.checks if not item.passed}
    assert "unsupported_market_symbol" in failed_codes
    assert result.ready is False


def test_preflight_requires_mapping_for_entire_execution_universe(tmp_path: Path) -> None:
    slug = _write_strategy(tmp_path)
    service = ClientSimulationService(
        workspace_root=tmp_path,
        config=_ft_config(
            allowed_symbols=["588000.SH", "600519.SH"],
            symbol_mapping={"588000.SH": "588000.SH"},
        ),
        broker_factory=lambda _: FakeBroker(),
    )

    result = service.preflight(slug, _request())

    failed_codes = {item.code for item in result.checks if not item.passed}
    assert "symbol_mapping_confirmed" in failed_codes
    assert result.ready is False


def test_preflight_loads_benchmark_data_without_allowing_index_orders(tmp_path: Path) -> None:
    slug = _write_strategy(tmp_path)
    (tmp_path / slug / "config.yaml").write_text(
        "market: A股\nsymbols: ['510500.SH']\nbenchmark: '000905.SH'\n",
        encoding="utf-8",
    )
    requested_symbols: list[str] = []

    def market_context(symbols, _config):
        requested_symbols.extend(symbols)
        bars = {
            symbol: {"at": "2026-08-10", "close": 1.0}
            for symbol in symbols
        }
        return {
            "bars_by_symbol": bars,
            "history_by_symbol": {symbol: [bar] for symbol, bar in bars.items()},
            "completed_bar_at": "2026-08-10",
        }

    service = ClientSimulationService(
        workspace_root=tmp_path,
        config=_ft_config(
            allowed_symbols=["510500.SH"],
            symbol_mapping={"510500.SH": "510500.SH"},
        ),
        broker_factory=lambda _: FakeBroker(),
        market_context_provider=market_context,
    )

    result = service.preflight(slug, _request())

    assert result.ready is True
    assert requested_symbols == ["510500.SH", "000905.SH"]


def test_background_evaluation_keeps_benchmark_market_data(tmp_path: Path) -> None:
    slug = _write_strategy(tmp_path)
    (tmp_path / slug / "config.yaml").write_text(
        "market: A股\nsymbols: ['510500.SH']\nbenchmark: '000905.SH'\n",
        encoding="utf-8",
    )
    requested: list[list[str]] = []

    def market_context(symbols, _config):
        requested.append(list(symbols))
        completed = (
            "2026-08-11T09:40:00+08:00"
            if len(requested) == 1
            else "2026-08-11T09:50:00+08:00"
        )
        bars = {
            symbol: {"at": completed, "symbol": symbol, "close": 1.0}
            for symbol in symbols
        }
        return {
            "bars_by_symbol": bars,
            "history_by_symbol": {symbol: [bar] for symbol, bar in bars.items()},
            "completed_bar_at": completed,
        }

    service = ClientSimulationService(
        workspace_root=tmp_path,
        config=_ft_config(
            allowed_symbols=["510500.SH"],
            symbol_mapping={"510500.SH": "510500.SH"},
        ),
        broker_factory=lambda _: FakeBroker(),
        market_context_provider=market_context,
        clock=lambda: datetime(2026, 8, 11, 10, 0, tzinfo=ZoneInfo("Asia/Shanghai")),
    )

    session = service.create_session(slug, _request())
    service.evaluate_latest_bar(slug, session.session_id)

    assert requested == [
        ["510500.SH", "000905.SH"],
        ["510500.SH", "000905.SH"],
    ]


def test_observe_session_persists_intents_without_submission(tmp_path: Path) -> None:
    slug = _write_strategy(tmp_path)
    broker = FakeBroker()
    service = ClientSimulationService(
        workspace_root=tmp_path,
        config=_ft_config(),
        broker_factory=lambda _: broker,
    )

    session = service.create_session(slug, _request("observe"))

    assert session.status == "running"
    assert session.account_login_status is True
    assert session.order_engine_status is True
    assert session.intents[0].status == "validated"
    assert broker.submit_calls == 0
    session_dir = tmp_path / slug / "paper_run" / "client_sessions" / session.session_id
    assert (session_dir / "session.json").exists()
    assert (session_dir / "order_intents.jsonl").exists()
    assert (session_dir / "account_snapshots.jsonl").exists()
    assert (session_dir / "monitoring_snapshots.jsonl").exists()


def test_manual_session_submits_only_after_approval(tmp_path: Path) -> None:
    slug = _write_strategy(tmp_path)
    broker = FakeBroker()
    service = ClientSimulationService(
        workspace_root=tmp_path,
        config=_ft_config(),
        broker_factory=lambda _: broker,
    )
    session = service.create_session(slug, _request("manual"))

    assert broker.submit_calls == 0
    approved = service.approve_intent(slug, session.session_id, session.intents[0].intent_id)

    assert broker.submit_calls == 1
    assert approved.orders[0].external_id == session.intents[0].intent_id


def test_auto_session_deduplicates_same_intent_across_sessions(tmp_path: Path) -> None:
    slug = _write_strategy(tmp_path)
    broker = FakeBroker()
    service = ClientSimulationService(
        workspace_root=tmp_path,
        config=_ft_config(),
        broker_factory=lambda _: broker,
    )

    first = service.create_session(slug, _request("auto"))
    service.stop_session(slug, first.session_id)
    second = service.create_session(slug, _request("auto"))

    assert broker.submit_calls == 1
    assert first.intents[0].intent_id == second.intents[0].intent_id
    assert second.intents[0].status == "rejected"
    assert second.intents[0].reason == "duplicate_intent"


def test_submission_unknown_reconciles_by_external_id_without_retry(tmp_path: Path) -> None:
    slug = _write_strategy(tmp_path)
    broker = FakeBroker(
        submit_error=FtClientError("client_unavailable", "timeout", retryable=True)
    )
    service = ClientSimulationService(
        workspace_root=tmp_path,
        config=_ft_config(),
        broker_factory=lambda _: broker,
    )
    expected_intent_id = service.compute_intent_id(
        slug, 1, "SIM001", "588000.SH:2026-08-10:grid:1:buy"
    )
    broker.reconcile_orders = [
        BrokerOrder(
            parent_order_id="88",
            external_id=expected_intent_id,
            trade_account="SIM001",
            basket_name="unknown-response",
            raw_status=1,
            normalized_status="working",
        )
    ]

    session = service.create_session(slug, _request("auto"))

    assert broker.submit_calls == 1
    assert session.orders[0].parent_order_id == "88"
    assert session.orders[0].external_id == expected_intent_id


def test_risk_rejects_non_lot_quantity(tmp_path: Path) -> None:
    slug = _write_strategy(tmp_path, quantity=150)
    broker = FakeBroker()
    service = ClientSimulationService(
        workspace_root=tmp_path,
        config=_ft_config(),
        broker_factory=lambda _: broker,
    )

    session = service.create_session(slug, _request("auto"))

    assert session.intents[0].status == "rejected"
    assert session.intents[0].reason == "quantity_not_board_lot"
    assert broker.submit_calls == 0


def test_stop_cancels_only_session_parent_orders(tmp_path: Path) -> None:
    slug = _write_strategy(tmp_path)
    broker = FakeBroker()
    service = ClientSimulationService(
        workspace_root=tmp_path,
        config=_ft_config(),
        broker_factory=lambda _: broker,
    )
    session = service.create_session(slug, _request("auto"))

    stopped = service.stop_session(slug, session.session_id)

    assert stopped.status == "stopped"
    assert broker.cancelled == [session.orders[0].parent_order_id]


def test_service_loads_persisted_sessions_after_restart(tmp_path: Path) -> None:
    slug = _write_strategy(tmp_path)
    broker = FakeBroker()
    first_service = ClientSimulationService(
        workspace_root=tmp_path,
        config=_ft_config(),
        broker_factory=lambda _: broker,
    )
    created = first_service.create_session(slug, _request("observe"))

    restarted = ClientSimulationService(
        workspace_root=tmp_path,
        config=_ft_config(),
        broker_factory=lambda _: FakeBroker(),
    )
    loaded = restarted.get_session(slug, created.session_id)

    assert loaded.session_id == created.session_id
    assert loaded.status == "paused"


def test_create_session_requires_generate_intents(tmp_path: Path) -> None:
    strategy_service = StrategyService(workspace_root=tmp_path)
    strategy = strategy_service.create_strategy("Legacy")
    strategy_dir = tmp_path / strategy.slug
    (strategy_dir / "strategy.py").write_text("def run_backtest(config):\n    return {}\n")
    strategy_service.workspace.update_strategy_status(strategy.slug, StrategyStatus.BACKTESTED)
    service = ClientSimulationService(
        workspace_root=tmp_path,
        config=_ft_config(),
        broker_factory=lambda _: FakeBroker(),
    )

    with pytest.raises(ClientSimulationError) as caught:
        service.create_session(strategy.slug, _request())

    assert caught.value.code == "strategy_incompatible"


def test_background_worker_polls_active_session_and_stops_cleanly(tmp_path: Path) -> None:
    slug = _write_strategy(tmp_path)
    broker = FakeBroker()
    service = ClientSimulationService(
        workspace_root=tmp_path,
        config=_ft_config(),
        broker_factory=lambda _: broker,
        enable_background_reconciliation=True,
        background_interval_seconds=0.01,
    )

    session = service.create_session(slug, _request("observe"))
    deadline = time.monotonic() + 1
    while time.monotonic() < deadline and broker.get_orders_calls == 0:
        time.sleep(0.01)

    assert broker.get_orders_calls > 0
    service.stop_session(slug, session.session_id)
    calls_after_stop = broker.get_orders_calls
    time.sleep(0.05)
    assert broker.get_orders_calls == calls_after_stop
    service.shutdown()


def test_websocket_event_is_persisted_and_triggers_reconciliation(tmp_path: Path) -> None:
    slug = _write_strategy(tmp_path)

    class StreamingBroker(FakeBroker):
        async def stream_events(self):
            yield BrokerEvent(topic="Mudan", data={"id": 88, "external_id": "event-id"})

    broker = StreamingBroker()
    service = ClientSimulationService(
        workspace_root=tmp_path,
        config=_ft_config(),
        broker_factory=lambda _: broker,
        enable_background_reconciliation=True,
        background_interval_seconds=0.5,
    )
    session = service.create_session(slug, _request("observe"))
    deadline = time.monotonic() + 1
    events = service.get_events(slug, session.session_id)
    while time.monotonic() < deadline and (
        not any(item["type"] == "broker_event" for item in events)
        or broker.get_orders_calls == 0
    ):
        time.sleep(0.01)
        events = service.get_events(slug, session.session_id)

    assert any(item["type"] == "broker_event" for item in events)
    assert broker.get_orders_calls > 0
    service.stop_session(slug, session.session_id)
    service.shutdown()


def test_new_completed_daily_bar_generates_intents_exactly_once(tmp_path: Path) -> None:
    slug = _write_daily_market_strategy(tmp_path)
    broker = FakeBroker()
    completed_bars = iter(["2026-08-10", "2026-08-11", "2026-08-11"])

    def market_context(symbols, config):
        completed = next(completed_bars)
        bar = {
            "at": completed,
            "symbol": symbols[0],
            "open": 1.0,
            "high": 1.1,
            "low": 0.9,
            "close": 1.0,
            "volume": 1000,
        }
        return {
            "bars_by_symbol": {symbols[0]: bar},
            "history_by_symbol": {symbols[0]: [bar]},
            "completed_bar_at": completed,
        }

    service = ClientSimulationService(
        workspace_root=tmp_path,
        config=_ft_config(),
        broker_factory=lambda _: broker,
        market_context_provider=market_context,
        clock=lambda: datetime(2026, 8, 12, 10, 0, tzinfo=ZoneInfo("Asia/Shanghai")),
    )

    session = service.create_session(slug, _request("auto"))
    advanced = service.evaluate_latest_bar(slug, session.session_id)
    unchanged = service.evaluate_latest_bar(slug, session.session_id)

    assert session.last_evaluated_bar_at == "2026-08-10"
    assert advanced.last_evaluated_bar_at == "2026-08-11"
    assert len(advanced.intents) == 2
    assert broker.submit_calls == 2
    assert len(unchanged.intents) == 2


def test_auto_intent_waits_for_execution_window(tmp_path: Path) -> None:
    slug = _write_daily_market_strategy(tmp_path)
    broker = FakeBroker()
    now = [datetime(2026, 8, 12, 9, 0, tzinfo=ZoneInfo("Asia/Shanghai"))]

    def market_context(symbols, config):
        bar = {
            "at": "2026-08-11",
            "symbol": symbols[0],
            "open": 1.0,
            "high": 1.1,
            "low": 0.9,
            "close": 1.0,
            "volume": 1000,
        }
        return {
            "bars_by_symbol": {symbols[0]: bar},
            "history_by_symbol": {symbols[0]: [bar]},
            "completed_bar_at": "2026-08-11",
        }

    service = ClientSimulationService(
        workspace_root=tmp_path,
        config=_ft_config(),
        broker_factory=lambda _: broker,
        market_context_provider=market_context,
        clock=lambda: now[0],
    )

    session = service.create_session(slug, _request("auto"))
    assert broker.submit_calls == 0
    assert session.intents[0].status == "validated"

    now[0] = datetime(2026, 8, 12, 9, 35, tzinfo=ZoneInfo("Asia/Shanghai"))
    evaluated = service.evaluate_latest_bar(slug, session.session_id)

    assert broker.submit_calls == 1
    assert evaluated.intents[0].status == "working"


def test_background_worker_evaluates_new_completed_daily_bar(tmp_path: Path) -> None:
    slug = _write_daily_market_strategy(tmp_path)
    broker = FakeBroker()
    calls = [0]

    def market_context(symbols, config):
        calls[0] += 1
        completed = "2026-08-10" if calls[0] == 1 else "2026-08-11"
        bar = {
            "at": completed,
            "symbol": symbols[0],
            "open": 1.0,
            "high": 1.1,
            "low": 0.9,
            "close": 1.0,
            "volume": 1000,
        }
        return {
            "bars_by_symbol": {symbols[0]: bar},
            "history_by_symbol": {symbols[0]: [bar]},
            "completed_bar_at": completed,
        }

    service = ClientSimulationService(
        workspace_root=tmp_path,
        config=_ft_config(),
        broker_factory=lambda _: broker,
        market_context_provider=market_context,
        clock=lambda: datetime(2026, 8, 12, 10, 0, tzinfo=ZoneInfo("Asia/Shanghai")),
        enable_background_reconciliation=True,
        background_interval_seconds=0.01,
        market_poll_interval_seconds=0.01,
    )
    session = service.create_session(slug, _request("auto"))
    deadline = time.monotonic() + 1
    while time.monotonic() < deadline and broker.submit_calls < 2:
        time.sleep(0.01)

    assert broker.submit_calls == 2
    assert service.get_session(slug, session.session_id).last_evaluated_bar_at == "2026-08-11"
    service.stop_session(slug, session.session_id)
    service.shutdown()


def test_ftshare_market_provider_uses_previous_completed_day_and_common_bar() -> None:
    calls = []

    def fetcher(symbol, **kwargs):
        calls.append((symbol, kwargs))
        dates = ["2026-08-10", "2026-08-11"] if symbol == "588000.SH" else ["2026-08-10"]
        return pd.DataFrame(
            {
                "open": [1.0] * len(dates),
                "high": [1.1] * len(dates),
                "low": [0.9] * len(dates),
                "close": [1.0] * len(dates),
                "volume": [1000] * len(dates),
            },
            index=pd.to_datetime(dates),
        )

    provider = FtshareDailyMarketContextProvider(
        fetcher=fetcher,
        clock=lambda: datetime(2026, 8, 12, 10, 0, tzinfo=ZoneInfo("Asia/Shanghai")),
    )

    context = provider(["588000.SH", "563300.SH"], {})

    assert context["completed_bar_at"] == "2026-08-10"
    assert context["bars_by_symbol"]["588000.SH"]["at"] == "2026-08-10"
    assert all(call[1]["end_date"] == "2026-08-11" for call in calls)


def test_ftshare_market_provider_routes_stock_and_etf_types() -> None:
    calls: dict[str, str] = {}

    def fetcher(symbol, **kwargs):
        calls[symbol] = kwargs["type_"]
        return pd.DataFrame(
            {
                "open": [1.0],
                "high": [1.1],
                "low": [0.9],
                "close": [1.0],
                "volume": [1000],
            },
            index=pd.to_datetime(["2026-08-10"]),
        )

    provider = FtshareDailyMarketContextProvider(
        fetcher=fetcher,
        clock=lambda: datetime(2026, 8, 12, 10, 0, tzinfo=ZoneInfo("Asia/Shanghai")),
    )

    provider(["600519.SH", "510500.SH"], {})

    assert calls == {"600519.SH": "stock", "510500.SH": "etf"}


def test_ten_minute_provider_uses_latest_common_bar_and_keeps_daily_history() -> None:
    minute_rows = {
        "510500.SH": [
            {"tm": "2026-08-11T09:30:00", "p": 10.0, "v": 100},
            {"tm": "2026-08-11T09:39:00", "p": 10.1, "v": 100},
            {"tm": "2026-08-11T09:40:00", "p": 10.2, "v": 100},
            {"tm": "2026-08-11T09:49:00", "p": 10.3, "v": 100},
        ],
        "000905.SH": [
            {"tm": "2026-08-11T09:30:00", "p": 6000.0, "v": 100},
            {"tm": "2026-08-11T09:39:00", "p": 6001.0, "v": 100},
        ],
    }
    requested_types: dict[str, str] = {}

    def minute_fetcher(symbol, **kwargs):
        requested_types[symbol] = kwargs["type_"]
        return minute_rows[symbol]

    def daily_fetcher(symbol, **_kwargs):
        return pd.DataFrame(
            {
                "open": [1.0, 2.0],
                "high": [1.1, 2.1],
                "low": [0.9, 1.9],
                "close": [1.0, 2.0],
                "volume": [1000, 2000],
            },
            index=pd.to_datetime(["2026-08-07", "2026-08-10"]),
        )

    provider = FtshareTenMinuteMarketContextProvider(
        minute_fetcher=minute_fetcher,
        daily_fetcher=daily_fetcher,
        clock=lambda: datetime(2026, 8, 11, 10, 0, tzinfo=ZoneInfo("Asia/Shanghai")),
    )

    context = provider(["510500.SH", "000905.SH"], {})

    assert context["completed_bar_at"] == "2026-08-11T09:40:00+08:00"
    assert set(context["bars_by_symbol"]) == {"510500.SH", "000905.SH"}
    assert context["bars_by_symbol"]["510500.SH"]["close"] == 10.1
    assert len(context["intraday_history_by_symbol"]["510500.SH"]) == 1
    assert context["history_by_symbol"]["000905.SH"][-1]["at"] == "2026-08-10"
    assert requested_types == {"510500.SH": "etf", "000905.SH": "index"}
