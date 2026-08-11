from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from autostrategy.api.app import create_app
from autostrategy.brokers.models import (
    AccountHealth,
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
    ClientSimulationService,
    FtshareTenMinuteMarketContextProvider,
)
from autostrategy.services.strategy_service import StrategyService


class ApiFakeBroker:
    def __init__(self) -> None:
        self.submit_calls = 0
        self.cancelled: list[str] = []

    def connect(self):
        return [
            FtAccount(
                ft_account="ft-user",
                ft_account_name="FT 模拟",
                broker_id="1",
                broker_name="测试券商",
                trade_account="SIM001",
                nickname="模拟账户",
                login_status=True,
            ),
            FtAccount(
                ft_account="ft-user",
                broker_id="2",
                trade_account="LIVE001",
                nickname="非白名单账户",
                login_status=True,
            ),
        ]

    def list_accounts(self):
        return self.connect()

    def disconnect(self):
        return None

    def health(self):
        return {
            "SIM001": AccountHealth(
                trade_account="SIM001", login_status=True, order_engine_status=True
            )
        }

    def get_funds(self, trade_account: str, broker_id: str):
        return FundSnapshot(
            trade_account=trade_account,
            asset=1_000_000,
            balance=1_000_000,
            available=800_000,
            risk_equity=1_000_000,
        )

    def get_positions(self, trade_account: str, broker_id: str):
        return [
            PositionSnapshot(
                trade_account=trade_account,
                stock_code="588000.SH",
                total_volume=1000,
                available_volume=1000,
                raw={"market_value": 1000},
            )
        ]

    def submit_order(self, intent):
        self.submit_calls += 1
        return BrokerOrder(
            parent_order_id="99",
            external_id=intent.intent_id,
            trade_account=intent.trade_account,
            basket_name=intent.session_id,
            raw_status=1,
            normalized_status="working",
        )

    def cancel_orders(self, parent_order_ids: list[str]):
        self.cancelled.extend(parent_order_ids)

    def get_orders(self, trade_account: str, broker_id: str):
        return []

    def get_child_orders(self, parent_order_id: str):
        return []

    def get_monitoring(self, ft_account: str):
        return MonitoringSnapshot(
            ft_account=ft_account,
            trade_accounts=[
                MonitoringMetric(
                    account_id=ft_account,
                    trade_account="SIM001",
                    total_rate=0.25,
                    exposure=0.2,
                )
            ],
        )


def _create_strategy(root: Path) -> str:
    strategy_service = StrategyService(workspace_root=root)
    strategy = strategy_service.create_strategy("FT API Demo")
    strategy_dir = root / strategy.slug
    (strategy_dir / "strategy.py").write_text(
        "def run_backtest(config):\n    return {}\n\n"
        "def generate_intents(context):\n"
        "    return {\n"
        "      'decisions': [],\n"
        "      'intents': [{\n"
        "        'intent_key': '588000.SH:20260810:buy',\n"
        "        'symbol': '588000.SH', 'side': 'buy', 'quantity': 100,\n"
        "        'signal_price': 1.0, 'reason': 'test',\n"
        "        'execution_window': {'start': '093500', 'end': '145000'}\n"
        "      }],\n"
        "      'strategy_state': {}\n"
        "    }\n",
        encoding="utf-8",
    )
    (strategy_dir / "config.yaml").write_text(
        "market: A股\nsymbols: ['588000.SH']\n", encoding="utf-8"
    )
    strategy_service.workspace.update_strategy_status(strategy.slug, StrategyStatus.BACKTESTED)
    return strategy.slug


def _client(tmp_path: Path) -> tuple[TestClient, str, ApiFakeBroker]:
    slug = _create_strategy(tmp_path)
    broker = ApiFakeBroker()
    config = FtClientConfig(
        enabled=True,
        confirmed_client_version="3.11.4",
        allowed_simulation_accounts=["SIM001"],
        allowed_symbols=["588000.SH", "563300.SH"],
        symbol_mapping={"588000.SH": "588000.SH", "563300.SH": "563300.SH"},
        external_id_max_length=64,
        external_id_scope_confirmed=True,
    )
    app = create_app(workspace_root=tmp_path)
    app.state.client_simulation_service = ClientSimulationService(
        workspace_root=tmp_path,
        config=config,
        broker_factory=lambda _: broker,
    )
    return TestClient(app), slug, broker


def _payload(mode: str = "observe") -> dict:
    return {
        "trade_account": "SIM001",
        "execution_mode": mode,
        "acknowledge_simulation": True,
        "execution_route": "algorithm_parent",
        "algorithm": {
            "strategy_type": "TWAP",
            "params": {},
            "reach_limit_continue": False,
            "over_time_continue": False,
        },
    }


def _connection_payload() -> dict:
    return {
        "base_url": "http://127.0.0.1:11356",
        "ft_account": "ft-user",
        "password": "one-time-secret",
        "password_transform": "plain",
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


def test_default_app_wires_ftshare_ten_minute_market_context(tmp_path: Path) -> None:
    app = create_app(workspace_root=tmp_path)

    assert isinstance(
        app.state.client_simulation_service._market_context_provider,
        FtshareTenMinuteMarketContextProvider,
    )


def test_connection_check_and_accounts_never_return_credentials(tmp_path: Path) -> None:
    client, _, _ = _client(tmp_path)

    checked = client.post(
        "/api/v1/broker-connections/ft-client/check", json=_connection_payload()
    )
    accounts = client.get("/api/v1/broker-connections/ft-client/accounts")

    assert checked.status_code == 200
    assert accounts.status_code == 200
    assert [item["trade_account"] for item in accounts.json()] == ["SIM001"]
    combined = checked.text + accounts.text
    assert "password" not in combined.lower()
    assert "token" not in combined.lower()
    assert "LIVE001" not in combined


def test_preflight_and_observe_session_flow(tmp_path: Path) -> None:
    client, slug, broker = _client(tmp_path)
    assert client.post(
        "/api/v1/broker-connections/ft-client/check", json=_connection_payload()
    ).status_code == 200

    preflight = client.post(
        f"/api/v1/strategies/{slug}/client-simulation/preflight", json=_payload()
    )
    created = client.post(
        f"/api/v1/strategies/{slug}/client-simulation/sessions", json=_payload()
    )

    assert preflight.status_code == 200
    assert preflight.json()["ready"] is True
    assert created.status_code == 201
    body = created.json()
    assert body["execution_mode"] == "observe"
    assert body["intents"][0]["status"] == "validated"
    assert body["monitoring"]["trade_accounts"][0]["total_rate"] == 0.25
    assert broker.submit_calls == 0

    session_id = body["session_id"]
    detail = client.get(
        f"/api/v1/strategies/{slug}/client-simulation/sessions/{session_id}"
    )
    account = client.get(
        f"/api/v1/strategies/{slug}/client-simulation/sessions/{session_id}/account"
    )
    events = client.get(
        f"/api/v1/strategies/{slug}/client-simulation/sessions/{session_id}/events"
    )
    assert detail.status_code == account.status_code == events.status_code == 200
    assert account.json()["funds"]["available"] == 800_000
    assert events.json()[0]["type"] == "session_started"
    persisted = "\n".join(
        path.read_text(encoding="utf-8", errors="ignore")
        for path in tmp_path.rglob("*")
        if path.is_file()
    )
    assert "one-time-secret" not in persisted


def test_manual_approval_and_precise_stop(tmp_path: Path) -> None:
    client, slug, broker = _client(tmp_path)
    created = client.post(
        f"/api/v1/strategies/{slug}/client-simulation/sessions", json=_payload("manual")
    )
    body = created.json()
    session_id = body["session_id"]
    intent_id = body["intents"][0]["intent_id"]

    approved = client.post(
        f"/api/v1/strategies/{slug}/client-simulation/sessions/{session_id}/intents/{intent_id}/approve"
    )
    stopped = client.post(
        f"/api/v1/strategies/{slug}/client-simulation/sessions/{session_id}/stop"
    )

    assert approved.status_code == 200
    assert approved.json()["orders"][0]["parent_order_id"] == "99"
    assert stopped.status_code == 200
    assert stopped.json()["status"] == "stopped"
    assert broker.cancelled == ["99"]


def test_preflight_failure_is_structured(tmp_path: Path) -> None:
    client, slug, _ = _client(tmp_path)
    payload = _payload("auto")
    payload["execution_route"] = "direct_order"

    response = client.post(
        f"/api/v1/strategies/{slug}/client-simulation/sessions", json=payload
    )

    assert response.status_code == 400
    assert response.json()["error"]["code"] == "preflight_failed"
    checks = response.json()["error"]["details"]["checks"]
    assert any(item["code"] == "algorithm_config_invalid" for item in checks)
