from __future__ import annotations

import hashlib
import json
from typing import Any

import httpx
import pytest
from pydantic import ValidationError

from autostrategy.brokers.ft_client import FtClientBroker, FtClientError
from autostrategy.brokers.models import (
    AlgorithmConfig,
    FtClientCredentials,
    OrderIntent,
    normalize_algorithm_child_status,
    normalize_direct_order_status,
    normalize_parent_status,
)
from autostrategy.config import FtClientConfig


class RecordedTransport:
    def __init__(self, responses: dict[tuple[str, str], dict[str, Any]]) -> None:
        self.responses = responses
        self.requests: list[httpx.Request] = []

    def handler(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        key = (request.method, request.url.path)
        response = self.responses.get(key)
        if response is None:
            return httpx.Response(404, json={"code": 1002, "data": f"missing fake {key}"})
        return httpx.Response(200, json=response)

    @property
    def last_request(self) -> httpx.Request:
        return self.requests[-1]

    @property
    def last_json(self) -> dict[str, Any]:
        return json.loads(self.last_request.content.decode("utf-8"))


def _config(**overrides: Any) -> FtClientConfig:
    values: dict[str, Any] = {
        "enabled": True,
        "confirmed_client_version": "3.11.4",
        "allowed_simulation_accounts": ["SIM001"],
        "symbol_mapping": {"588000.SH": "588000.SH"},
        "external_id_max_length": 64,
        "external_id_scope_confirmed": True,
    }
    values.update(overrides)
    return FtClientConfig(**values)


def _broker(
    monkeypatch: pytest.MonkeyPatch,
    responses: dict[tuple[str, str], dict[str, Any]],
    **config_overrides: Any,
) -> tuple[FtClientBroker, RecordedTransport]:
    monkeypatch.setenv("AUTOSTRATEGY_FT_ACCOUNT", "ft-user")
    monkeypatch.setenv("AUTOSTRATEGY_FT_PASSWORD", "secret")
    recorder = RecordedTransport(responses)
    client = httpx.Client(transport=httpx.MockTransport(recorder.handler))
    return FtClientBroker(_config(**config_overrides), http_client=client), recorder


def _login_response(login_status: bool = True) -> dict[str, Any]:
    return {
        "code": 0,
        "data": {
            "accs": [
                {
                    "broker_id": "1",
                    "trade_acc": "SIM001",
                    "trade_acc_nick": "模拟账户",
                    "broker_name": "测试券商",
                    "login_status": login_status,
                }
            ],
            "ft_acc_name": "FT 模拟",
            "token": "sensitive-token",
        },
    }


def test_ft_config_rejects_non_loopback_base_url() -> None:
    with pytest.raises(ValidationError):
        FtClientConfig(enabled=True, base_url="https://broker.example.com")


def test_ft_config_rejects_unknown_password_transform() -> None:
    with pytest.raises(ValidationError):
        FtClientConfig(password_transform="sha256")


def test_order_status_namespaces_are_independent() -> None:
    assert normalize_parent_status(0) == "submitted"
    assert normalize_parent_status(11) == "residual"
    assert normalize_parent_status(21) == "unknown"
    assert normalize_algorithm_child_status(3) == "cancelled"
    assert normalize_algorithm_child_status(8) == "expired"
    assert normalize_direct_order_status(3) == "filled"
    assert normalize_direct_order_status(6) == "cancel_pending"


def test_login_preserves_account_level_login_status(monkeypatch: pytest.MonkeyPatch) -> None:
    broker, recorder = _broker(
        monkeypatch,
        {("GET", "/api/ft_acc_login"): _login_response(login_status=False)},
    )

    accounts = broker.connect()

    assert accounts[0].trade_account == "SIM001"
    assert accounts[0].login_status is False
    assert recorder.last_request.url.params["ft_acc"] == "ft-user"
    assert recorder.last_request.url.params["password"] == "secret"
    assert "sensitive-token" not in repr(broker)


def test_login_supports_lowercase_md5_password(monkeypatch: pytest.MonkeyPatch) -> None:
    broker, recorder = _broker(
        monkeypatch,
        {("GET", "/api/ft_acc_login"): _login_response()},
        password_transform="md5_32_lower",
    )

    broker.connect()

    assert recorder.last_request.url.params["password"] == hashlib.md5(
        b"secret", usedforsecurity=False
    ).hexdigest()


def test_login_accepts_ephemeral_frontend_credentials_without_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("AUTOSTRATEGY_FT_ACCOUNT", raising=False)
    monkeypatch.delenv("AUTOSTRATEGY_FT_PASSWORD", raising=False)
    recorder = RecordedTransport({("GET", "/api/ft_acc_login"): _login_response()})
    client = httpx.Client(transport=httpx.MockTransport(recorder.handler))
    credentials = FtClientCredentials(
        ft_account="ui-user",
        password="ui-secret",
        password_transform="plain",
    )
    broker = FtClientBroker(_config(), http_client=client, credentials=credentials)

    broker.connect()

    assert recorder.last_request.url.params["ft_acc"] == "ui-user"
    assert recorder.last_request.url.params["password"] == "ui-secret"
    assert "ui-secret" not in repr(credentials)
    assert "ui-secret" not in repr(broker)


def test_broker_error_does_not_leak_credentials_or_token(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    broker, _ = _broker(
        monkeypatch,
        {("GET", "/api/ft_acc_login"): {"code": 1002, "data": "login rejected"}},
    )

    with pytest.raises(FtClientError) as caught:
        broker.connect()

    message = str(caught.value)
    assert "secret" not in message
    assert "sensitive-token" not in message
    assert "login rejected" in message


def test_funds_use_asset_and_fall_back_to_balance(monkeypatch: pytest.MonkeyPatch) -> None:
    broker, _ = _broker(
        monkeypatch,
        {
            ("GET", "/api/ft_acc_login"): _login_response(),
            ("GET", "/api/get_fund_by_acc"): {
                "code": 0,
                "data": {
                    "asset": 0,
                    "balance": 900_000,
                    "available": 600_000,
                    "frozen": 10_000,
                },
            },
        },
    )
    broker.connect()

    funds = broker.get_funds("SIM001", "1")

    assert funds.available == 600_000
    assert funds.risk_equity == 900_000
    assert "fund_asset_fallback" in funds.diagnostics


def test_submit_parent_order_maps_external_id_and_structured_algorithm_params(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    broker, recorder = _broker(
        monkeypatch,
        {
            ("GET", "/api/ft_acc_login"): _login_response(),
            ("POST", "/api/api_upload_mudan"): {
                "code": 0,
                "data": {"id": 42, "status": 1, "external_id": "intent-1"},
            },
        },
    )
    broker.connect()
    intent = OrderIntent(
        intent_id="intent-1",
        intent_key="grid:buy:1",
        session_id="session-1",
        symbol="588000.SH",
        broker_symbol="588000.SH",
        side="buy",
        quantity=100,
        signal_price=1.23,
        trade_account="SIM001",
        broker_id="1",
        execution_window_start="093500",
        execution_window_end="145000",
        algorithm=AlgorithmConfig(
            strategy_type="TWAP",
            params={"delay_end_time": 10},
        ),
    )

    order = broker.submit_order(intent)

    mudan = recorder.last_json["mudans"][0]
    assert recorder.last_json["token"] == "sensitive-token"
    assert mudan["external_id"] == "intent-1"
    assert mudan["basket_name"] == "session-1"
    assert mudan["strategy_type"] == "TWAP"
    assert mudan["algo_param"] == "delay_end_time=10"
    assert mudan["reach_limit_continue"] is False
    assert mudan["over_time_continue"] is False
    assert "limit_price" not in mudan["algo_param"]
    assert order.parent_order_id == "42"
    assert order.normalized_status == "working"


def test_algorithm_params_reject_delimiters() -> None:
    with pytest.raises(ValidationError):
        AlgorithmConfig(strategy_type="TWAP", params={"bad;key": "value"})
    with pytest.raises(ValidationError):
        AlgorithmConfig(strategy_type="TWAP", params={"key": "bad=value"})


def test_monitoring_parses_account_and_basket_dimensions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    broker, recorder = _broker(
        monkeypatch,
        {
            ("GET", "/api/ft_acc_login"): _login_response(),
            ("GET", "/api/get_algo_monitoring_info"): {
                "code": 0,
                "data": {
                    "trade_acc_infos": [
                        {
                            "account_id": "ft-user",
                            "trade_acc": "SIM001",
                            "plan_buy": 1000,
                            "plan_sale": 0,
                            "trade_buy": 500,
                            "trade_sale": 0,
                            "buy_rate": 0.5,
                            "sale_rate": 0,
                            "exposure": 1,
                            "cancel_rate": 0.1,
                            "total_rate": 0.5,
                            "error_rate": 0,
                        }
                    ],
                    "basket_infos": [
                        {
                            "account_id": "ft-user",
                            "trade_acc": "SIM001",
                            "basket_id": 88,
                            "basket_name": "session-1",
                            "plan_buy": 1000,
                            "plan_sale": 0,
                            "trade_buy": 500,
                            "trade_sale": 0,
                            "buy_rate": 0.5,
                            "sale_rate": 0,
                            "exposure": 1,
                            "cancel_rate": 0.1,
                            "total_rate": 0.5,
                            "error_rate": 0,
                        }
                    ],
                },
            },
        },
    )
    broker.connect()

    result = broker.get_monitoring("ft-user")

    assert recorder.last_request.url.params["ft_acc"] == "ft-user"
    assert result.trade_accounts[0].total_rate == 0.5
    assert result.baskets[0].basket_name == "session-1"
    assert result.baskets[0].exposure == 1


def test_cancel_parent_orders_uses_documented_batch_operation_fields(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    broker, recorder = _broker(
        monkeypatch,
        {
            ("GET", "/api/ft_acc_login"): _login_response(),
            ("POST", "/api/op_batch_mudan"): {"code": 0, "data": "撤销成功"},
        },
    )
    broker.connect()

    broker.cancel_orders(["610378"])

    assert recorder.last_json == {
        "token": "sensitive-token",
        "mudan_op_type": 4,
        "mudan_id": [610378],
        "mudan_type": 100,
    }


def test_account_order_queries_are_scoped_to_algorithm_mother_orders(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    broker, recorder = _broker(
        monkeypatch,
        {
            ("GET", "/api/ft_acc_login"): _login_response(),
            ("GET", "/api/get_mudan_by_acc"): {"code": 0, "data": []},
            ("GET", "/api/get_zidan_by_acc"): {"code": 0, "data": []},
        },
    )
    broker.connect()

    broker.get_orders("SIM001", "1")
    parent_params = dict(recorder.last_request.url.params)
    broker.get_account_child_orders("SIM001", "1")
    child_params = dict(recorder.last_request.url.params)

    assert parent_params["mudan_type"] == "100"
    assert child_params["mudan_type"] == "100"
