"""Non-convex FT intelligent trading client API v0.0.23 adapter."""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
from collections.abc import AsyncIterator
from typing import Any
from urllib.parse import urlparse

import httpx

from autostrategy.brokers.models import (
    AccountHealth,
    BrokerEvent,
    BrokerOrder,
    ChildOrder,
    FtAccount,
    FtClientCredentials,
    FundSnapshot,
    MonitoringMetric,
    MonitoringSnapshot,
    OrderIntent,
    PositionSnapshot,
    normalize_algorithm_child_status,
    normalize_parent_status,
)
from autostrategy.config import FtClientConfig


class FtClientError(RuntimeError):
    """Safe broker error that never includes credential-bearing URLs."""

    def __init__(self, code: str, message: str, *, retryable: bool = False) -> None:
        super().__init__(message)
        self.code = code
        self.retryable = retryable


class FtClientBroker:
    """Synchronous HTTP adapter plus asynchronous WebSocket event stream."""

    def __init__(
        self,
        config: FtClientConfig,
        *,
        http_client: httpx.Client | None = None,
        credentials: FtClientCredentials | None = None,
    ) -> None:
        self.config = config
        self._http_client = http_client or httpx.Client(
            base_url=config.base_url,
            timeout=httpx.Timeout(10.0, connect=3.0),
        )
        self._owns_client = http_client is None
        self._credentials = credentials
        self._token: str | None = None
        self._accounts: list[FtAccount] = []
        self._ft_account_name = ""

    def __repr__(self) -> str:
        return f"FtClientBroker(base_url={self.config.base_url!r}, connected={bool(self._token)})"

    @property
    def connected(self) -> bool:
        return self._token is not None

    def _credential(self, env_name: str) -> str:
        value = os.environ.get(env_name)
        if not value:
            raise FtClientError(
                "configuration_error",
                f"Required credential environment variable {env_name} is not set.",
            )
        return value

    def _password(self) -> str:
        if self._credentials is not None:
            password = self._credentials.password.get_secret_value()
            transform = self._credentials.password_transform
        else:
            password = self._credential(self.config.password_env)
            transform = self.config.password_transform
        if transform == "md5_32_lower":
            return hashlib.md5(password.encode("utf-8"), usedforsecurity=False).hexdigest()
        return password

    def _request(
        self,
        method: str,
        path: str,
        *,
        params: dict[str, Any] | None = None,
        json_body: dict[str, Any] | None = None,
        retry_query: bool = True,
        reauthenticate: bool = True,
    ) -> Any:
        attempts = 3 if method == "GET" and retry_query else 1
        last_error: Exception | None = None
        for attempt in range(attempts):
            try:
                response = self._http_client.request(
                    method,
                    f"{self.config.base_url}{path}",
                    params=params,
                    json=json_body,
                )
                response.raise_for_status()
                payload = response.json()
            except (httpx.HTTPError, ValueError) as exc:
                last_error = exc
                if attempt + 1 < attempts:
                    continue
                raise FtClientError(
                    "client_unavailable",
                    f"FT client request failed at {path}.",
                    retryable=method == "GET",
                ) from exc
            code = int(payload.get("code", -1)) if isinstance(payload, dict) else -1
            if code == 0:
                return payload.get("data")
            if code == 1001 and reauthenticate and self._token:
                self.connect()
                retry_params = dict(params or {})
                retry_json = dict(json_body or {})
                if "token" in retry_params:
                    retry_params["token"] = self._token
                if "token" in retry_json:
                    retry_json["token"] = self._token
                return self._request(
                    method,
                    path,
                    params=retry_params or None,
                    json_body=retry_json or None,
                    retry_query=False,
                    reauthenticate=False,
                )
            detail = payload.get("data") or payload.get("toast") or "FT client rejected request."
            raise FtClientError(f"ft_client_{code}", str(detail), retryable=False)
        raise FtClientError(
            "client_unavailable", f"FT client request failed at {path}."
        ) from last_error

    def connect(self) -> list[FtAccount]:
        ft_account = (
            self._credentials.ft_account
            if self._credentials is not None
            else self._credential(self.config.ft_account_env)
        )
        data = self._request(
            "GET",
            "/api/ft_acc_login",
            params={"ft_acc": ft_account, "password": self._password()},
            retry_query=False,
            reauthenticate=False,
        )
        if not isinstance(data, dict) or not data.get("token"):
            raise FtClientError("authentication_error", "FT client login returned no token.")
        self._token = str(data["token"])
        self._ft_account_name = str(data.get("ft_acc_name") or "")
        accounts: list[FtAccount] = []
        for raw in data.get("accs") or []:
            accounts.append(
                FtAccount(
                    ft_account=ft_account,
                    ft_account_name=self._ft_account_name,
                    broker_id=str(raw.get("broker_id") or ""),
                    broker_name=str(raw.get("broker_name") or ""),
                    trade_account=str(raw.get("trade_acc") or ""),
                    nickname=str(raw.get("trade_acc_nick") or ""),
                    login_status=bool(raw.get("login_status")),
                )
            )
        self._accounts = accounts
        return list(accounts)

    def _require_token(self) -> str:
        if not self._token:
            self.connect()
        assert self._token is not None
        return self._token

    def disconnect(self) -> None:
        if self._token:
            self._request(
                "GET",
                "/api/logout_all",
                params={"token": self._token},
                retry_query=False,
                reauthenticate=False,
            )
        self._token = None
        self._accounts = []
        if self._owns_client:
            self._http_client.close()

    def list_accounts(self) -> list[FtAccount]:
        if not self._accounts:
            return self.connect()
        return list(self._accounts)

    def health(self) -> dict[str, AccountHealth]:
        raw = self._request(
            "GET", "/api/query_acc_status", params={"token": self._require_token()}
        )
        result: dict[str, AccountHealth] = {}
        for trade_account, value in (raw or {}).items():
            result[str(trade_account)] = AccountHealth(
                trade_account=str(trade_account),
                name=str(value.get("name") or ""),
                login_status=int(value.get("login_status") or 0) == 1,
                order_engine_status=int(value.get("order_engine_status") or 0) == 1,
            )
        return result

    def get_funds(self, trade_account: str, broker_id: str) -> FundSnapshot:
        raw = self._request(
            "GET",
            "/api/get_fund_by_acc",
            params={
                "broker_id": broker_id,
                "acc_id": trade_account,
                "token": self._require_token(),
            },
        )
        raw = raw or {}
        asset = float(raw.get("asset") or 0)
        balance = float(raw.get("balance") or 0)
        diagnostics: list[str] = []
        risk_equity = asset
        if risk_equity <= 0 < balance:
            risk_equity = balance
            diagnostics.append("fund_asset_fallback")
        return FundSnapshot(
            trade_account=trade_account,
            balance=balance,
            asset=asset,
            available=float(raw.get("available") or 0),
            frozen=float(raw.get("frozen") or 0),
            profit=float(raw.get("profit") or 0),
            risk_equity=risk_equity,
            diagnostics=diagnostics,
            raw=raw,
        )

    def get_positions(self, trade_account: str, broker_id: str) -> list[PositionSnapshot]:
        raw_items = self._request(
            "GET",
            "/api/get_position_by_acc",
            params={
                "broker_id": broker_id,
                "acc_id": trade_account,
                "token": self._require_token(),
            },
        )
        return [
            PositionSnapshot(
                trade_account=trade_account,
                stock_code=str(raw.get("stock_code") or ""),
                total_volume=int(raw.get("total_vol") or 0),
                available_volume=int(raw.get("avail_vol") or 0),
                locked_volume=int(raw.get("lock_vol") or 0),
                in_transit_volume=int(raw.get("onway_vol") or raw.get("in_transit_vol") or 0),
                exchange_id=int(raw["exchange_id"]) if raw.get("exchange_id") is not None else None,
                raw=raw,
            )
            for raw in (raw_items or [])
        ]

    def submit_order(self, intent: OrderIntent) -> BrokerOrder:
        algorithm = intent.algorithm
        mudan = {
            "basket_name": intent.session_id,
            "order_vol": str(intent.quantity),
            "stock_code": intent.broker_symbol,
            "begin_time": intent.execution_window_start,
            "end_time": intent.execution_window_end,
            "bs_flag": intent.side.upper(),
            "trade_acc": intent.trade_account,
            "external_id": intent.intent_id,
            "strategy_type": algorithm.strategy_type,
            "algo_param": algorithm.serialize_params(),
            "reach_limit_continue": False,
            "over_time_continue": False,
        }
        raw = self._request(
            "POST",
            "/api/api_upload_mudan",
            json_body={"mudans": [mudan], "token": self._require_token()},
            retry_query=False,
        )
        if isinstance(raw, list):
            raw = raw[0] if raw else {}
        raw = raw or {}
        return self._parse_parent(
            raw,
            fallback_external_id=intent.intent_id,
            basket=intent.session_id,
        )

    def _parse_parent(
        self, raw: dict[str, Any], *, fallback_external_id: str = "", basket: str = ""
    ) -> BrokerOrder:
        raw_status = int(raw.get("status") or 0)
        return BrokerOrder(
            parent_order_id=str(raw.get("id") or raw.get("mudan_id") or ""),
            external_id=str(raw.get("external_id") or fallback_external_id),
            trade_account=str(raw.get("trade_acc") or ""),
            basket_name=str(raw.get("basket_name") or raw.get("tag") or basket),
            stock_code=str(raw.get("stock_code") or ""),
            order_volume=int(raw.get("order_vol") or 0),
            trade_volume=int(raw.get("trade_vol") or 0),
            raw_status=raw_status,
            raw_status_msg=str(raw.get("status_msg") or ""),
            normalized_status=normalize_parent_status(raw_status),
            raw=raw,
        )

    def get_orders(self, trade_account: str, broker_id: str) -> list[BrokerOrder]:
        data = self._request(
            "GET",
            "/api/get_mudan_by_acc",
            params={
                "broker_id": broker_id,
                "acc_id": trade_account,
                "mudan_type": 100,
                "token": self._require_token(),
            },
        )
        return [self._parse_parent(raw) for raw in (data or [])]

    def _parse_child(self, raw: dict[str, Any]) -> ChildOrder:
        raw_status = int(raw.get("status") or 0)
        return ChildOrder(
            child_order_id=str(raw.get("id") or raw.get("local_id") or ""),
            parent_order_id=str(raw.get("strategy_order_id") or raw.get("mudan_id") or ""),
            trade_account=str(raw.get("trade_acc") or ""),
            stock_code=str(raw.get("stock_code") or ""),
            order_volume=int(raw.get("order_vol") or 0),
            trade_volume=int(raw.get("trade_vol") or 0),
            trade_price=float(raw.get("trade_price") or 0),
            raw_status=raw_status,
            raw_status_msg=str(raw.get("status_msg") or ""),
            normalized_status=normalize_algorithm_child_status(raw_status),
            raw=raw,
        )

    def get_child_orders(self, parent_order_id: str) -> list[ChildOrder]:
        data = self._request(
            "GET",
            "/api/get_zidan_by_mudan_id",
            params={"mudan_id": parent_order_id, "token": self._require_token()},
        )
        return [self._parse_child(raw) for raw in (data or [])]

    def get_account_child_orders(
        self, trade_account: str, broker_id: str
    ) -> list[ChildOrder]:
        data = self._request(
            "GET",
            "/api/get_zidan_by_acc",
            params={
                "broker_id": broker_id,
                "acc_id": trade_account,
                "mudan_type": 100,
                "token": self._require_token(),
            },
        )
        return [self._parse_child(raw) for raw in (data or [])]

    def cancel_orders(self, parent_order_ids: list[str]) -> None:
        self._request(
            "POST",
            "/api/op_batch_mudan",
            json_body={
                "token": self._require_token(),
                "mudan_op_type": 4,
                "mudan_id": [
                    int(value) if value.isdigit() else value for value in parent_order_ids
                ],
                "mudan_type": 100,
            },
            retry_query=False,
        )

    def get_monitoring(self, ft_account: str) -> MonitoringSnapshot:
        raw = self._request(
            "GET",
            "/api/get_algo_monitoring_info",
            params={"ft_acc": ft_account, "token": self._require_token()},
        )
        raw = raw or {}
        diagnostics: list[str] = []

        def parse_metric(value: dict[str, Any]) -> MonitoringMetric:
            metric = MonitoringMetric(
                account_id=str(value.get("account_id") or ""),
                trade_account=str(value.get("trade_acc") or ""),
                basket_id=str(value["basket_id"]) if value.get("basket_id") is not None else None,
                basket_name=value.get("basket_name"),
                **{
                    key: float(value.get(key) or 0)
                    for key in (
                        "plan_buy",
                        "plan_sale",
                        "trade_buy",
                        "trade_sale",
                        "buy_rate",
                        "sale_rate",
                        "exposure",
                        "cancel_rate",
                        "total_rate",
                        "error_rate",
                    )
                },
            )
            rates = (
                metric.buy_rate,
                metric.sale_rate,
                metric.cancel_rate,
                metric.total_rate,
                metric.error_rate,
            )
            if any(rate < 0 or rate > 1 for rate in rates) or not -1 <= metric.exposure <= 1:
                diagnostics.append("monitoring_value_out_of_range")
            return metric

        return MonitoringSnapshot(
            ft_account=ft_account,
            trade_accounts=[parse_metric(item) for item in raw.get("trade_acc_infos") or []],
            baskets=[parse_metric(item) for item in raw.get("basket_infos") or []],
            diagnostics=list(dict.fromkeys(diagnostics)),
            raw=raw,
        )

    async def stream_events(self) -> AsyncIterator[BrokerEvent]:
        token = self._require_token()
        parsed = urlparse(self.config.base_url)
        websocket_url = f"ws://{parsed.netloc}/ws/{token}"
        try:
            from websockets.asyncio.client import connect
        except ImportError as exc:  # pragma: no cover - dependency error is environment-specific
            raise FtClientError(
                "configuration_error", "The websockets package is required for FT event streaming."
            ) from exc
        delay = 1
        while self._token:
            try:
                async with connect(websocket_url, ping_interval=None) as websocket:
                    delay = 1
                    async for message in websocket:
                        raw = json.loads(message)
                        if raw.get("topic") == "Ping":
                            await websocket.send(
                                json.dumps(
                                    {"topic": "Pong", "data": raw.get("data")},
                                    ensure_ascii=False,
                                )
                            )
                            continue
                        yield BrokerEvent(
                            topic=str(raw.get("topic") or "unknown"),
                            data=raw.get("data"),
                        )
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                if not self._token:
                    return
                await asyncio.sleep(delay)
                delay = min(delay * 2, 30)
                if delay == 30 and isinstance(exc, json.JSONDecodeError):
                    yield BrokerEvent(topic="Error", data={"code": "websocket_invalid_json"})
