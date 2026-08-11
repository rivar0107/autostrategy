"""FT-client-backed simulation session orchestration."""

from __future__ import annotations

import asyncio
import hashlib
import importlib.util
import json
import os
import tempfile
import threading
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from types import ModuleType
from typing import Any, Literal
from zoneinfo import ZoneInfo

import yaml
from pydantic import BaseModel, ConfigDict, Field, SecretStr

from autostrategy.brokers.base import BrokerAdapter
from autostrategy.brokers.cn_symbols import (
    ftshare_market_data_type,
    is_supported_cn_security,
)
from autostrategy.brokers.ft_client import FtClientBroker, FtClientError
from autostrategy.brokers.models import (
    AccountHealth,
    AlgorithmConfig,
    BrokerOrder,
    ChildOrder,
    FtAccount,
    FtClientCredentials,
    FundSnapshot,
    MonitoringSnapshot,
    OrderIntent,
    PositionSnapshot,
    utc_now,
)
from autostrategy.config import FtClientConfig
from autostrategy.core.strategy import StrategyStatus
from autostrategy.data.ftshare import (
    aggregate_ten_minute_bars,
    fetch_daily_ohlc,
    fetch_intraday_prices,
)
from autostrategy.services.exceptions import AutostrategyServiceError
from autostrategy.services.strategy_service import StrategyService

ExecutionMode = Literal["observe", "manual", "auto"]
SessionStatus = Literal[
    "draft",
    "preflight_failed",
    "ready",
    "starting",
    "running",
    "paused",
    "stopping",
    "stopped",
    "completed",
    "needs_attention",
    "failed",
]

_ACTIVE_SESSION_STATUSES = {"ready", "starting", "running", "paused", "stopping", "needs_attention"}
_TERMINAL_ORDER_STATUSES = {
    "completed",
    "filled",
    "cancelled",
    "stopped",
    "expired",
    "failed",
}
_SESSION_FILES = (
    "events.jsonl",
    "order_intents.jsonl",
    "broker_orders.jsonl",
    "child_orders.jsonl",
    "fills.jsonl",
    "account_snapshots.jsonl",
    "monitoring_snapshots.jsonl",
    "reconciliation.jsonl",
)


class ClientSimulationError(AutostrategyServiceError):
    """Structured simulation service error."""

    def __init__(self, code: str, message: str, details: dict[str, Any] | None = None) -> None:
        self.code = code
        super().__init__(message, details=details)


class SimulationModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class RiskLimits(SimulationModel):
    max_order_pct: float = Field(default=5.0, gt=0, le=100)
    max_symbol_position_pct: float = Field(default=20.0, gt=0, le=100)
    max_total_position_pct: float = Field(default=80.0, gt=0, le=100)


class FtConnectionInput(SimulationModel):
    """Customer-entered local connection fields; password remains memory-only."""

    base_url: str = "http://127.0.0.1:11356"
    ft_account: str = Field(min_length=1)
    password: SecretStr
    password_transform: Literal["plain", "md5_32_lower"] = "plain"
    confirmed_client_version: str = Field(min_length=1)
    allowed_simulation_accounts: list[str] = Field(min_length=1)
    allowed_symbols: list[str] = Field(default_factory=list)
    symbol_mapping: dict[str, str] = Field(default_factory=dict)
    allowed_algorithms: list[str] = Field(default_factory=lambda: ["TWAP"])
    external_id_max_length: int = Field(default=64, ge=1)
    external_id_scope_confirmed: bool = False


class ClientSimulationRequest(SimulationModel):
    trade_account: str
    execution_mode: ExecutionMode = "observe"
    acknowledge_simulation: bool = False
    execution_route: Literal["algorithm_parent", "direct_order"] = "algorithm_parent"
    execution_window_start: str = "093500"
    execution_window_end: str = "145000"
    algorithm: AlgorithmConfig = Field(default_factory=AlgorithmConfig)
    risk: RiskLimits = Field(default_factory=RiskLimits)


class PreflightCheck(SimulationModel):
    code: str
    passed: bool
    message: str
    details: dict[str, Any] = Field(default_factory=dict)


class PreflightResult(SimulationModel):
    ready: bool
    checks: list[PreflightCheck]
    account: FtAccount | None = None
    health: AccountHealth | None = None
    funds: FundSnapshot | None = None
    positions: list[PositionSnapshot] = Field(default_factory=list)
    monitoring: MonitoringSnapshot | None = None


class ClientSimulationSession(SimulationModel):
    session_id: str
    strategy_slug: str
    strategy_version: int
    strategy_digest: str = ""
    status: SessionStatus
    execution_mode: ExecutionMode
    execution_route: Literal["algorithm_parent", "direct_order"]
    ft_account: str
    trade_account: str
    broker_id: str
    account_nickname: str = ""
    account_login_status: bool = False
    order_engine_status: bool = False
    client_version: str
    minimum_client_version: str
    algorithm: AlgorithmConfig
    risk: RiskLimits
    execution_window_start: str = "093500"
    execution_window_end: str = "145000"
    last_evaluated_bar_at: str | None = None
    preflight: list[PreflightCheck] = Field(default_factory=list)
    funds: FundSnapshot | None = None
    positions: list[PositionSnapshot] = Field(default_factory=list)
    monitoring: MonitoringSnapshot | None = None
    intents: list[OrderIntent] = Field(default_factory=list)
    orders: list[BrokerOrder] = Field(default_factory=list)
    child_orders: list[ChildOrder] = Field(default_factory=list)
    latest_error: str | None = None
    created_at: str = Field(default_factory=utc_now)
    updated_at: str = Field(default_factory=utc_now)


@dataclass
class _PreflightContext:
    result: PreflightResult
    broker: BrokerAdapter
    account: FtAccount | None
    market_context: dict[str, Any]


class FtshareDailyMarketContextProvider:
    """Build a synchronized context from completed FTShare stock/ETF daily bars."""

    def __init__(
        self,
        *,
        fetcher: Callable[..., Any] = fetch_daily_ohlc,
        clock: Callable[[], datetime] | None = None,
        history_limit: int = 500,
    ) -> None:
        self._fetcher = fetcher
        self._clock = clock or (
            lambda: datetime.now(ZoneInfo("Asia/Shanghai"))
        )
        self._history_limit = history_limit

    def __call__(
        self, symbols: list[str], config_data: dict[str, Any]
    ) -> dict[str, Any]:
        now = self._clock().astimezone(ZoneInfo("Asia/Shanghai"))
        end_date = (now.date() - timedelta(days=1)).isoformat()
        histories: dict[str, list[dict[str, Any]]] = {}
        latest_dates: list[str] = []
        for symbol in symbols:
            asset_type = ftshare_market_data_type(symbol)
            if asset_type is None:
                raise ValueError(f"Unsupported Shanghai/Shenzhen market-data symbol: {symbol}")
            frame = self._fetcher(
                symbol,
                limit=self._history_limit,
                type_=asset_type,
                end_date=end_date,
            )
            bars: list[dict[str, Any]] = []
            for at, row in frame.iterrows():
                at_text = at.date().isoformat() if hasattr(at, "date") else str(at)[:10]
                bars.append(
                    {
                        "at": at_text,
                        "symbol": symbol,
                        "open": float(row["open"]),
                        "high": float(row["high"]),
                        "low": float(row["low"]),
                        "close": float(row["close"]),
                        "volume": float(row.get("volume", 0) or 0),
                    }
                )
            bars.sort(key=lambda item: item["at"])
            histories[symbol] = bars
            if bars:
                latest_dates.append(bars[-1]["at"])
        if len(latest_dates) != len(symbols) or not latest_dates:
            return {
                "bars_by_symbol": {},
                "history_by_symbol": histories,
                "completed_bar_at": None,
            }
        completed_bar_at = min(latest_dates)
        synchronized = {
            symbol: [bar for bar in bars if bar["at"] <= completed_bar_at]
            for symbol, bars in histories.items()
        }
        return {
            "bars_by_symbol": {
                symbol: bars[-1] for symbol, bars in synchronized.items() if bars
            },
            "history_by_symbol": synchronized,
            "completed_bar_at": completed_bar_at,
        }


class FtshareTenMinuteMarketContextProvider:
    """Build synchronized completed 10-minute bars plus separate daily history."""

    def __init__(
        self,
        *,
        minute_fetcher: Callable[..., Any] = fetch_intraday_prices,
        daily_fetcher: Callable[..., Any] = fetch_daily_ohlc,
        clock: Callable[[], datetime] | None = None,
        daily_history_limit: int = 500,
    ) -> None:
        self._minute_fetcher = minute_fetcher
        self._daily_fetcher = daily_fetcher
        self._clock = clock or (lambda: datetime.now(ZoneInfo("Asia/Shanghai")))
        self._daily_history_limit = daily_history_limit
        self._daily_cache: dict[tuple[str, str], list[dict[str, Any]]] = {}

    def __call__(
        self, symbols: list[str], config_data: dict[str, Any]
    ) -> dict[str, Any]:
        del config_data
        now = self._clock().astimezone(ZoneInfo("Asia/Shanghai"))
        daily_end_date = (now.date() - timedelta(days=1)).isoformat()
        daily_histories: dict[str, list[dict[str, Any]]] = {}
        intraday_histories: dict[str, list[dict[str, Any]]] = {}

        for symbol in symbols:
            asset_type = ftshare_market_data_type(symbol)
            if asset_type is None:
                raise ValueError(f"Unsupported Shanghai/Shenzhen market-data symbol: {symbol}")
            cache_key = (symbol, daily_end_date)
            daily_bars = self._daily_cache.get(cache_key)
            if daily_bars is None:
                frame = self._daily_fetcher(
                    symbol,
                    limit=self._daily_history_limit,
                    type_=asset_type,
                    end_date=daily_end_date,
                )
                daily_bars = self._frame_bars(frame, symbol)
                self._daily_cache[cache_key] = daily_bars
            daily_histories[symbol] = list(daily_bars)
            minute_rows = self._minute_fetcher(symbol, type_=asset_type)
            intraday_histories[symbol] = aggregate_ten_minute_bars(
                list(minute_rows), symbol=symbol, now=now
            )

        common_times: set[str] | None = None
        for bars in intraday_histories.values():
            times = {str(bar["at"]) for bar in bars}
            common_times = times if common_times is None else common_times & times
        if not common_times:
            return {
                "bars_by_symbol": {},
                "history_by_symbol": daily_histories,
                "intraday_history_by_symbol": intraday_histories,
                "completed_bar_at": None,
            }

        completed_bar_at = max(common_times)
        synchronized_intraday = {
            symbol: [bar for bar in bars if str(bar["at"]) <= completed_bar_at]
            for symbol, bars in intraday_histories.items()
        }
        return {
            "bars_by_symbol": {
                symbol: next(
                    bar for bar in reversed(bars) if str(bar["at"]) == completed_bar_at
                )
                for symbol, bars in synchronized_intraday.items()
            },
            "history_by_symbol": daily_histories,
            "intraday_history_by_symbol": synchronized_intraday,
            "completed_bar_at": completed_bar_at,
        }

    @staticmethod
    def _frame_bars(frame: Any, symbol: str) -> list[dict[str, Any]]:
        bars: list[dict[str, Any]] = []
        for at, row in frame.iterrows():
            at_text = at.date().isoformat() if hasattr(at, "date") else str(at)[:10]
            bars.append(
                {
                    "at": at_text,
                    "symbol": symbol,
                    "open": float(row["open"]),
                    "high": float(row["high"]),
                    "low": float(row["low"]),
                    "close": float(row["close"]),
                    "volume": float(row.get("volume", 0) or 0),
                }
            )
        bars.sort(key=lambda item: item["at"])
        return bars


class ClientSimulationService:
    """Application service for safe FT client simulation sessions."""

    def __init__(
        self,
        workspace_root: Path | None = None,
        *,
        config: FtClientConfig,
        broker_factory: Callable[[FtClientConfig], BrokerAdapter] | None = None,
        market_context_provider: Callable[
            [list[str], dict[str, Any]], dict[str, Any]
        ]
        | None = None,
        clock: Callable[[], datetime] | None = None,
        enable_background_reconciliation: bool = False,
        background_interval_seconds: float | None = None,
        market_poll_interval_seconds: float = 60.0,
    ) -> None:
        self.strategy_service = StrategyService(workspace_root=workspace_root)
        self.workspace_root = self.strategy_service.workspace_root
        self.config = config
        self._custom_broker_factory = broker_factory
        self._market_context_provider = market_context_provider
        self._clock = clock or (lambda: datetime.now(ZoneInfo("Asia/Shanghai")))
        self._credentials: FtClientCredentials | None = None
        self._background_enabled = enable_background_reconciliation
        self._background_interval = (
            background_interval_seconds
            if background_interval_seconds is not None
            else config.poll_interval_seconds
        )
        self._market_poll_interval = max(market_poll_interval_seconds, 0.01)
        self._sessions: dict[tuple[str, str], ClientSimulationSession] = {}
        self._brokers: dict[tuple[str, str], BrokerAdapter] = {}
        self._worker_stops: dict[tuple[str, str], threading.Event] = {}
        self._worker_threads: dict[tuple[str, str], list[threading.Thread]] = {}
        self._lock = threading.RLock()

    @staticmethod
    def compute_intent_id(
        strategy_slug: str, strategy_version: int, trade_account: str, intent_key: str
    ) -> str:
        value = f"{strategy_slug}{strategy_version}{trade_account}{intent_key}"
        return hashlib.sha256(value.encode("utf-8")).hexdigest()

    def configure_connection(self, connection: FtConnectionInput) -> None:
        """Apply non-secret fields to memory and retain the password only as SecretStr."""
        values = self.config.model_dump()
        values.update(
            {
                "enabled": True,
                "base_url": connection.base_url,
                "confirmed_client_version": connection.confirmed_client_version,
                "password_transform": connection.password_transform,
                "allowed_simulation_accounts": connection.allowed_simulation_accounts,
                "allowed_symbols": connection.allowed_symbols,
                "symbol_mapping": connection.symbol_mapping,
                "allowed_algorithms": connection.allowed_algorithms,
                "external_id_max_length": connection.external_id_max_length,
                "external_id_scope_confirmed": connection.external_id_scope_confirmed,
            }
        )
        self.config = FtClientConfig.model_validate(values)
        self._credentials = FtClientCredentials(
            ft_account=connection.ft_account,
            password=connection.password,
            password_transform=connection.password_transform,
        )
        if self._background_enabled:
            self._resume_persisted_workers()

    def _make_broker(self) -> BrokerAdapter:
        if self._custom_broker_factory is not None:
            return self._custom_broker_factory(self.config)
        return FtClientBroker(self.config, credentials=self._credentials)

    def check_connection(
        self, connection: FtConnectionInput | None = None
    ) -> PreflightResult:
        if connection is not None:
            self.configure_connection(connection)
        request = ClientSimulationRequest(
            trade_account=self.config.allowed_simulation_accounts[0]
            if self.config.allowed_simulation_accounts
            else "",
            acknowledge_simulation=False,
        )
        checks: list[PreflightCheck] = []
        if not self.config.enabled:
            checks.append(self._check("configuration_error", False, "FT client is disabled."))
            return PreflightResult(ready=False, checks=checks)
        broker = self._make_broker()
        try:
            accounts = broker.connect()
        except (FtClientError, Exception) as exc:
            code = exc.code if isinstance(exc, FtClientError) else "client_unavailable"
            checks.append(self._check(code, False, str(exc)))
            return PreflightResult(ready=False, checks=checks)
        checks.append(self._check("client_connected", True, "FT client login succeeded."))
        selected = next(
            (item for item in accounts if item.trade_account == request.trade_account),
            None,
        )
        return PreflightResult(ready=True, checks=checks, account=selected)

    def list_accounts(self) -> list[FtAccount]:
        broker = self._make_broker()
        accounts = broker.connect()
        allowed = set(self.config.allowed_simulation_accounts)
        return [account for account in accounts if account.trade_account in allowed]

    def preflight(self, slug: str, request: ClientSimulationRequest) -> PreflightResult:
        return self._run_preflight(slug, request).result

    def _run_preflight(
        self, slug: str, request: ClientSimulationRequest
    ) -> _PreflightContext:
        checks: list[PreflightCheck] = []
        strategy = self.strategy_service.get_strategy(slug)
        strategy_dir = self.strategy_service.get_strategy_paths(slug).workspace
        config_data = self._load_strategy_config(strategy_dir)

        checks.append(
            self._check(
                "ft_client_enabled",
                self.config.enabled,
                "FT client connection is enabled."
                if self.config.enabled
                else "FT client connection is disabled.",
            )
        )
        version_ok = self._version_at_least(
            self.config.confirmed_client_version, self.config.min_client_version
        )
        checks.append(
            self._check(
                "client_version_verified" if version_ok else "client_version_unverified",
                version_ok,
                f"FT client version must be at least {self.config.min_client_version}.",
                {
                    "confirmed": self.config.confirmed_client_version,
                    "minimum": self.config.min_client_version,
                },
            )
        )
        strategy_ready = strategy.status in {
            StrategyStatus.BACKTESTED,
            StrategyStatus.PAPER_RUNNING,
            StrategyStatus.OPTIMIZED,
            StrategyStatus.ACTIVE,
        }
        checks.append(
            self._check(
                "strategy_backtested",
                strategy_ready,
                "Strategy must be backtested before client simulation.",
            )
        )
        compatible = self._strategy_has_intent_entry(strategy_dir)
        checks.append(
            self._check(
                "strategy_compatible" if compatible else "strategy_incompatible",
                compatible,
                "strategy.py must expose generate_intents(context).",
            )
        )
        acknowledged = request.execution_mode == "observe" or request.acknowledge_simulation
        checks.append(
            self._check(
                "simulation_acknowledged",
                acknowledged,
                "The selected account must be explicitly acknowledged as simulation.",
            )
        )
        route_ok = request.execution_route == "algorithm_parent"
        checks.append(
            self._check(
                "execution_route_supported" if route_ok else "algorithm_config_invalid",
                route_ok,
                "Only algorithm_parent is enabled; direct_order is disabled.",
            )
        )
        algorithm_ok = (
            request.algorithm.strategy_type in self.config.allowed_algorithms
            and not request.algorithm.reach_limit_continue
            and not request.algorithm.over_time_continue
        )
        checks.append(
            self._check(
                "algorithm_config_valid" if algorithm_ok else "algorithm_config_invalid",
                algorithm_ok,
                "Algorithm must be allow-listed and continuation flags must be false.",
            )
        )
        external_ok = bool(
            self.config.external_id_scope_confirmed
            and self.config.external_id_max_length
            and self.config.external_id_max_length >= 64
        )
        checks.append(
            self._check(
                "external_id_verified" if external_ok else "external_id_unverified",
                external_ok,
                "external_id must support the complete 64-character platform intent ID.",
            )
        )
        symbols = self._strategy_symbols(config_data)
        execution_universe = self.config.allowed_symbols
        unsupported_market = sorted(
            symbol
            for symbol in set([*execution_universe, *symbols])
            if not is_supported_cn_security(symbol)
        )
        checks.append(
            self._check(
                (
                    "execution_universe_supported"
                    if execution_universe and not unsupported_market
                    else "unsupported_market_symbol"
                ),
                bool(execution_universe) and not unsupported_market,
                "Execution universe must contain only tradable Shanghai/Shenzhen A shares or ETFs.",
                {
                    "unsupported": unsupported_market,
                    "allowed_symbols": execution_universe,
                },
            )
        )
        unsupported = sorted(set(symbols) - set(execution_universe))
        missing_mapping = sorted(
            symbol
            for symbol in execution_universe
            if not self.config.symbol_mapping.get(symbol)
        )
        checks.append(
            self._check(
                "symbols_allowed",
                not unsupported and bool(symbols) and not unsupported_market,
                "Strategy symbols must be in the FT simulation allow-list.",
                {"unsupported": unsupported, "symbols": symbols},
            )
        )
        market_symbols = self._strategy_market_symbols(config_data, symbols)
        market_context = self._market_context(config_data)
        if self._market_context_provider is not None:
            try:
                market_context = self._market_context_provider(market_symbols, config_data)
                market_ready = bool(
                    market_context.get("completed_bar_at")
                    and all(
                        market_context.get("bars_by_symbol", {}).get(symbol)
                        for symbol in market_symbols
                    )
                )
                checks.append(
                    self._check(
                        "market_data_ready" if market_ready else "market_data_unavailable",
                        market_ready,
                        "A common completed market-data bar must be available for every symbol.",
                    )
                )
            except Exception as exc:
                checks.append(
                    self._check(
                        "market_data_unavailable",
                        False,
                        f"Completed market data is unavailable: {exc}",
                    )
                )
        checks.append(
            self._check(
                "symbol_mapping_confirmed",
                not missing_mapping and bool(execution_universe),
                "Every allowed execution symbol needs an explicit FT client code mapping.",
                {"missing": missing_mapping},
            )
        )
        active = self._find_active_session(request.trade_account)
        checks.append(
            self._check(
                "account_session_available",
                active is None,
                "A trade account can have only one active client simulation session.",
                {"active_session_id": active.session_id if active else None},
            )
        )

        broker = self._make_broker()
        account: FtAccount | None = None
        health: AccountHealth | None = None
        funds: FundSnapshot | None = None
        positions: list[PositionSnapshot] = []
        monitoring: MonitoringSnapshot | None = None
        try:
            accounts = broker.connect()
            account = next(
                (item for item in accounts if item.trade_account == request.trade_account), None
            )
            allowed_account = bool(
                account and request.trade_account in self.config.allowed_simulation_accounts
            )
            checks.append(
                self._check(
                    "account_allowed" if allowed_account else "account_not_allowed",
                    allowed_account,
                    "Trade account must be in allowed_simulation_accounts.",
                )
            )
            account_logged_in = bool(account and account.login_status)
            checks.append(
                self._check(
                    "account_login_ok" if account_logged_in else "account_login_failed",
                    account_logged_in,
                    "Target account login_status must be true.",
                )
            )
            if account:
                health = broker.health().get(account.trade_account)
                engine_ready = bool(
                    health and health.login_status and health.order_engine_status
                )
                checks.append(
                    self._check(
                        "order_engine_ready" if engine_ready else "order_engine_unavailable",
                        engine_ready,
                        "Account and order engine must both be connected.",
                    )
                )
                funds = broker.get_funds(account.trade_account, account.broker_id)
                positions = broker.get_positions(account.trade_account, account.broker_id)
                fund_ok = funds.risk_equity > 0
                checks.append(
                    self._check(
                        "funds_ready" if fund_ok else "funds_unavailable",
                        fund_ok,
                        "A positive asset or balance is required for risk checks.",
                        {"diagnostics": funds.diagnostics},
                    )
                )
                monitoring = broker.get_monitoring(account.ft_account)
        except Exception as exc:
            code = exc.code if isinstance(exc, FtClientError) else "client_unavailable"
            checks.append(self._check(code, False, str(exc)))

        return _PreflightContext(
            result=PreflightResult(
                ready=all(item.passed for item in checks),
                checks=checks,
                account=account,
                health=health,
                funds=funds,
                positions=positions,
                monitoring=monitoring,
            ),
            broker=broker,
            account=account,
            market_context=market_context,
        )

    def create_session(
        self, slug: str, request: ClientSimulationRequest
    ) -> ClientSimulationSession:
        with self._lock:
            preflight = self._run_preflight(slug, request)
            failed = [item for item in preflight.result.checks if not item.passed]
            if failed:
                first = failed[0]
                incompatible = next(
                    (item for item in failed if item.code == "strategy_incompatible"), None
                )
                if incompatible:
                    raise ClientSimulationError(
                        incompatible.code, incompatible.message, {"checks": self._dump(failed)}
                    )
                raise ClientSimulationError(
                    "preflight_failed",
                    first.message,
                    {"checks": self._dump(failed)},
                )
            assert preflight.account is not None and preflight.result.funds is not None
            strategy = self.strategy_service.get_strategy(slug)
            strategy_dir = self.strategy_service.get_strategy_paths(slug).workspace
            session_id = f"sess_{uuid.uuid4().hex}"
            session = ClientSimulationSession(
                session_id=session_id,
                strategy_slug=slug,
                strategy_version=strategy.version,
                strategy_digest=strategy.content_digest,
                status="starting",
                execution_mode=request.execution_mode,
                execution_route=request.execution_route,
                ft_account=preflight.account.ft_account,
                trade_account=preflight.account.trade_account,
                broker_id=preflight.account.broker_id,
                account_nickname=preflight.account.nickname,
                account_login_status=bool(
                    preflight.result.health and preflight.result.health.login_status
                ),
                order_engine_status=bool(
                    preflight.result.health
                    and preflight.result.health.order_engine_status
                ),
                client_version=self.config.confirmed_client_version or "unverified",
                minimum_client_version=self.config.min_client_version,
                algorithm=request.algorithm,
                risk=request.risk,
                execution_window_start=request.execution_window_start,
                execution_window_end=request.execution_window_end,
                last_evaluated_bar_at=preflight.market_context.get(
                    "completed_bar_at"
                ),
                preflight=preflight.result.checks,
                funds=preflight.result.funds,
                positions=preflight.result.positions,
                monitoring=preflight.result.monitoring,
            )
            key = (slug, session_id)
            self._sessions[key] = session
            self._brokers[key] = preflight.broker
            self._initialize_session_files(slug, session_id)
            self._persist_account_snapshots(session)

            output = self._generate_intents(
                strategy_dir, session, market_context=preflight.market_context
            )
            for raw_intent in output.get("intents") or []:
                intent = self._normalize_intent(session, request, raw_intent)
                rejection = self._risk_rejection(session, intent)
                if rejection:
                    intent.status = "rejected"
                    intent.reason = rejection
                elif self._intent_seen(intent.intent_id):
                    intent.status = "rejected"
                    intent.reason = "duplicate_intent"
                else:
                    intent.status = "validated"
                session.intents.append(intent)
                self._append_jsonl(slug, session_id, "order_intents.jsonl", intent)

            self._write_strategy_state(
                slug, session_id, output.get("strategy_state") or {}
            )
            session.status = "running"
            session.updated_at = utc_now()
            self._append_event(session, "session_started", {"mode": request.execution_mode})
            if request.execution_mode == "auto":
                for intent in session.intents:
                    if intent.status == "validated" and (
                        self._market_context_provider is None
                        or self._inside_execution_window(session)
                    ):
                        self._submit_intent(session, intent)
            self._persist_session(session)
            self._start_background_workers(session)
            return session.model_copy(deep=True)

    def approve_intent(
        self, slug: str, session_id: str, intent_id: str
    ) -> ClientSimulationSession:
        with self._lock:
            session = self._get_mutable_session(slug, session_id)
            if session.execution_mode != "manual":
                raise ClientSimulationError(
                    "invalid_execution_mode", "Only manual sessions accept intent approval."
                )
            intent = next((item for item in session.intents if item.intent_id == intent_id), None)
            if intent is None:
                raise ClientSimulationError(
                    "intent_not_found", f"Intent {intent_id} was not found."
                )
            if intent.status != "validated":
                raise ClientSimulationError(
                    "intent_not_approvable", f"Intent is in {intent.status} state."
                )
            self._submit_intent(session, intent)
            self._persist_session(session)
            self._start_background_workers(session)
            return session.model_copy(deep=True)

    def reject_intent(
        self, slug: str, session_id: str, intent_id: str, reason: str = "user_rejected"
    ) -> ClientSimulationSession:
        with self._lock:
            session = self._get_mutable_session(slug, session_id)
            intent = next((item for item in session.intents if item.intent_id == intent_id), None)
            if intent is None:
                raise ClientSimulationError(
                    "intent_not_found", f"Intent {intent_id} was not found."
                )
            if intent.status != "validated":
                raise ClientSimulationError(
                    "intent_not_rejectable", f"Intent is in {intent.status} state."
                )
            intent.status = "rejected"
            intent.reason = reason
            self._append_jsonl(slug, session_id, "order_intents.jsonl", intent)
            self._append_event(
                session,
                "intent_rejected",
                {"intent_id": intent_id, "reason": reason},
            )
            self._persist_session(session)
            return session.model_copy(deep=True)

    def pause_session(self, slug: str, session_id: str) -> ClientSimulationSession:
        with self._lock:
            session = self._get_mutable_session(slug, session_id)
            session.status = "paused"
            session.updated_at = utc_now()
            self._append_event(session, "session_paused", {})
            self._persist_session(session)
            return session.model_copy(deep=True)

    def resume_session(self, slug: str, session_id: str) -> ClientSimulationSession:
        with self._lock:
            session = self._get_mutable_session(slug, session_id)
            request = ClientSimulationRequest(
                trade_account=session.trade_account,
                execution_mode=session.execution_mode,
                acknowledge_simulation=True,
                execution_route=session.execution_route,
                algorithm=session.algorithm,
                risk=session.risk,
            )
            result = self._run_preflight(slug, request).result
            failed = [
                item
                for item in result.checks
                if not item.passed and item.code != "account_session_available"
            ]
            if failed:
                raise ClientSimulationError(
                    "preflight_failed", failed[0].message, {"checks": self._dump(failed)}
                )
            session.status = "running"
            session.preflight = result.checks
            session.account_login_status = bool(
                result.health and result.health.login_status
            )
            session.order_engine_status = bool(
                result.health and result.health.order_engine_status
            )
            session.funds = result.funds
            session.positions = result.positions
            session.monitoring = result.monitoring
            session.updated_at = utc_now()
            self._append_event(session, "session_resumed", {})
            self._persist_account_snapshots(session)
            self._persist_session(session)
            return session.model_copy(deep=True)

    def stop_session(self, slug: str, session_id: str) -> ClientSimulationSession:
        with self._lock:
            session = self._get_mutable_session(slug, session_id)
            session.status = "stopping"
            broker = self._broker_for(session)
            active_ids = [
                order.parent_order_id
                for order in session.orders
                if order.parent_order_id and order.normalized_status not in _TERMINAL_ORDER_STATUSES
            ]
            if active_ids:
                broker.cancel_orders(active_ids)
                self._append_event(session, "cancel_requested", {"parent_order_ids": active_ids})
            session.status = "stopped"
            session.updated_at = utc_now()
            self._append_event(session, "session_stopped", {})
            self._persist_session(session)
            self._stop_background_workers(session)
            return session.model_copy(deep=True)

    def shutdown(self) -> None:
        """Stop all daemon reconciliation/event workers owned by this service."""
        for event in list(self._worker_stops.values()):
            event.set()
        for threads in list(self._worker_threads.values()):
            for thread in threads:
                thread.join(timeout=1)
        self._worker_stops.clear()
        self._worker_threads.clear()

    def reconcile(self, slug: str, session_id: str) -> ClientSimulationSession:
        with self._lock:
            session = self._get_mutable_session(slug, session_id)
            broker = self._broker_for(session)
            health = broker.health().get(session.trade_account)
            session.account_login_status = bool(health and health.login_status)
            session.order_engine_status = bool(health and health.order_engine_status)
            session.funds = broker.get_funds(session.trade_account, session.broker_id)
            session.positions = broker.get_positions(session.trade_account, session.broker_id)
            session.monitoring = broker.get_monitoring(session.ft_account)
            orders = broker.get_orders(session.trade_account, session.broker_id)
            own_external_ids = {intent.intent_id for intent in session.intents}
            session.orders = [order for order in orders if order.external_id in own_external_ids]
            children: list[ChildOrder] = []
            for order in session.orders:
                children.extend(broker.get_child_orders(order.parent_order_id))
            session.child_orders = children
            session.updated_at = utc_now()
            self._persist_account_snapshots(session)
            for order in session.orders:
                self._append_jsonl(slug, session_id, "broker_orders.jsonl", order)
            for child in children:
                self._append_jsonl(slug, session_id, "child_orders.jsonl", child)
            self._append_jsonl(
                slug,
                session_id,
                "reconciliation.jsonl",
                {"at": session.updated_at, "order_count": len(session.orders)},
            )
            self._persist_session(session)
            return session.model_copy(deep=True)

    def evaluate_latest_bar(
        self, slug: str, session_id: str
    ) -> ClientSimulationSession:
        """Evaluate one newly completed market bar, at most once per session."""
        with self._lock:
            session = self._get_mutable_session(slug, session_id)
            if session.status != "running" or self._market_context_provider is None:
                return session.model_copy(deep=True)
            strategy_dir = self.strategy_service.get_strategy_paths(slug).workspace
            strategy_config = self._load_strategy_config(strategy_dir)
            execution_symbols = self._strategy_symbols(strategy_config)
            symbols = self._strategy_market_symbols(strategy_config, execution_symbols)
            market_context = self._market_context_provider(symbols, strategy_config)
            completed_bar_at = str(market_context.get("completed_bar_at") or "")
            if not completed_bar_at:
                return session.model_copy(deep=True)
            if completed_bar_at == session.last_evaluated_bar_at:
                self._submit_ready_auto_intents(session)
                self._persist_session(session)
                return session.model_copy(deep=True)
            if (
                session.last_evaluated_bar_at
                and completed_bar_at < session.last_evaluated_bar_at
            ):
                session.status = "needs_attention"
                session.latest_error = "market_data_time_reversal"
                self._append_event(
                    session,
                    "market_data_time_reversal",
                    {
                        "previous": session.last_evaluated_bar_at,
                        "current": completed_bar_at,
                    },
                )
                self._persist_session(session)
                return session.model_copy(deep=True)

            broker = self._broker_for(session)
            health = broker.health().get(session.trade_account)
            session.account_login_status = bool(health and health.login_status)
            session.order_engine_status = bool(
                health and health.order_engine_status
            )
            session.funds = broker.get_funds(session.trade_account, session.broker_id)
            session.positions = broker.get_positions(
                session.trade_account, session.broker_id
            )
            request = ClientSimulationRequest(
                trade_account=session.trade_account,
                execution_mode=session.execution_mode,
                acknowledge_simulation=True,
                execution_route=session.execution_route,
                execution_window_start=session.execution_window_start,
                execution_window_end=session.execution_window_end,
                algorithm=session.algorithm,
                risk=session.risk,
            )
            output = self._generate_intents(
                strategy_dir, session, market_context=market_context
            )
            for raw_intent in output.get("intents") or []:
                intent = self._normalize_intent(session, request, raw_intent)
                rejection = self._risk_rejection(session, intent)
                if rejection:
                    intent.status = "rejected"
                    intent.reason = rejection
                elif self._intent_seen(intent.intent_id):
                    intent.status = "rejected"
                    intent.reason = "duplicate_intent"
                else:
                    intent.status = "validated"
                session.intents.append(intent)
                self._append_jsonl(
                    slug, session_id, "order_intents.jsonl", intent
                )
                if (
                    session.execution_mode == "auto"
                    and intent.status == "validated"
                    and self._inside_execution_window(session)
                ):
                    self._submit_intent(session, intent)
            self._write_strategy_state(
                slug, session_id, output.get("strategy_state") or {}
            )
            session.last_evaluated_bar_at = completed_bar_at
            session.updated_at = utc_now()
            self._append_event(
                session,
                "strategy_evaluated",
                {"completed_bar_at": completed_bar_at},
            )
            self._persist_account_snapshots(session)
            self._persist_session(session)
            return session.model_copy(deep=True)

    def get_session(self, slug: str, session_id: str) -> ClientSimulationSession:
        with self._lock:
            session = self._get_mutable_session(slug, session_id)
            return session.model_copy(deep=True)

    def list_sessions(self, slug: str) -> list[ClientSimulationSession]:
        root = self._sessions_root(slug)
        if not root.exists():
            return []
        sessions: list[ClientSimulationSession] = []
        for path in sorted(root.glob("sess_*/session.json"), reverse=True):
            sessions.append(self.get_session(slug, path.parent.name))
        return sessions

    def get_events(self, slug: str, session_id: str) -> list[dict[str, Any]]:
        path = self._session_dir(slug, session_id) / "events.jsonl"
        if not path.exists():
            return []
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]

    def get_account_snapshot(self, slug: str, session_id: str) -> dict[str, Any]:
        session = self.get_session(slug, session_id)
        return {
            "trade_account": session.trade_account,
            "account_nickname": session.account_nickname,
            "funds": self._dump(session.funds),
            "positions": self._dump(session.positions),
            "monitoring": self._dump(session.monitoring),
        }

    def _submit_intent(self, session: ClientSimulationSession, intent: OrderIntent) -> None:
        broker = self._broker_for(session)
        intent.status = "submitting"
        self._append_jsonl(
            session.strategy_slug, session.session_id, "order_intents.jsonl", intent
        )
        try:
            order = broker.submit_order(intent)
        except FtClientError as exc:
            intent.status = "submission_unknown" if exc.retryable else "failed"
            session.latest_error = str(exc)
            self._append_jsonl(
                session.strategy_slug, session.session_id, "order_intents.jsonl", intent
            )
            if exc.retryable:
                candidates = broker.get_orders(session.trade_account, session.broker_id)
                matched = next(
                    (order for order in candidates if order.external_id == intent.intent_id), None
                )
                if matched:
                    session.orders.append(matched)
                    intent.status = matched.normalized_status
                    self._append_jsonl(
                        session.strategy_slug,
                        session.session_id,
                        "broker_orders.jsonl",
                        matched,
                    )
                    self._append_event(
                        session,
                        "submission_reconciled",
                        {"intent_id": intent.intent_id, "parent_order_id": matched.parent_order_id},
                    )
            return
        session.orders.append(order)
        intent.status = order.normalized_status
        self._append_jsonl(
            session.strategy_slug, session.session_id, "order_intents.jsonl", intent
        )
        self._append_jsonl(
            session.strategy_slug, session.session_id, "broker_orders.jsonl", order
        )
        self._append_event(
            session,
            "order_submitted",
            {"intent_id": intent.intent_id, "parent_order_id": order.parent_order_id},
        )

    def _submit_ready_auto_intents(self, session: ClientSimulationSession) -> None:
        if session.execution_mode != "auto" or not self._inside_execution_window(session):
            return
        for intent in session.intents:
            if intent.status == "validated":
                self._submit_intent(session, intent)

    def _inside_execution_window(self, session: ClientSimulationSession) -> bool:
        current = self._clock().astimezone(ZoneInfo("Asia/Shanghai")).strftime("%H%M%S")
        return session.execution_window_start <= current <= session.execution_window_end

    def _normalize_intent(
        self,
        session: ClientSimulationSession,
        request: ClientSimulationRequest,
        raw: dict[str, Any],
    ) -> OrderIntent:
        symbol = str(raw.get("symbol") or "")
        intent_key = str(raw.get("intent_key") or "")
        if not intent_key:
            raise ClientSimulationError("strategy_error", "Strategy intent_key cannot be empty.")
        window = raw.get("execution_window") or {}
        return OrderIntent(
            intent_id=self.compute_intent_id(
                session.strategy_slug,
                session.strategy_version,
                session.trade_account,
                intent_key,
            ),
            intent_key=intent_key,
            session_id=session.session_id,
            symbol=symbol,
            broker_symbol=self.config.symbol_mapping.get(symbol, ""),
            side=raw.get("side"),
            quantity=int(raw.get("quantity") or 0),
            signal_price=(
                float(raw["signal_price"])
                if raw.get("signal_price") is not None
                else None
            ),
            reason=str(raw.get("reason") or ""),
            trade_account=session.trade_account,
            broker_id=session.broker_id,
            execution_window_start=str(window.get("start") or request.execution_window_start),
            execution_window_end=str(window.get("end") or request.execution_window_end),
            algorithm=request.algorithm,
        )

    def _risk_rejection(
        self, session: ClientSimulationSession, intent: OrderIntent
    ) -> str | None:
        if not session.account_login_status or not session.order_engine_status:
            return "order_engine_unavailable"
        if not is_supported_cn_security(intent.symbol):
            return "unsupported_market_symbol"
        if intent.symbol not in self.config.allowed_symbols or not intent.broker_symbol:
            return "symbol_not_allowed"
        if intent.quantity <= 0:
            return "quantity_invalid"
        if intent.quantity % 100:
            return "quantity_not_board_lot"
        if not session.funds or session.funds.risk_equity <= 0:
            return "funds_unavailable"
        if intent.side == "sell":
            available = sum(
                item.available_volume
                for item in session.positions
                if item.stock_code == intent.broker_symbol
            )
            if intent.quantity > available:
                return "position_unavailable"
            return None
        if intent.signal_price is None or intent.signal_price <= 0:
            return "signal_price_missing_for_risk"
        estimated_amount = intent.quantity * intent.signal_price
        if estimated_amount > session.funds.available:
            return "insufficient_available_cash"
        risk_equity = session.funds.risk_equity
        if estimated_amount > risk_equity * session.risk.max_order_pct / 100:
            return "max_order_exceeded"
        position_value = sum(float(item.raw.get("market_value") or 0) for item in session.positions)
        symbol_value = sum(
            float(item.raw.get("market_value") or 0)
            for item in session.positions
            if item.stock_code == intent.broker_symbol
        )
        symbol_limit = risk_equity * session.risk.max_symbol_position_pct / 100
        if symbol_value + estimated_amount > symbol_limit:
            return "max_symbol_position_exceeded"
        total_limit = risk_equity * session.risk.max_total_position_pct / 100
        if position_value + estimated_amount > total_limit:
            return "max_total_position_exceeded"
        return None

    def _generate_intents(
        self,
        strategy_dir: Path,
        session: ClientSimulationSession,
        *,
        market_context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        module = self._load_strategy_module(strategy_dir)
        generate_intents = getattr(module, "generate_intents", None)
        if not callable(generate_intents):
            raise ClientSimulationError(
                "strategy_incompatible", "strategy.py does not expose generate_intents(context)."
            )
        strategy_config = self._load_strategy_config(strategy_dir)
        context = {
            "session": {
                "session_id": session.session_id,
                "mode": "ft_client_simulation",
                "execution_mode": session.execution_mode,
                "now": utc_now(),
            },
            "market": market_context or self._market_context(strategy_config),
            "account": {
                "trade_account": session.trade_account,
                "available_cash": session.funds.available if session.funds else 0,
                "risk_equity": session.funds.risk_equity if session.funds else 0,
                "positions": self._dump(session.positions),
            },
            "strategy_state": self._read_strategy_state(
                session.strategy_slug, session.session_id
            ),
            "config": strategy_config,
        }
        try:
            result = generate_intents(context)
        except Exception as exc:
            raise ClientSimulationError(
                "strategy_error", f"generate_intents failed: {exc}"
            ) from exc
        if not isinstance(result, dict) or not isinstance(result.get("intents", []), list):
            raise ClientSimulationError(
                "strategy_error", "generate_intents must return a dict containing an intents list."
            )
        return result

    def _strategy_has_intent_entry(self, strategy_dir: Path) -> bool:
        try:
            module = self._load_strategy_module(strategy_dir)
            return callable(getattr(module, "generate_intents", None))
        except Exception:
            return False

    def _load_strategy_module(self, strategy_dir: Path) -> ModuleType:
        path = strategy_dir / "strategy.py"
        if not path.exists():
            raise ClientSimulationError("strategy_incompatible", "strategy.py is missing.")
        digest = hashlib.sha256(str(path).encode()).hexdigest()
        name = f"autostrategy_client_{digest}_{uuid.uuid4().hex}"
        spec = importlib.util.spec_from_file_location(name, path)
        if spec is None or spec.loader is None:
            raise ClientSimulationError("strategy_incompatible", "strategy.py cannot be loaded.")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def _load_strategy_config(self, strategy_dir: Path) -> dict[str, Any]:
        path = strategy_dir / "config.yaml"
        if not path.exists():
            return {}
        loaded = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        return loaded if isinstance(loaded, dict) else {}

    @staticmethod
    def _strategy_symbols(config_data: dict[str, Any]) -> list[str]:
        symbols = config_data.get("symbols") or []
        if isinstance(symbols, str):
            return [symbols]
        return [str(item) for item in symbols]

    @staticmethod
    def _strategy_market_symbols(
        config_data: dict[str, Any], execution_symbols: list[str]
    ) -> list[str]:
        candidates: list[Any] = [*execution_symbols]
        for key in ("benchmark", "benchmarks"):
            value = config_data.get(key)
            if isinstance(value, list):
                candidates.extend(value)
            elif value:
                candidates.append(value)
        universe = ((config_data.get("data") or {}).get("universe") or {})
        for key in ("core_index_symbol", "benchmark", "benchmark_symbol"):
            if universe.get(key):
                candidates.append(universe[key])
        return list(
            dict.fromkeys(str(item).strip().upper() for item in candidates if str(item).strip())
        )

    @staticmethod
    def _market_context(config_data: dict[str, Any]) -> dict[str, Any]:
        bars = config_data.get("feed_bars") or []
        completed = None
        if bars:
            completed = bars[-1].get("at") or bars[-1].get("date")
        return {
            "bars_by_symbol": {},
            "history_by_symbol": {},
            "completed_bar_at": completed,
        }

    @staticmethod
    def _version_at_least(value: str | None, minimum: str) -> bool:
        if not value:
            return False
        try:
            parsed = tuple(int(part) for part in value.split("."))
            required = tuple(int(part) for part in minimum.split("."))
        except ValueError:
            return False
        length = max(len(parsed), len(required))
        return parsed + (0,) * (length - len(parsed)) >= required + (0,) * (length - len(required))

    def _find_active_session(self, trade_account: str) -> ClientSimulationSession | None:
        for session in self._sessions.values():
            if (
                session.trade_account == trade_account
                and session.status in _ACTIVE_SESSION_STATUSES
            ):
                return session
        for path in self.workspace_root.glob("*/paper_run/client_sessions/sess_*/session.json"):
            try:
                session = ClientSimulationSession.model_validate_json(
                    path.read_text(encoding="utf-8")
                )
            except (OSError, ValueError):
                continue
            if (
                session.trade_account == trade_account
                and session.status in _ACTIVE_SESSION_STATUSES
            ):
                return session
        return None

    def _intent_seen(self, intent_id: str) -> bool:
        pattern = "*/paper_run/client_sessions/sess_*/order_intents.jsonl"
        for path in self.workspace_root.glob(pattern):
            try:
                lines = path.read_text(encoding="utf-8").splitlines()
            except OSError:
                continue
            for line in lines:
                try:
                    value = json.loads(line)
                except ValueError:
                    continue
                if value.get("intent_id") == intent_id and value.get("status") != "rejected":
                    return True
        return False

    def _broker_for(self, session: ClientSimulationSession) -> BrokerAdapter:
        key = (session.strategy_slug, session.session_id)
        broker = self._brokers.get(key)
        if broker is None:
            broker = self._make_broker()
            broker.connect()
            self._brokers[key] = broker
        return broker

    def _start_background_workers(self, session: ClientSimulationSession) -> None:
        if not self._background_enabled or session.status not in _ACTIVE_SESSION_STATUSES:
            return
        key = (session.strategy_slug, session.session_id)
        if key in self._worker_stops:
            return
        stop_event = threading.Event()
        self._worker_stops[key] = stop_event

        def poll() -> None:
            last_market_poll = 0.0
            while not stop_event.wait(self._background_interval):
                try:
                    current = self.get_session(*key)
                    if current.status not in _ACTIVE_SESSION_STATUSES:
                        return
                    self.reconcile(*key)
                    now_monotonic = time.monotonic()
                    if (
                        self._market_context_provider is not None
                        and now_monotonic - last_market_poll
                        >= self._market_poll_interval
                    ):
                        self.evaluate_latest_bar(*key)
                        last_market_poll = now_monotonic
                except Exception as exc:
                    with self._lock:
                        current = self._sessions.get(key)
                        if current is not None:
                            current.latest_error = str(exc)
                            self._append_event(
                                current,
                                "reconciliation_error",
                                {"code": getattr(exc, "code", "reconciliation_error")},
                            )
                            self._persist_session(current)

        poll_thread = threading.Thread(
            target=poll,
            name=f"ft-poll-{session.session_id}",
            daemon=True,
        )
        threads = [poll_thread]
        broker = self._broker_for(session)
        if callable(getattr(broker, "stream_events", None)):
            websocket_thread = threading.Thread(
                target=lambda: asyncio.run(
                    self._consume_broker_events(
                        session.strategy_slug,
                        session.session_id,
                        stop_event,
                    )
                ),
                name=f"ft-ws-{session.session_id}",
                daemon=True,
            )
            threads.append(websocket_thread)
        self._worker_threads[key] = threads
        for thread in threads:
            thread.start()

    async def _consume_broker_events(
        self, slug: str, session_id: str, stop_event: threading.Event
    ) -> None:
        key = (slug, session_id)
        session = self._sessions.get(key)
        if session is None:
            return
        broker = self._broker_for(session)
        stream = getattr(broker, "stream_events", None)
        if not callable(stream):
            return
        try:
            async for event in stream():
                if stop_event.is_set():
                    return
                with self._lock:
                    current = self._sessions.get(key)
                    if current is None:
                        return
                    self._append_event(
                        current,
                        "broker_event",
                        {"topic": event.topic, "data": self._dump(event.data)},
                    )
                if event.topic in {"Mudan", "Zidan", "Trade"}:
                    self.reconcile(slug, session_id)
        except Exception as exc:
            if stop_event.is_set():
                return
            with self._lock:
                current = self._sessions.get(key)
                if current is not None:
                    current.latest_error = str(exc)
                    self._append_event(
                        current,
                        "websocket_error",
                        {"code": getattr(exc, "code", "websocket_error")},
                    )
                    self._persist_session(current)

    def _stop_background_workers(self, session: ClientSimulationSession) -> None:
        key = (session.strategy_slug, session.session_id)
        event = self._worker_stops.pop(key, None)
        if event is not None:
            event.set()
        self._worker_threads.pop(key, None)

    def _resume_persisted_workers(self) -> None:
        for path in self.workspace_root.glob("*/paper_run/client_sessions/sess_*/session.json"):
            slug = path.parents[3].name
            try:
                session = self.get_session(slug, path.parent.name)
            except ClientSimulationError:
                continue
            if session.status in _ACTIVE_SESSION_STATUSES:
                self._start_background_workers(self._get_mutable_session(slug, session.session_id))

    def _get_mutable_session(
        self, slug: str, session_id: str
    ) -> ClientSimulationSession:
        key = (slug, session_id)
        session = self._sessions.get(key)
        if session is not None:
            return session
        path = self._session_dir(slug, session_id) / "session.json"
        if not path.exists():
            raise ClientSimulationError(
                "session_not_found", f"Client simulation session {session_id} was not found."
            )
        session = ClientSimulationSession.model_validate_json(path.read_text(encoding="utf-8"))
        if session.status in _ACTIVE_SESSION_STATUSES and not self.config.auto_resume:
            session.status = "paused"
            session.updated_at = utc_now()
            self._atomic_json(path, session)
        self._sessions[key] = session
        return session

    def _sessions_root(self, slug: str) -> Path:
        workspace = self.strategy_service.get_strategy_paths(slug).workspace
        return workspace / "paper_run" / "client_sessions"

    def _session_dir(self, slug: str, session_id: str) -> Path:
        return self._sessions_root(slug) / session_id

    def _initialize_session_files(self, slug: str, session_id: str) -> None:
        root = self._session_dir(slug, session_id)
        (root / "logs").mkdir(parents=True, exist_ok=True)
        for name in _SESSION_FILES:
            (root / name).touch(exist_ok=True)
        (root / "logs" / "session.log").touch(exist_ok=True)

    def _persist_session(self, session: ClientSimulationSession) -> None:
        session.updated_at = utc_now()
        path = self._session_dir(session.strategy_slug, session.session_id) / "session.json"
        self._atomic_json(path, session)

    def _persist_account_snapshots(self, session: ClientSimulationSession) -> None:
        self._append_jsonl(
            session.strategy_slug,
            session.session_id,
            "account_snapshots.jsonl",
            {
                "collected_at": utc_now(),
                "funds": self._dump(session.funds),
                "positions": self._dump(session.positions),
            },
        )
        if session.monitoring:
            self._append_jsonl(
                session.strategy_slug,
                session.session_id,
                "monitoring_snapshots.jsonl",
                session.monitoring,
            )

    def _append_event(
        self, session: ClientSimulationSession, event_type: str, data: dict[str, Any]
    ) -> None:
        self._append_jsonl(
            session.strategy_slug,
            session.session_id,
            "events.jsonl",
            {"at": utc_now(), "type": event_type, "data": data},
        )

    def _append_jsonl(
        self, slug: str, session_id: str, filename: str, value: Any
    ) -> None:
        path = self._session_dir(slug, session_id) / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(self._dump(value), ensure_ascii=False, separators=(",", ":")))
            file.write("\n")
            file.flush()
            os.fsync(file.fileno())

    def _write_strategy_state(
        self, slug: str, session_id: str, state: dict[str, Any]
    ) -> None:
        self._atomic_json(self._session_dir(slug, session_id) / "strategy_state.json", state)

    def _read_strategy_state(self, slug: str, session_id: str) -> dict[str, Any]:
        path = self._session_dir(slug, session_id) / "strategy_state.json"
        if not path.exists():
            return {}
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}
        return value if isinstance(value, dict) else {}

    @classmethod
    def _atomic_json(cls, path: Path, value: Any) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(cls._dump(value), ensure_ascii=False, indent=2)
        with tempfile.NamedTemporaryFile(
            "w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", delete=False
        ) as file:
            file.write(payload)
            file.flush()
            os.fsync(file.fileno())
            temporary = Path(file.name)
        os.replace(temporary, path)

    @staticmethod
    def _dump(value: Any) -> Any:
        if isinstance(value, BaseModel):
            return value.model_dump(mode="json")
        if isinstance(value, list):
            return [ClientSimulationService._dump(item) for item in value]
        if isinstance(value, dict):
            return {key: ClientSimulationService._dump(item) for key, item in value.items()}
        return value

    @staticmethod
    def _check(
        code: str,
        passed: bool,
        message: str,
        details: dict[str, Any] | None = None,
    ) -> PreflightCheck:
        return PreflightCheck(code=code, passed=passed, message=message, details=details or {})
