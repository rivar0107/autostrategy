"""Normalized broker models shared by simulation services."""

from __future__ import annotations

import re
from datetime import UTC, datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, SecretStr, field_validator

NormalizedOrderStatus = Literal[
    "created",
    "validated",
    "rejected",
    "submitting",
    "submission_unknown",
    "submitted",
    "working",
    "pause_pending",
    "paused",
    "partially_filled",
    "cancel_pending",
    "stopping",
    "completed",
    "filled",
    "cancelled",
    "stopped",
    "expired",
    "residual",
    "failed",
    "unknown",
]

_PARENT_STATUS: dict[int, NormalizedOrderStatus] = {
    0: "submitted",
    1: "working",
    2: "paused",
    3: "completed",
    4: "cancelled",
    5: "expired",
    6: "failed",
    7: "stopping",
    8: "cancel_pending",
    9: "pause_pending",
    10: "stopped",
    11: "residual",
}
_ALGORITHM_CHILD_STATUS: dict[int, NormalizedOrderStatus] = {
    0: "submitted",
    1: "working",
    2: "partially_filled",
    3: "cancelled",
    4: "filled",
    5: "cancelled",
    6: "failed",
    7: "failed",
    8: "expired",
    9: "failed",
}
_DIRECT_ORDER_STATUS: dict[int, NormalizedOrderStatus] = {
    0: "submitted",
    1: "working",
    2: "partially_filled",
    3: "filled",
    4: "cancelled",
    5: "failed",
    6: "cancel_pending",
}
_PARAM_KEY = re.compile(r"^[A-Za-z][A-Za-z0-9_]*$")
_TIME = re.compile(r"^\d{6}$")


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def normalize_parent_status(raw_status: int) -> NormalizedOrderStatus:
    return _PARENT_STATUS.get(raw_status, "unknown")


def normalize_algorithm_child_status(raw_status: int) -> NormalizedOrderStatus:
    return _ALGORITHM_CHILD_STATUS.get(raw_status, "unknown")


def normalize_direct_order_status(raw_status: int) -> NormalizedOrderStatus:
    return _DIRECT_ORDER_STATUS.get(raw_status, "unknown")


class BrokerModel(BaseModel):
    model_config = ConfigDict(extra="ignore")


class FtAccount(BrokerModel):
    ft_account: str
    ft_account_name: str = ""
    broker_id: str
    broker_name: str = ""
    trade_account: str
    nickname: str = ""
    login_status: bool


class FtClientCredentials(BrokerModel):
    """Ephemeral credentials accepted from the local UI and never serialized to artifacts."""

    ft_account: str
    password: SecretStr
    password_transform: Literal["plain", "md5_32_lower"] = "plain"

    @field_validator("ft_account")
    @classmethod
    def validate_ft_account(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("FT account cannot be blank.")
        return stripped


class AccountHealth(BrokerModel):
    trade_account: str
    name: str = ""
    login_status: bool
    order_engine_status: bool
    collected_at: str = Field(default_factory=utc_now)


class FundSnapshot(BrokerModel):
    trade_account: str
    balance: float = 0.0
    asset: float = 0.0
    available: float = 0.0
    frozen: float = 0.0
    profit: float = 0.0
    risk_equity: float = 0.0
    diagnostics: list[str] = Field(default_factory=list)
    collected_at: str = Field(default_factory=utc_now)
    raw: dict[str, Any] = Field(default_factory=dict)


class PositionSnapshot(BrokerModel):
    trade_account: str
    stock_code: str
    total_volume: int = 0
    available_volume: int = 0
    locked_volume: int = 0
    in_transit_volume: int = 0
    exchange_id: int | None = None
    collected_at: str = Field(default_factory=utc_now)
    raw: dict[str, Any] = Field(default_factory=dict)


class AlgorithmConfig(BrokerModel):
    strategy_type: str = "TWAP"
    params: dict[str, str | int | float | bool] = Field(default_factory=dict)
    reach_limit_continue: bool = False
    over_time_continue: bool = False

    @field_validator("strategy_type")
    @classmethod
    def validate_strategy_type(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped or any(char in stripped for char in ";=\r\n"):
            raise ValueError("Invalid algorithm strategy_type.")
        return stripped

    @field_validator("params")
    @classmethod
    def validate_params(
        cls, values: dict[str, str | int | float | bool]
    ) -> dict[str, str | int | float | bool]:
        for key, value in values.items():
            if not _PARAM_KEY.fullmatch(key):
                raise ValueError(f"Invalid algorithm parameter name: {key!r}.")
            if isinstance(value, str) and any(char in value for char in ";=\r\n"):
                raise ValueError(f"Invalid algorithm parameter value for {key!r}.")
        return values

    def serialize_params(self) -> str:
        def render(value: str | int | float | bool) -> str:
            if isinstance(value, bool):
                return str(value).lower()
            return str(value)

        return ";".join(f"{key}={render(self.params[key])}" for key in sorted(self.params))


class OrderIntent(BrokerModel):
    intent_id: str
    intent_key: str
    session_id: str
    symbol: str
    broker_symbol: str
    side: Literal["buy", "sell"]
    quantity: int = Field(gt=0)
    signal_price: float | None = None
    reason: str = ""
    trade_account: str
    broker_id: str
    execution_window_start: str
    execution_window_end: str
    algorithm: AlgorithmConfig = Field(default_factory=AlgorithmConfig)
    status: NormalizedOrderStatus = "created"
    created_at: str = Field(default_factory=utc_now)

    @field_validator("execution_window_start", "execution_window_end")
    @classmethod
    def validate_execution_time(cls, value: str) -> str:
        if not _TIME.fullmatch(value):
            raise ValueError("Execution time must use HHMMSS.")
        hours, minutes, seconds = int(value[:2]), int(value[2:4]), int(value[4:])
        if hours > 23 or minutes > 59 or seconds > 59:
            raise ValueError("Execution time is invalid.")
        return value


class BrokerOrder(BrokerModel):
    parent_order_id: str
    external_id: str
    trade_account: str
    basket_name: str = ""
    stock_code: str = ""
    order_volume: int = 0
    trade_volume: int = 0
    raw_status: int
    raw_status_msg: str = ""
    normalized_status: NormalizedOrderStatus
    collected_at: str = Field(default_factory=utc_now)
    raw: dict[str, Any] = Field(default_factory=dict)


class ChildOrder(BrokerModel):
    child_order_id: str
    parent_order_id: str
    trade_account: str
    stock_code: str = ""
    order_volume: int = 0
    trade_volume: int = 0
    trade_price: float = 0.0
    raw_status: int
    raw_status_msg: str = ""
    normalized_status: NormalizedOrderStatus
    collected_at: str = Field(default_factory=utc_now)
    raw: dict[str, Any] = Field(default_factory=dict)


class MonitoringMetric(BrokerModel):
    account_id: str
    trade_account: str
    basket_id: str | None = None
    basket_name: str | None = None
    plan_buy: float = 0.0
    plan_sale: float = 0.0
    trade_buy: float = 0.0
    trade_sale: float = 0.0
    buy_rate: float = 0.0
    sale_rate: float = 0.0
    exposure: float = 0.0
    cancel_rate: float = 0.0
    total_rate: float = 0.0
    error_rate: float = 0.0


class MonitoringSnapshot(BrokerModel):
    ft_account: str
    trade_accounts: list[MonitoringMetric] = Field(default_factory=list)
    baskets: list[MonitoringMetric] = Field(default_factory=list)
    diagnostics: list[str] = Field(default_factory=list)
    collected_at: str = Field(default_factory=utc_now)
    raw: dict[str, Any] = Field(default_factory=dict)


class BrokerEvent(BrokerModel):
    topic: str
    data: dict[str, Any] | int | float | str | None
    received_at: str = Field(default_factory=utc_now)
