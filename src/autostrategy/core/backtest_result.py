"""Validated contracts for successful backtest results."""

from __future__ import annotations

import math
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


def _assert_finite(value: Any, path: str = "result") -> None:
    """Reject non-finite numbers anywhere in a persisted result."""
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"{path} must contain only finite numbers")
    if isinstance(value, dict):
        for key, item in value.items():
            _assert_finite(item, f"{path}.{key}")
    elif isinstance(value, list | tuple):
        for index, item in enumerate(value):
            _assert_finite(item, f"{path}[{index}]")


class EquityPoint(BaseModel):
    """One point on the strategy equity curve."""

    model_config = ConfigDict(extra="forbid")

    date: str
    equity: float = Field(ge=0)

    @field_validator("equity")
    @classmethod
    def validate_finite_equity(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("equity must be finite")
        return value


class TradeRecord(BaseModel):
    """Normalized executed trade record."""

    model_config = ConfigDict(extra="allow")

    date: str
    symbol: str
    action: Literal["buy", "sell"]
    price: float = Field(gt=0)
    quantity: float = Field(gt=0)
    cost: float | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def validate_finite_values(self) -> TradeRecord:
        _assert_finite(self.model_dump())
        return self


class BacktestMetrics(BaseModel):
    """Required metrics and optional evidence emitted by a strategy backtest."""

    model_config = ConfigDict(extra="allow")

    annual_return: float
    max_drawdown: float = Field(ge=0)
    sharpe: float
    win_rate: float = Field(ge=0, le=100)
    profit_loss_ratio: float = Field(ge=0)
    total_trades: int = Field(ge=1)
    initial_cash: float | None = Field(default=None, gt=0)
    final_value: float | None = Field(default=None, ge=0)
    total_return: float | None = None
    equity_curve: list[EquityPoint] = Field(default_factory=list)
    trades: list[TradeRecord] = Field(default_factory=list)
    benchmark: dict[str, Any] | None = None
    out_of_sample: dict[str, Any] | None = None

    @model_validator(mode="before")
    @classmethod
    def normalize_legacy_trade_records(cls, value: Any) -> Any:
        if isinstance(value, dict) and "trades" not in value and "trade_records" in value:
            value = dict(value)
            value["trades"] = value["trade_records"]
        return value

    @model_validator(mode="after")
    def validate_finite_values(self) -> BacktestMetrics:
        _assert_finite(self.model_dump())
        return self


class ResearchQuality(BaseModel):
    """Evidence available for judging whether a backtest is research-ready."""

    trade_sample: Literal["insufficient", "limited", "adequate"]
    has_equity_curve: bool
    has_trade_records: bool
    has_benchmark: bool
    has_out_of_sample: bool
    warnings: list[str] = Field(default_factory=list)


class BacktestWorkflowResult(BaseModel):
    """Strict shape of a successful backtest workflow response."""

    backtest: BacktestMetrics
    score: float = Field(ge=0, le=100)
    criteria: list[dict[str, Any]]
    diagnostics: list[dict[str, Any]]
    research_quality: ResearchQuality

    @model_validator(mode="before")
    @classmethod
    def add_legacy_research_quality(cls, value: Any) -> Any:
        """Keep previously persisted successful results readable."""
        if isinstance(value, dict) and "research_quality" not in value:
            backtest = value.get("backtest")
            if isinstance(backtest, dict):
                value = dict(value)
                value["research_quality"] = assess_research_quality(backtest).model_dump()
        return value

    @field_validator("score")
    @classmethod
    def validate_finite_score(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("score must be finite")
        return value


def assess_research_quality(backtest: BacktestMetrics | dict[str, Any]) -> ResearchQuality:
    """Describe evidence gaps without inventing benchmark or out-of-sample results."""
    raw = backtest.model_dump() if isinstance(backtest, BacktestMetrics) else backtest
    total_trades = int(raw.get("total_trades", 0) or 0)
    if total_trades < 10:
        trade_sample = "insufficient"
    elif total_trades < 30:
        trade_sample = "limited"
    else:
        trade_sample = "adequate"

    has_equity_curve = bool(raw.get("equity_curve"))
    has_trade_records = bool(raw.get("trades") or raw.get("trade_records"))
    has_benchmark = bool(
        raw.get("benchmark")
        or raw.get("benchmark_curve")
        or raw.get("benchmark_return") is not None
    )
    has_out_of_sample = bool(raw.get("out_of_sample") or raw.get("walk_forward"))

    warnings: list[str] = []
    if trade_sample == "insufficient":
        warnings.append("交易样本不足 10 笔，统计指标不稳定。")
    elif trade_sample == "limited":
        warnings.append("交易样本少于 30 笔，结论仅可作为初步参考。")
    if not has_equity_curve:
        warnings.append("缺少净值曲线，无法核验收益与回撤路径。")
    if not has_trade_records:
        warnings.append("缺少逐笔交易记录，无法审计成交与成本。")
    if not has_benchmark:
        warnings.append("缺少真实基准序列或基准收益，不能判断超额收益。")
    if not has_out_of_sample:
        warnings.append("缺少样本外或 walk-forward 结果，不能证明泛化能力。")

    return ResearchQuality(
        trade_sample=trade_sample,
        has_equity_curve=has_equity_curve,
        has_trade_records=has_trade_records,
        has_benchmark=has_benchmark,
        has_out_of_sample=has_out_of_sample,
        warnings=warnings,
    )
