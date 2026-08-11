"""Virtual paper-trading account model for Phase 5C.

Maintains cash, positions and equity while replaying strategy decisions
(``buy`` / ``sell`` / ``hold``). The model is deliberately simple:

- Fills happen at the event's ``price`` (bar close by convention).
- No commission or slippage unless the config explicitly provides them.
- ``hold`` decisions are recorded but never mutate the account.

The account is pure and side-effect free — it is consumed by
``run_paper_replay_workflow`` which persists snapshots and events.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

ACTIONS = {"buy", "sell", "hold"}


@dataclass
class Position:
    """A single open position."""

    symbol: str
    quantity: float = 0.0
    avg_price: float = 0.0
    last_price: float = 0.0

    @property
    def market_value(self) -> float:
        return round(self.quantity * self.last_price, 2)

    @property
    def unrealized_pnl(self) -> float:
        return round((self.last_price - self.avg_price) * self.quantity, 2)

    def snapshot(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "quantity": self.quantity,
            "avg_price": round(self.avg_price, 4),
            "last_price": round(self.last_price, 4),
            "market_value": self.market_value,
            "unrealized_pnl": self.unrealized_pnl,
        }


@dataclass
class PaperAccount:
    """Virtual account state for a paper run."""

    cash: float
    initial_cash: float
    commission: float = 0.0
    positions: dict[str, Position] = field(default_factory=dict)
    realized_pnl: float = 0.0
    trade_count: int = 0

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> PaperAccount:
        initial_cash = float(config.get("initial_cash", 1_000_000) or 0)
        return cls(
            cash=initial_cash,
            initial_cash=initial_cash,
            commission=float(config.get("commission", 0.0) or 0.0),
        )

    def mark_price(self, symbol: str, price: float) -> None:
        """Update the last known price for a symbol (marks to market)."""
        if symbol in self.positions:
            self.positions[symbol].last_price = price

    @property
    def equity(self) -> float:
        return round(self.cash + sum(p.market_value for p in self.positions.values()), 2)

    @property
    def unrealized_pnl(self) -> float:
        return round(sum(p.unrealized_pnl for p in self.positions.values()), 2)

    def apply(self, decision: dict[str, Any]) -> dict[str, Any]:
        """Apply a ``buy``/``sell``/``hold`` decision and return a fill event.

        The returned event is the standard paper-run event dict extended
        with ``cash_after``, ``equity_after`` and ``position_after``.
        Rejected decisions (insufficient cash/position) are returned with
        ``status: "rejected"`` and never mutate the account.
        """
        action = str(decision.get("action", "hold")).lower()
        symbol = str(decision.get("symbol", ""))
        price = float(decision.get("price", 0) or 0)
        size = float(decision.get("size", decision.get("quantity", 0)) or 0)

        if action not in ACTIONS:
            action = "hold"
        if symbol and price > 0:
            self.mark_price(symbol, price)

        event: dict[str, Any] = dict(decision)
        event["action"] = action

        if action == "hold" or not symbol or price <= 0:
            event["status"] = "held"
            event.update(self._after_fields(symbol))
            return event

        if action == "buy":
            cost = price * size * (1 + self.commission)
            if size <= 0 or cost > self.cash:
                event["status"] = "rejected"
                event["reject_reason"] = "insufficient_cash"
                event.update(self._after_fields(symbol))
                return event
            position = self.positions.setdefault(symbol, Position(symbol=symbol))
            total_qty = position.quantity + size
            position.avg_price = (
                (position.avg_price * position.quantity + price * size) / total_qty
                if total_qty
                else price
            )
            position.quantity = total_qty
            position.last_price = price
            self.cash = round(self.cash - cost, 2)
            self.trade_count += 1
            event["status"] = "filled"
        else:  # sell
            position = self.positions.get(symbol)
            if position is None or size <= 0 or position.quantity < size:
                event["status"] = "rejected"
                event["reject_reason"] = "insufficient_position"
                event.update(self._after_fields(symbol))
                return event
            proceeds = price * size * (1 - self.commission)
            self.realized_pnl = round(
                self.realized_pnl
                + (price - position.avg_price) * size
                - price * size * self.commission,
                2,
            )
            position.quantity -= size
            position.last_price = price
            self.cash = round(self.cash + proceeds, 2)
            self.trade_count += 1
            if position.quantity <= 0:
                del self.positions[symbol]
            event["status"] = "filled"

        event.update(self._after_fields(symbol))
        return event

    def _after_fields(self, symbol: str) -> dict[str, Any]:
        position = self.positions.get(symbol)
        return {
            "cash_after": round(self.cash, 2),
            "equity_after": self.equity,
            "position_after": position.quantity if position else 0,
        }

    def snapshot(self) -> dict[str, Any]:
        """Return the account summary used in paper_run result artifacts."""
        return {
            "initial_cash": self.initial_cash,
            "cash": round(self.cash, 2),
            "equity": self.equity,
            "final_value": self.equity,
            "realized_pnl": self.realized_pnl,
            "unrealized_pnl": self.unrealized_pnl,
            "trade_count": self.trade_count,
            "position_count": len(self.positions),
            "positions": [p.snapshot() for p in self.positions.values()],
        }
