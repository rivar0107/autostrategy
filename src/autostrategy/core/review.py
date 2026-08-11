"""Phase 5E paper-run review: post-run summary, key events and a
structured snapshot that a Learning Agent can consume.

The review is derived purely from the paper-run result dict and its
decision events — it never mutates strategy code and never calls an
LLM by itself.
"""

from __future__ import annotations

from typing import Any


def _max_drawdown(equity_series: list[float]) -> float:
    """Max peak-to-trough drawdown in percent."""
    peak = 0.0
    max_dd = 0.0
    for value in equity_series:
        peak = max(peak, value)
        if peak > 0:
            max_dd = max(max_dd, (peak - value) / peak * 100)
    return round(max_dd, 2)


def build_review(result: dict[str, Any], events: list[dict[str, Any]]) -> dict[str, Any]:
    """Build the review block for a completed paper run.

    Returns a dict with ``metrics`` (return/drawdown/trades/win_rate/
    turnover), ``key_events`` (buys, sells, rejections, errors) and a
    human/agent-readable ``markdown`` summary.
    """
    paper = result.get("paper") if isinstance(result.get("paper"), dict) else {}
    initial_cash = float(paper.get("initial_cash", 0) or 0)
    final_value = float(paper.get("final_value", paper.get("equity", initial_cash)) or 0)

    decisions = [e for e in events if isinstance(e, dict) and e.get("action")]
    fills = [
        e
        for e in decisions
        if e.get("status") in (None, "filled") and e.get("action") in ("buy", "sell")
    ]
    buys = [e for e in fills if e.get("action") == "buy"]
    sells = [e for e in fills if e.get("action") == "sell"]
    rejected = [e for e in decisions if e.get("status") == "rejected"]

    equity_series = [
        float(e["equity_after"])
        for e in decisions
        if isinstance(e.get("equity_after"), (int, float))
    ]
    drawdown = (
        _max_drawdown(equity_series)
        if equity_series
        else float(
            paper.get("max_drawdown", result.get("summary", {}).get("paper_max_drawdown", 0)) or 0
        )
    )

    total_return = (final_value - initial_cash) / initial_cash * 100 if initial_cash else 0.0
    realized_pnl = float(paper.get("realized_pnl", 0) or 0)
    turnover = round(
        sum(float(e.get("price", 0) or 0) * float(e.get("size", 0) or 0) for e in fills), 2
    )

    metrics = {
        "initial_cash": round(initial_cash, 2),
        "final_value": round(final_value, 2),
        "total_return": round(total_return, 2),
        "max_drawdown": round(float(drawdown), 2),
        "trade_count": len(fills),
        "buy_count": len(buys),
        "sell_count": len(sells),
        "rejected_count": len(rejected),
        "realized_pnl": round(realized_pnl, 2),
        "unrealized_pnl": round(float(paper.get("unrealized_pnl", 0) or 0), 2),
        "turnover": turnover,
    }

    key_events = _key_events(decisions)
    return {
        "metrics": metrics,
        "key_events": key_events,
        "markdown": _render_markdown(result, metrics, key_events),
    }


def _key_events(decisions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Extract the events a reviewer actually cares about."""
    key = []
    for event in decisions:
        action = event.get("action")
        status = event.get("status")
        if action in ("buy", "sell") and status in (None, "filled"):
            key.append(
                {
                    "type": action,
                    "timestamp": event.get("timestamp"),
                    "symbol": event.get("symbol"),
                    "price": event.get("price"),
                    "size": event.get("size"),
                    "reason": event.get("reason", ""),
                }
            )
        elif status == "rejected":
            key.append(
                {
                    "type": "rejected",
                    "timestamp": event.get("timestamp"),
                    "symbol": event.get("symbol"),
                    "action": action,
                    "reason": event.get("reject_reason", event.get("reason", "")),
                }
            )
    return key


def _render_markdown(
    result: dict[str, Any], metrics: dict[str, Any], key_events: list[dict[str, Any]]
) -> str:
    """Stable, human-readable review document (Learning Agent input)."""
    lines = [
        "# Paper Run 复盘",
        "",
        f"- 运行状态: {result.get('run_status', 'unknown')}",
        f"- 初始资金: {metrics['initial_cash']}",
        f"- 最终权益: {metrics['final_value']}",
        f"- 总收益: {metrics['total_return']}%",
        f"- 最大回撤: {metrics['max_drawdown']}%",
        f"- 成交笔数: {metrics['trade_count']}"
        f"（买 {metrics['buy_count']} / 卖 {metrics['sell_count']}，"
        f"拒绝 {metrics['rejected_count']}）",
        f"- 已实现盈亏: {metrics['realized_pnl']}",
        f"- 未实现盈亏: {metrics['unrealized_pnl']}",
        f"- 成交额: {metrics['turnover']}",
        "",
        "## 关键事件",
        "",
    ]
    if not key_events:
        lines.append("- 无关键交易事件")
    for event in key_events:
        if event["type"] == "rejected":
            lines.append(
                f"- [{event['timestamp']}] 拒绝 {event.get('action')} "
                f"{event.get('symbol')}: {event.get('reason')}"
            )
        else:
            lines.append(
                f"- [{event['timestamp']}] {event['type'].upper()} {event.get('symbol')} "
                f"{event.get('size')} @ {event.get('price')} — {event.get('reason')}"
            )
    return "\n".join(lines) + "\n"
