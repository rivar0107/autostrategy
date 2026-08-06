"""Productized backtest engine for autostrategy."""

from __future__ import annotations

import copy
import importlib.util
import json
import re
import sys
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path
from time import sleep
from typing import Any

import numpy as np
import yaml

MARKET_BENCHMARKS = {
    "A股": {"index": "000300.SH", "avg_annual_return": 8.0},
    "港股": {"index": "HSI", "avg_annual_return": 5.0},
    "美股": {"index": "^GSPC", "avg_annual_return": 10.0},
}

PASS_CRITERIA = {
    "annual_return": {"min": 3.0, "desc": "> 无风险利率 × 2（约3%）"},
    "max_drawdown": {"max": 20, "desc": "< 20%"},
    "sharpe": {"min": 1.0, "desc": "> 1.0"},
    "win_rate": {"min": 45, "desc": "> 45%"},
    "profit_loss_ratio": {"min": 1.5, "desc": "> 1.5"},
}


def resolve_baseline_return(market: str, config: dict | None = None) -> float:
    """Resolve benchmark return for a market or weighted multi-market config."""
    if market in MARKET_BENCHMARKS:
        return MARKET_BENCHMARKS[market]["avg_annual_return"]
    if config and "symbols" in config:
        market_counts: dict[str, int] = {}
        for symbol in config["symbols"]:
            symbol_market = symbol.get("market", "A股")
            market_counts[symbol_market] = market_counts.get(symbol_market, 0) + 1
        total = sum(market_counts.values())
        if total > 0:
            weighted = sum(
                MARKET_BENCHMARKS.get(market_name, MARKET_BENCHMARKS["A股"])[
                    "avg_annual_return"
                ]
                * count
                for market_name, count in market_counts.items()
            ) / total
            return weighted
    return MARKET_BENCHMARKS["A股"]["avg_annual_return"]


def score_strategy(
    backtest: dict,
    design: dict,
    market: str = "A股",
    config: dict | None = None,
) -> float:
    """Map backtest metrics and strategy complexity to a 0-100 score."""
    baseline_return = resolve_baseline_return(market, config)
    score = 0.0
    score += min(backtest.get("annual_return", 0) / (baseline_return * 2), 1.0) * 25
    score += max(1 - backtest.get("max_drawdown", 100) / 30.0, 0) * 20
    score += min(backtest.get("sharpe", 0) / 2.0, 1.0) * 25
    score += min(backtest.get("win_rate", 0) / 60.0, 1.0) * 15
    score += min(backtest.get("profit_loss_ratio", 0) / 2.5, 1.0) * 15

    condition_count = (
        design.get("num_buy_conditions", 0)
        + design.get("num_sell_conditions", 0)
        + design.get("num_filters", 0)
        + design.get("num_risk_rules", 0)
    )
    complexity_penalty = max(0, (condition_count - 10) * 1.5)
    return max(0, score - complexity_penalty)


def run_diagnostics(backtest: dict) -> list[dict]:
    """Run heuristic diagnostics against a backtest result."""
    diagnostics = []

    period_returns = backtest.get("period_returns", [])
    if period_returns:
        avg = sum(period_returns) / len(period_returns)
        variance = sum((value - avg) ** 2 for value in period_returns) / len(period_returns)
        cv = (variance**0.5) / abs(avg) if avg != 0 else 999
        if cv > 3.0:
            diagnostics.append({"item": "过拟合", "status": "⚠️", "detail": f"收益波动系数 {cv:.1f}"})
        else:
            diagnostics.append({"item": "过拟合", "status": "✅", "detail": f"收益波动系数 {cv:.1f}"})

    universe = backtest.get("universe_size", 0)
    survivors = backtest.get("survivor_count", 0)
    if universe > 0 and survivors > 0 and survivors < universe:
        ratio = survivors / universe
        status = "⚠️" if ratio < 0.5 else "✅"
        diagnostics.append({"item": "幸存者偏差", "status": status, "detail": f"使用 {survivors}/{universe} 只股票"})

    future_leak = backtest.get("future_leak_detected", False)
    if future_leak:
        diagnostics.append({"item": "未来函数", "status": "❌", "detail": "检测到可能使用了未来数据"})
    else:
        diagnostics.append({"item": "未来函数", "status": "✅", "detail": "未检测到未来数据使用（需人工复核）"})

    avg_volume = backtest.get("avg_daily_volume", 0)
    avg_trade_value = backtest.get("avg_trade_value", 0)
    if avg_volume > 0 and avg_trade_value / avg_volume > 0.1:
        diagnostics.append({"item": "流动性", "status": "⚠️", "detail": "单笔交易占日均成交额比例过高"})
    else:
        diagnostics.append({"item": "流动性", "status": "✅", "detail": "交易量与市场流动性匹配"})

    first_half = backtest.get("first_half_return", 0)
    second_half = backtest.get("second_half_return", 0)
    if first_half != 0 and second_half != 0:
        diff = abs(first_half - second_half) / max(abs(first_half), abs(second_half))
        status = "⚠️" if diff > 0.8 else "✅"
        diagnostics.append({"item": "稳定性", "status": status, "detail": f"前后半段收益差异 {diff:.0%}"})

    return diagnostics


def check_pass_criteria(backtest: dict) -> list[dict]:
    """Check if backtest result meets default pass criteria."""
    results = []
    labels = {
        "annual_return": "年化收益率",
        "max_drawdown": "最大回撤",
        "sharpe": "夏普比率",
        "win_rate": "胜率",
        "profit_loss_ratio": "盈亏比",
    }
    for key, criteria in PASS_CRITERIA.items():
        value = backtest.get(key)
        label = labels.get(key, key)
        if value is None:
            results.append({"metric": label, "value": "N/A", "criteria": criteria["desc"], "passed": None})
            continue
        passed = True
        if "min" in criteria and criteria["min"] is not None:
            passed = passed and value >= criteria["min"]
        if "max" in criteria and criteria["max"] is not None:
            passed = passed and value <= criteria["max"]
        unit = "%" if key in ("annual_return", "max_drawdown", "win_rate") else ""
        display = f"{value:.2f}{unit}" if isinstance(value, float) else str(value)
        results.append({"metric": label, "value": display, "criteria": criteria["desc"], "passed": passed})
    return results


def load_config(strategy_dir: Path) -> dict:
    """Load config.yaml from a strategy directory."""
    config_path = strategy_dir / "config.yaml"
    if config_path.exists():
        with open(config_path, encoding="utf-8") as file:
            return yaml.safe_load(file) or {}
    return {}


def run_single_backtest(strategy_dir: Path, config: dict) -> dict:
    """Load strategy.py and run its backtest interface."""
    module = _load_strategy_module(strategy_dir)
    if module is None:
        return {"error": f"strategy.py 不存在: {strategy_dir / 'strategy.py'}"}
    return _execute_strategy(module, config, strategy_dir)


def run_backtest_workflow(strategy_dir: Path, output_path: Path | None = None) -> dict:
    """Run a standard backtest workflow and save JSON result."""
    config = load_config(strategy_dir)
    design = extract_design_complexity(strategy_dir)
    backtest = run_single_backtest(strategy_dir, config)
    output_dir = strategy_dir / "backtest" / "results"

    if "error" in backtest:
        result = {"error": backtest["error"], "score": 0}
        save_json(output_path or output_dir / "backtest_result.json", result)
        return result

    market = config.get("market", "A股")
    total_score = score_strategy(backtest, design, market, config)
    result = {
        "backtest": backtest,
        "score": round(total_score, 1),
        "criteria": check_pass_criteria(backtest),
        "diagnostics": run_diagnostics(backtest),
    }
    save_json(output_path or output_dir / "backtest_result.json", result)
    return result


def run_paper_replay_workflow(strategy_dir: Path, stop_requested=None) -> dict:
    """Run a replay-first paper run workflow and save paper artifacts."""
    config = load_config(strategy_dir)
    module = _load_strategy_module(strategy_dir)
    result_path = strategy_dir / "paper_run" / "results" / "paper_run_result.json"
    events_path = strategy_dir / "paper_run" / "results" / "paper_run_events.jsonl"
    log_path = strategy_dir / "paper_run" / "logs" / "paper_run.log"
    started_at = _utc_now()

    if module is None:
        result = _paper_failed_result(started_at, f"strategy.py 不存在: {strategy_dir / 'strategy.py'}")
        _write_paper_artifacts(result_path, events_path, log_path, result, [])
        return result

    if not hasattr(module, "run_paper"):
        result = _paper_failed_result(started_at, "strategy.py 未暴露 run_paper(config) 函数")
        _write_paper_artifacts(result_path, events_path, log_path, result, [])
        return result

    try:
        raw = module.run_paper(config)
    except Exception as exc:
        result = _paper_failed_result(started_at, f"run_paper() 执行失败: {exc}")
        _write_paper_artifacts(result_path, events_path, log_path, result, [])
        return result

    try:
        result = _run_incremental_paper_replay(
            raw,
            started_at,
            result_path,
            events_path,
            log_path,
            stop_requested=stop_requested,
        )
    except Exception as exc:
        result = _paper_failed_result(started_at, f"replay 执行失败: {exc}")
        _write_paper_artifacts(result_path, events_path, log_path, result, [])
    return result


def _run_incremental_paper_replay(
    raw: Any,
    started_at: str,
    result_path: Path,
    events_path: Path,
    log_path: Path,
    stop_requested=None,
) -> dict:
    if isinstance(raw, dict):
        return _finish_static_paper_replay(raw, started_at, result_path, events_path, log_path, stop_requested)
    if isinstance(raw, str | bytes) or not isinstance(raw, Iterable):
        result = _paper_failed_result(started_at, "run_paper() 必须返回 dict 或可迭代 replay 事件")
        _write_paper_artifacts(result_path, events_path, log_path, result, [])
        return result

    events: list[dict] = []
    final_raw: dict = {}
    result = _paper_result_from_raw(
        {"events": events, "replay": {"bars_processed": 0, "progress": 0.0}},
        started_at,
        run_status="running",
    )
    _write_paper_artifacts(result_path, events_path, log_path, result, events)

    for item in raw:
        if stop_requested and stop_requested():
            result = _paper_result_from_raw(_raw_with_events(final_raw, events), started_at, run_status="stopped")
            _write_paper_artifacts(result_path, events_path, log_path, result, events)
            return result

        if not isinstance(item, dict):
            result = _paper_failed_result(started_at, "replay 事件必须是 dict")
            _write_paper_artifacts(result_path, events_path, log_path, result, events)
            return result

        if _looks_like_event(item):
            events.append(item)
            final_raw = _merge_replay_event(final_raw, item, events)
        else:
            final_raw = _merge_replay_snapshot(final_raw, item, events)
            events = _paper_events_from_raw(final_raw)

        result = _paper_result_from_raw(_raw_with_events(final_raw, events), started_at, run_status="running")
        _write_paper_artifacts(result_path, events_path, log_path, result, events)
        sleep(float(final_raw.get("replay_interval_seconds", 0) or 0))

    if "error" in final_raw:
        result = _paper_failed_result(started_at, str(final_raw["error"]), _raw_with_events(final_raw, events))
    elif stop_requested and stop_requested():
        result = _paper_result_from_raw(_raw_with_events(final_raw, events), started_at, run_status="stopped")
    else:
        result = _paper_result_from_raw(_raw_with_events(final_raw, events), started_at, run_status="completed")
    _write_paper_artifacts(result_path, events_path, log_path, result, events)
    return result


def _finish_static_paper_replay(
    raw: dict,
    started_at: str,
    result_path: Path,
    events_path: Path,
    log_path: Path,
    stop_requested=None,
) -> dict:
    if stop_requested and stop_requested():
        result = _paper_result_from_raw(raw, started_at, run_status="stopped")
    elif "error" in raw:
        result = _paper_failed_result(started_at, str(raw["error"]), raw)
    else:
        result = _paper_result_from_raw(raw, started_at, run_status="completed")
    _write_paper_artifacts(result_path, events_path, log_path, result, _paper_events_from_raw(raw))
    return result


def _looks_like_event(item: dict) -> bool:
    return any(key in item for key in {"timestamp", "action", "symbol", "price", "size", "reason"})


def _merge_replay_event(raw: dict, event: dict, events: list[dict]) -> dict:
    merged = dict(raw)
    replay = dict(merged.get("replay") if isinstance(merged.get("replay"), dict) else {})
    replay["bars_processed"] = int(replay.get("bars_processed", 0) or 0) + 1
    if event.get("timestamp") is not None:
        replay["current_at"] = event["timestamp"]
    if event.get("progress") is not None:
        replay["progress"] = event["progress"]
    merged["replay"] = replay
    merged["latest_decision"] = event
    merged["events"] = events
    return merged


def _merge_replay_snapshot(raw: dict, snapshot: dict, events: list[dict]) -> dict:
    merged = dict(raw)
    for key, value in snapshot.items():
        if key == "events" and isinstance(value, list):
            merged[key] = value
        elif key == "replay" and isinstance(value, dict):
            replay = dict(merged.get("replay") if isinstance(merged.get("replay"), dict) else {})
            replay.update(value)
            merged[key] = replay
        elif key == "paper" and isinstance(value, dict):
            paper = dict(merged.get("paper") if isinstance(merged.get("paper"), dict) else {})
            paper.update(value)
            merged[key] = paper
        elif key == "summary" and isinstance(value, dict):
            summary = dict(merged.get("summary") if isinstance(merged.get("summary"), dict) else {})
            summary.update(value)
            merged[key] = summary
        else:
            merged[key] = value
    if "events" not in merged:
        merged["events"] = events
    return merged


def _raw_with_events(raw: dict, events: list[dict]) -> dict:
    merged = dict(raw)
    merged["events"] = events
    return merged


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _paper_failed_result(started_at: str, error: str, raw: dict | None = None) -> dict:
    result = _paper_result_from_raw(raw or {}, started_at, run_status="failed")
    result["error"] = error
    result["diagnostics"] = [{"item": "paper_run", "status": "❌", "detail": error}]
    return result


def _paper_result_from_raw(raw: dict, started_at: str, run_status: str) -> dict:
    updated_at = _utc_now()
    paper = raw.get("paper") if isinstance(raw.get("paper"), dict) else raw
    summary = raw.get("summary") if isinstance(raw.get("summary"), dict) else _paper_summary(paper)
    events = _paper_events_from_raw(raw)
    latest_decision = raw.get("latest_decision") or (events[-1] if events else None)
    bars_processed = int(raw.get("bars_processed") or len(raw.get("equity_curve", [])) or len(events))
    replay = raw.get("replay") if isinstance(raw.get("replay"), dict) else {}
    current_at = replay.get("current_at") or paper.get("current_at")
    if current_at is None and latest_decision:
        current_at = latest_decision.get("timestamp")
    replay = {
        "current_at": current_at,
        "bars_processed": replay.get("bars_processed", bars_processed),
        "progress": replay.get("progress", 1.0 if run_status in {"completed", "failed", "stopped"} else 0.0),
    }
    return {
        "mode": "paper_run",
        "run_status": run_status,
        "started_at": raw.get("started_at") or started_at,
        "updated_at": updated_at,
        "replay": replay,
        "summary": summary,
        "latest_decision": latest_decision,
        "diagnostics": raw.get("diagnostics", []),
        "error": raw.get("error"),
    }


def _paper_summary(paper: dict) -> dict:
    initial_cash = float(paper.get("initial_cash", 0) or 0)
    final_value = float(paper.get("final_value", initial_cash) or 0)
    total_return = paper.get("total_return")
    if total_return is None and initial_cash:
        total_return = (final_value - initial_cash) / initial_cash * 100
    return {
        "paper_return": round(float(total_return or 0), 2),
        "paper_max_drawdown": float(paper.get("max_drawdown", 0) or 0),
        "trade_count": int(paper.get("trade_count", paper.get("total_trades", 0)) or 0),
        "position_count": int(paper.get("position_count", 0) or 0),
        "final_value": round(final_value, 2),
    }


def _paper_events_from_raw(raw: dict) -> list[dict]:
    events = raw.get("events", raw.get("decisions", []))
    return events if isinstance(events, list) else []


def _write_paper_artifacts(
    result_path: Path,
    events_path: Path,
    log_path: Path,
    result: dict,
    events: list[dict],
) -> None:
    save_json(result_path, result)
    events_path.parent.mkdir(parents=True, exist_ok=True)
    events_path.write_text("".join(json.dumps(event, ensure_ascii=False) + "\n" for event in events), encoding="utf-8")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(f"{result['updated_at']} {result['run_status']}\n", encoding="utf-8")


def save_json(path: Path, data: dict) -> None:
    """Save JSON result, converting numpy scalar types."""
    def convert(obj):
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        raise TypeError

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2, default=convert)


def _load_strategy_module(strategy_dir: Path):
    """Dynamically load strategy.py from a strategy directory."""
    strategy_file = strategy_dir / "strategy.py"
    if not strategy_file.exists():
        return None
    module_name = f"strategy_{strategy_dir.name}_{id(strategy_file)}"
    spec = importlib.util.spec_from_file_location(module_name, strategy_file)
    if spec is None or spec.loader is None:
        return None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        return None
    return module


def _execute_strategy(module, config: dict, strategy_dir: Path) -> dict:
    """Execute a loaded strategy module."""
    if hasattr(module, "run_backtest"):
        try:
            return module.run_backtest(config)
        except Exception as exc:
            return {"error": f"run_backtest() 执行失败: {exc}"}
    if hasattr(module, "Strategy"):
        try:
            return _run_backtrader(module, config, strategy_dir)
        except Exception as exc:
            return {"error": f"Backtrader 回测失败: {exc}"}
    return {"error": "strategy.py 未暴露 run_backtest() 函数或 Strategy class"}


def _run_backtrader(module, config: dict, strategy_dir: Path) -> dict:
    """Run a Backtrader Strategy class."""
    import backtrader as bt
    import pandas as pd

    cerebro = bt.Cerebro()
    data_file = strategy_dir / "data" / "data.csv"
    if data_file.exists():
        df = pd.read_csv(data_file, parse_dates=["date"])
        df.set_index("date", inplace=True)
        cerebro.adddata(bt.feeds.PandasData(dataname=df))
    else:
        fetch_file = strategy_dir / "data" / "fetch_data.py"
        if fetch_file.exists():
            spec = importlib.util.spec_from_file_location("fetch_data", fetch_file)
            if spec is not None and spec.loader is not None:
                fetch_module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(fetch_module)
                if hasattr(fetch_module, "fetch"):
                    df = fetch_module.fetch(config)
                    if df is not None:
                        cerebro.adddata(bt.feeds.PandasData(dataname=df))

    initial_cash = config.get("initial_cash", 1000000)
    cerebro.broker.setcash(initial_cash)
    cerebro.broker.setcommission(commission=config.get("commission", 0.0003))
    cerebro.broker.set_slippage_perc(config.get("slippage", 0.001))
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name="sharpe", riskfreerate=0.015)
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name="drawdown")
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name="trades")
    cerebro.addstrategy(module.Strategy)
    results = cerebro.run()
    strat = results[0]
    final_value = cerebro.broker.getvalue()
    total_return = (final_value - initial_cash) / initial_cash * 100
    days = len(cerebro.datas[0]) if cerebro.datas else 252
    years = max(days / 252, 0.1)
    annual_return = ((1 + total_return / 100) ** (1 / years) - 1) * 100
    sharpe = strat.analyzers.sharpe.get_analysis().get("sharperatio", 0) or 0
    max_drawdown = strat.analyzers.drawdown.get_analysis().get("max", {}).get("drawdown", 0) or 0
    trades = strat.analyzers.trades.get_analysis()
    total_trades = trades.get("total", {}).get("total", 0)
    won = trades.get("won", {}).get("total", 0)
    win_rate = won / max(total_trades, 1) * 100
    avg_win = trades.get("won", {}).get("pnl", {}).get("average", 0) or 0
    avg_loss = abs(trades.get("lost", {}).get("pnl", {}).get("average", 0.01) or 0.01)
    profit_loss_ratio = avg_win / avg_loss if avg_loss > 0 else 0
    return {
        "annual_return": round(annual_return, 2),
        "max_drawdown": round(max_drawdown, 2),
        "sharpe": round(sharpe, 2),
        "win_rate": round(win_rate, 1),
        "profit_loss_ratio": round(profit_loss_ratio, 2),
        "total_trades": total_trades,
        "initial_cash": initial_cash,
        "final_value": round(final_value, 2),
        "total_return": round(total_return, 2),
    }


def extract_design_complexity(strategy_dir: Path) -> dict:
    """Extract condition counts from STRATEGY_DESIGN.md for scoring."""
    design_doc = strategy_dir / "STRATEGY_DESIGN.md"
    design = {}
    if not design_doc.exists():
        return design
    content = design_doc.read_text(encoding="utf-8")
    signal_section = _extract_section_content(
        content,
        ["信号逻辑", "开仓信号", "平仓信号", "买入信号", "卖出信号", "买入条件", "卖出条件"],
    )
    if signal_section:
        design["num_buy_conditions"] = _count_conditions_in_subsection(
            signal_section,
            [r"条件\d+.*?(买入|BUY|开多)", r"^\d+\.\s+.*?(买入|BUY|开多)", r"[①-⓿].*?(买入|BUY|开多)"],
        )
        design["num_sell_conditions"] = _count_conditions_in_subsection(
            signal_section,
            [r"条件\d+.*?(卖出|SELL|平多|平空)", r"^\d+\.\s+.*?(卖出|SELL|平多|平空)", r"[①-⓿].*?(卖出|SELL|平多|平空)"],
        )
        design["num_filters"] = _count_conditions_in_subsection(signal_section, [r"(过滤|过滤条件)"])
    risk_section = _extract_section_content(content, ["风控规则", "止损", "通用风控", "组合级风控"])
    if risk_section:
        design["num_risk_rules"] = _count_conditions_in_subsection(
            risk_section, [r"(止损|止盈|回撤|清仓|暂停|仓位|连续)"]
        )
    return design


def _extract_section_content(text: str, markers: list[str], max_len: int = 2000) -> str:
    """Extract a markdown section by marker."""
    for marker in markers:
        idx = text.find(marker)
        if idx != -1:
            end = idx + max_len
            for match in re.finditer(r"\n## [^#]", text[idx + 4 :]):
                end = idx + 4 + match.start()
                break
            return text[idx:end]
    return ""


def _count_conditions_in_subsection(section: str, patterns: list[str]) -> int:
    """Count condition-like patterns in a markdown section."""
    count = 0
    for pattern in patterns:
        count += len(re.findall(pattern, section, flags=re.MULTILINE))
    return count


def deep_copy_config(config: dict) -> dict:
    """Deep copy config to avoid mutating the original."""
    return copy.deepcopy(config)
