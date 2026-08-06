#!/usr/bin/env python3
"""Compatibility wrapper for the productized autostrategy backtest engine.

This script keeps the legacy invocation style working:

    python3 scripts/run_backtest.py <strategy_dir>
    python3 scripts/run_backtest.py <strategy_dir> --output results.json

The reusable implementation now lives in
`autostrategy.core.backtest_engine` and is used by both this wrapper and
`autostrategy backtest run`.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from autostrategy.core.backtest_engine import run_backtest_workflow


def main() -> None:
    """Run the legacy backtest CLI wrapper."""
    parser = argparse.ArgumentParser(description="Autostrategy 回测执行")
    parser.add_argument("strategy_dir", help="策略目录路径")
    parser.add_argument("--output", "-o", help="输出 JSON 文件路径")
    parser.add_argument(
        "--split",
        type=float,
        default=None,
        help="Train/Test Split 比例（兼容参数，产品版暂未启用）",
    )
    parser.add_argument(
        "--sensitivity",
        action="store_true",
        help="参数敏感度分析（兼容参数，产品版暂未启用）",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="生成回测净值曲线图（兼容参数，产品版暂未启用）",
    )
    args = parser.parse_args()

    strategy_dir = Path(args.strategy_dir).expanduser()
    if not strategy_dir.exists():
        print(f"❌ 目录不存在: {strategy_dir}")
        sys.exit(1)

    if args.split is not None:
        print("⚠️ --split 参数已保留兼容，产品版当前使用标准回测。")
    if args.sensitivity:
        print("⚠️ --sensitivity 参数已保留兼容，产品版当前使用标准回测。")
    if args.plot:
        print("⚠️ --plot 参数已保留兼容，产品版当前不会生成图表。")

    output_path = Path(args.output).expanduser() if args.output else None
    result = run_backtest_workflow(strategy_dir, output_path=output_path)

    if "error" in result:
        print(f"❌ 回测失败: {result['error']}")
        sys.exit(1)

    backtest = result.get("backtest", {})
    print("=" * 50)
    print("  回测结果")
    print("=" * 50)
    print(f"  年化收益率:  {backtest.get('annual_return', 'N/A')}")
    print(f"  最大回撤:    {backtest.get('max_drawdown', 'N/A')}")
    print(f"  夏普比率:    {backtest.get('sharpe', 'N/A')}")
    print(f"  胜率:        {backtest.get('win_rate', 'N/A')}")
    print(f"  盈亏比:      {backtest.get('profit_loss_ratio', 'N/A')}")
    print(f"  交易次数:    {backtest.get('total_trades', 'N/A')}")
    print(f"\n  score_strategy(): {result.get('score', 0):.1f}/100")
    print("  📄 结果已保存到 backtest/results/backtest_result.json")
    sys.exit(0)


if __name__ == "__main__":
    main()
