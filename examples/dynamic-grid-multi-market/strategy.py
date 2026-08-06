"""
动态网格多标的策略

严格翻译自 STRATEGY_DESIGN.md，不可自由发挥。
标的：腾讯控股(0700.HK)、科创50ETF(588000.SH)、中证2000ETF(563300.SH)、
     小鹏汽车(9868.HK)、特斯拉(TSLA)
"""

import numpy as np
import pandas as pd


def compute_atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int) -> pd.Series:
    """指标 1: ATR — TR = max(H-L, |H-PrevC|, |L-PrevC|); ATR(N) = SMA(TR, N)"""
    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(window=period).mean()


def compute_grid_lines(base_price: float, atr_value: float,
                       grid_multiplier: float, grid_levels: int) -> list[float]:
    """指标 4: 网格线序列 — GridLine(i) = Base + i × GridStep"""
    grid_step = atr_value * grid_multiplier
    lines = []
    for i in range(-grid_levels, grid_levels + 1):
        lines.append(base_price + i * grid_step)
    return sorted(lines)


def run_backtest(config: dict) -> dict:
    """
    主回测入口。按 STRATEGY_DESIGN.md 逐条翻译。

    输入: config.yaml 的内容（dict）
    输出: 回测结果字典
    """
    # ── 参数提取 ──────────────────────────────────────────
    indicators = config.get("indicators", {})
    risk = config.get("risk", {})
    symbols = config.get("symbols", [])

    atr_period = indicators.get("atr_period", 14)
    ma_period = indicators.get("ma_period", 60)
    grid_multiplier = indicators.get("grid_multiplier", 1.5)
    grid_levels = indicators.get("grid_levels", 5)
    rebalance_days = indicators.get("rebalance_days", 5)

    single_grid_pct = risk.get("single_grid_pct", 5) / 100
    stop_loss_pct = risk.get("stop_loss_pct", 5) / 100
    max_position_pct = risk.get("max_position_pct", 20) / 100
    total_position_pct = risk.get("total_position_pct", 80) / 100
    extreme_drop_pct = risk.get("extreme_drop_pct", 10) / 100

    initial_cash = config.get("initial_cash", 1000000)
    slippage = config.get("slippage", 0.001)

    # ── 加载数据 ──────────────────────────────────────────
    data_dir = None
    try:
        from pathlib import Path
        import os
        # 尝试从 data/ 目录加载 CSV
        strategy_dir = Path(__file__).parent
        data_dir = strategy_dir / "data"
    except Exception:
        pass

    all_data = {}
    for sym_conf in symbols:
        code = sym_conf["code"]
        csv_path = data_dir / f"{code.replace('.', '_')}.csv" if data_dir else None
        if csv_path and csv_path.exists():
            df = pd.read_csv(csv_path, parse_dates=["date"])
            df.set_index("date", inplace=True)
            all_data[code] = df
        else:
            # 生成模拟数据用于演示
            all_data[code] = _generate_mock_data(
                code, config.get("start_date", "2024-01-01"),
                config.get("end_date", "2025-12-31"),
                sym_conf.get("market", "A股"),
            )

    # ── 合并日期索引 ──────────────────────────────────────
    all_dates = set()
    for df in all_data.values():
        all_dates.update(df.index)
    all_dates = sorted(all_dates)

    if not all_dates:
        return _empty_result()

    # ── 初始化状态 ────────────────────────────────────────
    cash = float(initial_cash)
    # 每个标的的持仓状态: {code: {grid_line_idx: {"shares": N, "cost_price": P, "buy_date": D}}}
    positions = {sym["code"]: {} for sym in symbols}
    # 每个标的的网格线（rebalance 时更新）
    grid_state = {sym["code"]: {"lines": [], "last_rebalance": None} for sym in symbols}
    # A股T+1约束: {code: last_buy_date}
    last_buy_date = {sym["code"]: None for sym in symbols}
    # 交易记录
    trades = []
    # 每日净值
    daily_values = []
    consecutive_loss_count = 0
    paused_until = {}  # {code: 恢复日期}

    # ── 主循环 ────────────────────────────────────────────
    for date_idx, date in enumerate(all_dates):
        total_position_value = 0

        for sym_conf in symbols:
            code = sym_conf["code"]
            lot_size = sym_conf.get("lot_size", 100)
            sym_commission = sym_conf.get("commission", 0.0003)
            sym_stamp_tax = sym_conf.get("stamp_tax", 0)
            t_plus_1 = sym_conf.get("t_plus_1", False)
            price_limit_pct = sym_conf.get("price_limit_pct")
            name = sym_conf.get("name", code)

            df = all_data.get(code)
            if df is None or date not in df.index:
                continue

            row = df.loc[date]
            close_price = float(row["close"])
            high_price = float(row["high"])
            low_price = float(row["low"])
            open_price = float(row["open"])

            # ── 极端行情检测（规则 6）──
            if date in paused_until and paused_until[code] is not None:
                if date <= paused_until[code]:
                    # 计算持仓市值后跳过交易
                    for grid_key, pos in positions[code].items():
                        total_position_value += pos["shares"] * close_price
                    continue
                else:
                    paused_until[code] = None

            daily_drop = (close_price - open_price) / open_price if open_price > 0 else 0
            if daily_drop < -extreme_drop_pct:
                paused_until[code] = all_dates[min(date_idx + 3, len(all_dates) - 1)]
                for grid_key, pos in positions[code].items():
                    total_position_value += pos["shares"] * close_price
                continue

            # ── Rebalance 网格线 ──
            loc = df.index.get_loc(date)
            if loc < ma_period:
                for grid_key, pos in positions[code].items():
                    total_position_value += pos["shares"] * close_price
                continue

            gs = grid_state[code]
            need_rebalance = (
                gs["last_rebalance"] is None
                or (gs["last_rebalance"] is not None and (date - gs["last_rebalance"]).days >= rebalance_days)
            )
            if need_rebalance:
                slice_df = df.iloc[:loc + 1]
                base_price = float(slice_df["close"].tail(ma_period).mean())
                atr_val = float(compute_atr(
                    slice_df["high"].tail(atr_period + 1),
                    slice_df["low"].tail(atr_period + 1),
                    slice_df["close"].tail(atr_period + 1),
                    atr_period,
                ).iloc[-1])
                gs["lines"] = compute_grid_lines(base_price, atr_val, grid_multiplier, grid_levels)
                gs["last_rebalance"] = date

            grid_lines = gs["lines"]
            if not grid_lines:
                for grid_key, pos in positions[code].items():
                    total_position_value += pos["shares"] * close_price
                continue

            # ── 涨跌停检查（A股）──
            prev_close = float(df.iloc[loc - 1]["close"]) if loc > 0 else close_price
            at_limit_up = False
            at_limit_down = False
            if price_limit_pct and prev_close > 0:
                limit_up = prev_close * (1 + price_limit_pct / 100)
                limit_down = prev_close * (1 - price_limit_pct / 100)
                at_limit_up = close_price >= limit_up * 0.99
                at_limit_down = close_price <= limit_down * 1.01

            # ── 卖出信号（优先于买入，信号优先级规则 1）──
            grids_to_sell = []
            for grid_key in list(positions[code].keys()):
                pos = positions[code][grid_key]

                # A股T+1检查（条件3）
                if t_plus_1 and last_buy_date[code] == date:
                    total_position_value += pos["shares"] * close_price
                    continue

                # 条件2: 止损
                pnl_pct = (close_price - pos["cost_price"]) / pos["cost_price"]
                if pnl_pct < -stop_loss_pct:
                    grids_to_sell.append((grid_key, "止损"))
                    continue

                # 条件1: 价格突破上一条网格线
                sell_target = pos["cost_price"] + (grid_lines[1] - grid_lines[0]) if len(grid_lines) > 1 else pos["cost_price"] * 1.02
                if close_price >= sell_target:
                    grids_to_sell.append((grid_key, "网格盈利"))
                    continue

                total_position_value += pos["shares"] * close_price

            # 执行卖出
            for grid_key, reason in grids_to_sell:
                pos = positions[code].pop(grid_key)
                sell_price = close_price * (1 - slippage)
                sell_amount = pos["shares"] * sell_price
                commission_cost = sell_amount * sym_commission
                stamp_cost = sell_amount * sym_stamp_tax
                cash += sell_amount - commission_cost - stamp_cost
                pnl = (sell_price - pos["cost_price"]) * pos["shares"] - commission_cost - stamp_cost
                if pnl < 0:
                    consecutive_loss_count += 1
                else:
                    consecutive_loss_count = 0
                trades.append({
                    "date": str(date.date()) if hasattr(date, "date") else str(date),
                    "symbol": code, "name": name, "action": "SELL",
                    "price": round(sell_price, 2), "shares": pos["shares"],
                    "reason": reason, "pnl": round(pnl, 2),
                })

            # ── 买入信号 ──
            if not at_limit_up:  # 不在涨停板时触发
                # 降仓检查（连续亏损）
                position_multiplier = 0.5 if consecutive_loss_count >= risk.get("max_loss_grids", 3) else 1.0
                # 总仓位检查（条件3）
                current_total_position = sum(
                    p["shares"] * close_price for sym_positions in positions.values() for p in sym_positions.values()
                )
                total_value = cash + current_total_position

                if current_total_position / total_value < total_position_pct:
                    single_invest = total_value * single_grid_pct * position_multiplier

                    for i, grid_price in enumerate(grid_lines):
                        if i in positions[code]:
                            continue  # 条件2: 同网格线不重复建仓

                        # 条件1: 价格跌破网格线
                        prev_close_val = float(df.iloc[loc - 1]["close"]) if loc > 0 else close_price
                        if prev_close_val >= grid_price and close_price < grid_price:
                            invest = min(single_invest, cash)
                            if invest <= 0:
                                continue

                            buy_price = close_price * (1 + slippage)
                            shares = int(invest / buy_price / lot_size) * lot_size
                            if shares <= 0:
                                continue

                            # 单只标的仓位上限（规则1）
                            current_sym_value = sum(
                                p["shares"] * close_price for p in positions[code].values()
                            )
                            if (current_sym_value + shares * buy_price) / total_value > max_position_pct:
                                max_shares = int(total_value * max_position_pct * 0.95 / buy_price / lot_size) * lot_size
                                shares = min(shares, max_shares)
                            if shares <= 0:
                                continue

                            cost = shares * buy_price
                            commission_cost = cost * sym_commission
                            cash -= cost + commission_cost
                            positions[code][i] = {
                                "shares": shares,
                                "cost_price": buy_price,
                                "buy_date": date,
                            }
                            if t_plus_1:
                                last_buy_date[code] = date
                            trades.append({
                                "date": str(date.date()) if hasattr(date, "date") else str(date),
                                "symbol": code, "name": name, "action": "BUY",
                                "price": round(buy_price, 2), "shares": shares,
                                "reason": f"网格线{i} ({grid_price:.2f})", "pnl": 0,
                            })

        # ── 记录每日净值 ──
        total_pos_val = 0
        for sym_conf in symbols:
            code = sym_conf["code"]
            df = all_data.get(code)
            if df is not None and date in df.index:
                close_p = float(df.loc[date]["close"])
                for pos in positions[code].values():
                    total_pos_val += pos["shares"] * close_p
        daily_values.append({
            "date": str(date.date()) if hasattr(date, "date") else str(date),
            "value": round(cash + total_pos_val, 2),
        })

    # ── 计算回测指标 ──
    return _compute_metrics(daily_values, trades, initial_cash)


def _generate_mock_data(code: str, start: str, end: str, market: str) -> pd.DataFrame:
    """生成模拟K线数据（仅用于无真实数据时的演示）"""
    dates = pd.bdate_range(start, end)
    np.random.seed(hash(code) % 2**31)

    # 不同标的的起始价格和波动率
    price_map = {
        "0700.HK": (350.0, 0.02), "588000.SH": (0.95, 0.015),
        "563300.SH": (2.5, 0.012), "9868.HK": (45.0, 0.03), "TSLA": (240.0, 0.025),
    }
    base_price, vol = price_map.get(code, (100.0, 0.02))

    returns = np.random.normal(0, vol, len(dates))
    prices = base_price * np.cumprod(1 + returns)

    df = pd.DataFrame({
        "open": prices * (1 + np.random.uniform(-0.005, 0.005, len(dates))),
        "high": prices * (1 + np.abs(np.random.normal(0, vol * 0.5, len(dates)))),
        "low": prices * (1 - np.abs(np.random.normal(0, vol * 0.5, len(dates)))),
        "close": prices,
        "volume": np.random.randint(1000000, 50000000, len(dates)),
    }, index=dates)
    df.index.name = "date"
    return df


def _compute_metrics(daily_values: list, trades: list, initial_cash: float) -> dict:
    """计算回测指标"""
    if not daily_values:
        return _empty_result()

    values = [d["value"] for d in daily_values]
    final_value = values[-1]
    total_return = (final_value - initial_cash) / initial_cash * 100

    # 年化收益率
    days = len(values)
    years = max(days / 252, 0.1)
    annual_return = ((1 + total_return / 100) ** (1 / years) - 1) * 100

    # 最大回撤
    peak = values[0]
    max_dd = 0
    for v in values:
        if v > peak:
            peak = v
        dd = (peak - v) / peak * 100
        if dd > max_dd:
            max_dd = dd

    # 夏普比率（日收益率 → 年化）
    daily_returns = [(values[i] - values[i-1]) / values[i-1] for i in range(1, len(values)) if values[i-1] > 0]
    if daily_returns:
        avg_daily = np.mean(daily_returns)
        std_daily = np.std(daily_returns)
        sharpe = (avg_daily / std_daily) * np.sqrt(252) if std_daily > 0 else 0
    else:
        sharpe = 0

    # 胜率 & 盈亏比
    sell_trades = [t for t in trades if t["action"] == "SELL"]
    wins = [t for t in sell_trades if t["pnl"] > 0]
    losses = [t for t in sell_trades if t["pnl"] <= 0]
    win_rate = len(wins) / max(len(sell_trades), 1) * 100

    avg_win = np.mean([t["pnl"] for t in wins]) if wins else 0
    avg_loss = abs(np.mean([t["pnl"] for t in losses])) if losses else 1
    pl_ratio = avg_win / avg_loss if avg_loss > 0 else 0

    # 前后半段收益
    mid = len(values) // 2
    first_half = (values[mid] - values[0]) / values[0] * 100 if values[0] > 0 else 0
    second_half = (values[-1] - values[mid]) / values[mid] * 100 if values[mid] > 0 else 0

    return {
        "annual_return": round(annual_return, 2),
        "max_drawdown": round(max_dd, 2),
        "sharpe": round(sharpe, 2),
        "win_rate": round(win_rate, 1),
        "profit_loss_ratio": round(pl_ratio, 2),
        "total_trades": len(trades),
        "initial_cash": initial_cash,
        "final_value": round(final_value, 2),
        "total_return": round(total_return, 2),
        "sell_trades": len(sell_trades),
        "win_trades": len(wins),
        "loss_trades": len(losses),
        "first_half_return": round(first_half, 2),
        "second_half_return": round(second_half, 2),
        "period_returns": [round(daily_returns[i] * 100, 2) for i in range(0, len(daily_returns), max(1, len(daily_returns) // 10))],
    }


def _empty_result() -> dict:
    return {
        "annual_return": 0, "max_drawdown": 0, "sharpe": 0,
        "win_rate": 0, "profit_loss_ratio": 0, "total_trades": 0,
        "error": "无可用数据",
    }
