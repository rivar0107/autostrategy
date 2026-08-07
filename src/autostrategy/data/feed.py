"""Local mock market-data feed for Phase 5D paper-trading replays.

A feed reads bars from a local CSV or JSONL file and yields normalized
bar events in chronological order. Bar schema:

    {"at": ISO-8601 str, "symbol": str, "open": float, "high": float,
     "low": float, "close": float, "volume": float}

Feeds are pure local fixtures — they never touch the network, so replay
results are reproducible. ``autostrategy.data.ftshare`` remains the way
to *download* data into a fixture file.
"""

from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Iterator

REQUIRED_BAR_FIELDS = ("at", "symbol", "open", "high", "low", "close", "volume")

_AT_ALIASES = ("at", "timestamp", "date", "datetime", "time")


def normalize_bar(raw: dict[str, Any]) -> dict[str, Any]:
    """Normalize one raw record into the canonical bar schema."""
    at = next((raw[key] for key in _AT_ALIASES if raw.get(key) is not None), None)
    if at is None:
        raise ValueError(f"bar 缺少时间字段（支持 {_AT_ALIASES}）: {raw}")
    bar = {
        "at": _to_iso(at),
        "symbol": str(raw.get("symbol", "")).strip(),
        "open": float(raw["open"]),
        "high": float(raw["high"]),
        "low": float(raw["low"]),
        "close": float(raw["close"]),
        "volume": float(raw.get("volume", 0) or 0),
    }
    if not bar["symbol"]:
        raise ValueError(f"bar 缺少 symbol: {raw}")
    return bar


def _to_iso(value: Any) -> str:
    if isinstance(value, datetime):
        return value.isoformat()
    text = str(value).strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        return datetime.fromisoformat(text).isoformat()
    except ValueError:
        pass
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y%m%d", "%Y/%m/%d"):
        try:
            return datetime.strptime(text, fmt).isoformat()
        except ValueError:
            continue
    raise ValueError(f"无法解析 bar 时间: {value!r}")


def _parse_at(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00")).replace(tzinfo=None)


def _iter_records(path: Path) -> Iterable[dict[str, Any]]:
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        with open(path, encoding="utf-8") as file:
            for line in file:
                line = line.strip()
                if line:
                    yield json.loads(line)
    elif suffix == ".csv":
        with open(path, encoding="utf-8", newline="") as file:
            yield from csv.DictReader(file)
    else:
        raise ValueError(f"不支持的 feed 文件格式: {path.name}（仅支持 .csv / .jsonl）")


def load_bars(
    path: str | Path,
    symbols: list[str] | None = None,
    start: str | None = None,
    end: str | None = None,
) -> list[dict[str, Any]]:
    """Load bars from a CSV/JSONL fixture, filtered and sorted by time."""
    feed_path = Path(path)
    if not feed_path.exists():
        raise FileNotFoundError(f"feed 文件不存在: {feed_path}")
    wanted = {str(s) for s in symbols} if symbols else None
    start_at = _parse_at(_to_iso(start)) if start else None
    end_at = _parse_at(_to_iso(end)) if end else None

    bars = [normalize_bar(record) for record in _iter_records(feed_path)]
    if wanted is not None:
        bars = [bar for bar in bars if bar["symbol"] in wanted]
    if start_at is not None:
        bars = [bar for bar in bars if _parse_at(bar["at"]) >= start_at]
    if end_at is not None:
        bars = [bar for bar in bars if _parse_at(bar["at"]) <= end_at]
    bars.sort(key=lambda bar: (_parse_at(bar["at"]), bar["symbol"]))
    return bars


class LocalFeed:
    """A replayable local market-data feed over a fixture file."""

    def __init__(
        self,
        path: str | Path,
        symbols: list[str] | None = None,
        start: str | None = None,
        end: str | None = None,
    ) -> None:
        self.path = Path(path)
        self._bars = load_bars(self.path, symbols=symbols, start=start, end=end)

    def __iter__(self) -> Iterator[dict[str, Any]]:
        return iter(self._bars)

    def __len__(self) -> int:
        return len(self._bars)

    @property
    def bars(self) -> list[dict[str, Any]]:
        return list(self._bars)

    def metadata(self) -> dict[str, Any]:
        """Feed summary shown in paper-run results and the frontend."""
        symbols = sorted({bar["symbol"] for bar in self._bars})
        return {
            "source": str(self.path),
            "bar_count": len(self._bars),
            "symbol_count": len(symbols),
            "symbols": symbols,
            "start": self._bars[0]["at"] if self._bars else None,
            "end": self._bars[-1]["at"] if self._bars else None,
        }
