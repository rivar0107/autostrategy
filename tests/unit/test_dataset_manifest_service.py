"""Tests for frozen research dataset manifests."""

from datetime import date
from pathlib import Path

import pandas as pd
import yaml

from autostrategy.core.research import DateRange
from autostrategy.core.workspace import Workspace
from autostrategy.services.dataset_manifest_service import DatasetManifestService
from autostrategy.services.version_service import VersionService


def _create_data_strategy(tmp_path, *, mapping: bool = False):
    workspace = Workspace(root=tmp_path)
    strategy = workspace.create_strategy("dataset-demo")
    workspace.write_text_file(
        strategy.slug,
        "config.yaml",
        "market: A股\n"
        "symbols:\n  - 510300.SH\n"
        "commission: 0.0003\n"
        "slippage: 0.001\n"
        "period:\n  start: 2020-01-01\n  end: 2025-12-31\n",
    )
    body = (
        "import pandas as pd\n\n"
        "def fetch(config):\n"
        "    dates = pd.date_range('2020-01-01', '2025-12-31', freq='D')\n"
        "    frame = pd.DataFrame({\n"
        "        'open': range(len(dates)), 'high': range(len(dates)),\n"
        "        'low': range(len(dates)), 'close': range(len(dates)),\n"
        "        'volume': [1000] * len(dates),\n"
        "    }, index=dates)\n"
        "    frame.index.name = 'date'\n"
    )
    body += (
        "    return {'510300.SH': frame, '000300.SH': frame.copy()}\n"
        if mapping
        else "    return frame\n"
    )
    workspace.write_text_file(strategy.slug, "data/fetch_data.py", body)
    workspace.write_text_file(
        strategy.slug,
        "strategy.py",
        "def run_backtest(config):\n    return {}\n",
    )
    workspace.refresh_strategy_digest(strategy.slug)
    version = VersionService(workspace_root=tmp_path).ensure_current_version(strategy.slug)
    return workspace, strategy.slug, version.version_id


def _splits():
    return {
        "train": DateRange(start=date(2020, 1, 1), end=date(2022, 12, 31)),
        "validation": DateRange(start=date(2023, 1, 1), end=date(2024, 12, 31)),
        "test": DateRange(start=date(2025, 1, 1), end=date(2025, 12, 31)),
    }


def test_capture_freezes_dataframe_and_persists_manifest(tmp_path):
    _, slug, version_id = _create_data_strategy(tmp_path)
    service = DatasetManifestService(workspace_root=tmp_path)

    manifest = service.capture(
        slug,
        version_id,
        benchmark="000300.SH",
        data_source="fixture",
        **_splits(),
    )
    restarted = DatasetManifestService(workspace_root=tmp_path)

    assert manifest.locked is True
    assert manifest.output_type == "dataframe"
    assert manifest.symbols == ["510300.SH"]
    assert len(manifest.data_digest) == 64
    assert (manifest.snapshot_path / "manifest.json").exists()
    assert (manifest.snapshot_path / next(iter(manifest.snapshot_files.values()))).exists()
    assert restarted.get_manifest(slug, manifest.manifest_id) == manifest


def test_capture_supports_mapping_of_frames(tmp_path):
    _, slug, version_id = _create_data_strategy(tmp_path, mapping=True)
    service = DatasetManifestService(workspace_root=tmp_path)

    manifest = service.capture(
        slug,
        version_id,
        benchmark="000300.SH",
        data_source="fixture",
        **_splits(),
    )

    assert manifest.output_type == "mapping"
    assert set(manifest.snapshot_files) == {"510300.SH", "000300.SH"}
    assert len(set(manifest.snapshot_files.values())) == 2


def test_materialized_split_uses_frozen_adapter_and_normalized_dates(tmp_path):
    workspace, slug, version_id = _create_data_strategy(tmp_path, mapping=True)
    service = DatasetManifestService(workspace_root=tmp_path)
    manifest = service.capture(
        slug,
        version_id,
        benchmark="000300.SH",
        data_source="fixture",
        **_splits(),
    )
    workspace.write_text_file(
        slug,
        "data/fetch_data.py",
        "def fetch(config):\n    raise RuntimeError('upstream must not be called')\n",
    )
    destination = tmp_path / "materialized-validation"

    service.materialize_split(
        slug,
        manifest.manifest_id,
        version_id,
        "validation",
        destination,
    )

    config = yaml.safe_load((destination / "config.yaml").read_text(encoding="utf-8"))
    assert str(config["start_date"]) == "2023-01-01"
    assert str(config["end_date"]) == "2024-12-31"
    assert str(config["period"]["start"]) == "2023-01-01"
    assert str(config["period"]["end"]) == "2024-12-31"
    assert config["commission"] == 0.0003
    assert config["slippage"] == 0.001

    fetch = _load_fetch(destination / "data" / "fetch_data.py")
    frames = fetch(config)
    assert set(frames) == {"510300.SH", "000300.SH"}
    for frame in frames.values():
        assert frame.index.min() == pd.Timestamp("2023-01-01")
        assert frame.index.max() == pd.Timestamp("2024-12-31")


def _load_fetch(path: Path):
    namespace = {"__file__": str(path), "__name__": "frozen_fetch_test"}
    exec(compile(path.read_text(encoding="utf-8"), str(path), "exec"), namespace)
    return namespace["fetch"]
