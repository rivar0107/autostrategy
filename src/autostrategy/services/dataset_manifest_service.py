"""Capture immutable strategy datasets and materialize isolated research splits."""

from __future__ import annotations

import hashlib
import json
import shutil
import sys
import types
from pathlib import Path
from tempfile import mkdtemp
from typing import Any, Literal
from uuid import uuid4

import pandas as pd
import yaml

from autostrategy.core.research import DatasetManifest, DateRange
from autostrategy.persistence.research_store import ResearchStore
from autostrategy.services.exceptions import ValidationServiceError
from autostrategy.services.version_service import VersionService


class DatasetManifestService:
    """Freeze one upstream fetch and reuse it across all experiment phases."""

    def __init__(self, workspace_root: Path | None = None) -> None:
        self.version_service = VersionService(workspace_root=workspace_root)
        self.workspace = self.version_service.workspace
        self.store = ResearchStore(self.workspace.root)

    def capture(
        self,
        slug: str,
        version_id: str,
        *,
        train: DateRange,
        validation: DateRange,
        test: DateRange,
        benchmark: str,
        data_source: str = "strategy_fetch",
        frequency: str = "daily",
        adjustment: str = "forward",
        commission: float | None = None,
        slippage: float | None = None,
    ) -> DatasetManifest:
        """Execute a versioned fetch adapter once and persist canonical CSV bytes."""
        version = self.version_service.get_version(slug, version_id)
        manifest_id = uuid4().hex
        datasets_root = self.workspace.resolve_strategy_path(slug, ".autostrategy/datasets")
        datasets_root.mkdir(parents=True, exist_ok=True)
        target = datasets_root / manifest_id
        temp = Path(mkdtemp(prefix=".dataset-", dir=datasets_root))
        try:
            version_workspace = temp / "version"
            self.version_service.materialize_version(slug, version.version_id, version_workspace)
            config_path = version_workspace / "config.yaml"
            config = _load_config(config_path)
            full_range = DateRange(start=train.start, end=test.end)
            config = normalize_split_config(config, full_range)
            raw = self._execute_fetch(version_workspace, config)
            frames, output_type = _normalize_frames(raw, config, full_range)

            frames_dir = temp / "frames"
            frames_dir.mkdir()
            snapshot_files: dict[str, str] = {}
            for index, (key, frame) in enumerate(frames.items()):
                relative_path = f"frames/frame-{index}.csv"
                (temp / relative_path).write_text(
                    frame.to_csv(index=True, index_label="date", date_format="%Y-%m-%d"),
                    encoding="utf-8",
                )
                snapshot_files[key] = relative_path

            resolved_commission = float(
                config.get("commission", 0.0) if commission is None else commission
            )
            resolved_slippage = float(
                config.get("slippage", 0.0) if slippage is None else slippage
            )
            digest = _dataset_digest(
                temp,
                snapshot_files,
                {
                    "strategy_slug": slug,
                    "version_id": version_id,
                    "data_source": data_source,
                    "frequency": frequency,
                    "adjustment": adjustment,
                    "benchmark": benchmark,
                    "commission": resolved_commission,
                    "slippage": resolved_slippage,
                    "train": train.model_dump(mode="json"),
                    "validation": validation.model_dump(mode="json"),
                    "test": test.model_dump(mode="json"),
                    "output_type": output_type,
                    "snapshot_files": snapshot_files,
                },
            )
            manifest = DatasetManifest(
                manifest_id=manifest_id,
                strategy_slug=slug,
                version_id=version_id,
                data_source=data_source,
                symbols=list(frames),
                frequency=frequency,
                adjustment=adjustment,
                benchmark=benchmark,
                commission=resolved_commission,
                slippage=resolved_slippage,
                train=train,
                validation=validation,
                test=test,
                snapshot_path=target,
                snapshot_files=snapshot_files,
                output_type=output_type,
                data_digest=digest,
            )
            shutil.rmtree(version_workspace)
            (temp / "manifest.json").write_text(
                json.dumps(manifest.model_dump(mode="json"), ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            temp.replace(target)
            try:
                self.store.create_manifest(manifest)
            except Exception:
                shutil.rmtree(target)
                raise
            return manifest
        finally:
            if temp.exists():
                shutil.rmtree(temp)

    def get_manifest(self, slug: str, manifest_id: str) -> DatasetManifest:
        manifest = self.store.get_manifest(slug, manifest_id)
        if manifest is None:
            raise ValidationServiceError(f"Dataset manifest '{manifest_id}' not found.")
        self._verify_manifest(manifest)
        return manifest

    def list_manifests(self, slug: str) -> list[DatasetManifest]:
        self.version_service.ensure_current_version(slug)
        manifests = self.store.list_manifests(slug)
        for manifest in manifests:
            self._verify_manifest(manifest)
        return manifests

    def materialize_split(
        self,
        slug: str,
        manifest_id: str,
        version_id: str,
        split: Literal["train", "validation", "test"],
        destination: Path,
    ) -> Path:
        """Build an executable version workspace backed only by frozen CSV files."""
        manifest = self.get_manifest(slug, manifest_id)
        if manifest.version_id != version_id:
            version = self.version_service.get_version(slug, version_id)
            if version.strategy_slug != manifest.strategy_slug:
                raise ValidationServiceError("Version and dataset belong to different strategies.")
        self.version_service.materialize_version(slug, version_id, destination)
        frozen_dir = destination / "data" / "frozen"
        frozen_dir.mkdir(parents=True, exist_ok=True)
        adapter_files: dict[str, str] = {}
        for key, relative_path in manifest.snapshot_files.items():
            source = manifest.snapshot_path / relative_path
            target_name = Path(relative_path).name
            shutil.copy2(source, frozen_dir / target_name)
            adapter_files[key] = target_name

        adapter_path = destination / "data" / "fetch_data.py"
        adapter_path.parent.mkdir(parents=True, exist_ok=True)
        adapter_path.write_text(
            _frozen_adapter_source(manifest.output_type, adapter_files), encoding="utf-8"
        )
        config_path = destination / "config.yaml"
        config = normalize_split_config(_load_config(config_path), getattr(manifest, split))
        config["commission"] = manifest.commission
        config["slippage"] = manifest.slippage
        config["benchmark"] = manifest.benchmark
        config_path.write_text(
            yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding="utf-8"
        )
        return destination

    @staticmethod
    def _execute_fetch(strategy_dir: Path, config: dict[str, Any]) -> Any:
        fetch_path = strategy_dir / "data" / "fetch_data.py"
        if not fetch_path.exists():
            raise ValidationServiceError("Version does not contain data/fetch_data.py.")
        module_name = f"dataset_fetch_{uuid4().hex}"
        module = types.ModuleType(module_name)
        module.__file__ = str(fetch_path)
        sys.modules[module_name] = module
        sys.path.insert(0, str(strategy_dir))
        try:
            source = fetch_path.read_text(encoding="utf-8")
            exec(compile(source, str(fetch_path), "exec"), module.__dict__)
            if not hasattr(module, "fetch"):
                raise ValidationServiceError("data/fetch_data.py must expose fetch(config).")
            return module.fetch(config)
        finally:
            sys.modules.pop(module_name, None)
            try:
                sys.path.remove(str(strategy_dir))
            except ValueError:
                pass

    @staticmethod
    def _verify_manifest(manifest: DatasetManifest) -> None:
        actual = _dataset_digest(
            manifest.snapshot_path,
            manifest.snapshot_files,
            {
                "strategy_slug": manifest.strategy_slug,
                "version_id": manifest.version_id,
                "data_source": manifest.data_source,
                "frequency": manifest.frequency,
                "adjustment": manifest.adjustment,
                "benchmark": manifest.benchmark,
                "commission": manifest.commission,
                "slippage": manifest.slippage,
                "train": manifest.train.model_dump(mode="json"),
                "validation": manifest.validation.model_dump(mode="json"),
                "test": manifest.test.model_dump(mode="json"),
                "output_type": manifest.output_type,
                "snapshot_files": manifest.snapshot_files,
            },
        )
        if actual != manifest.data_digest:
            raise ValidationServiceError(
                f"Dataset manifest digest mismatch for '{manifest.manifest_id}'."
            )


def normalize_split_config(config: dict[str, Any], split: DateRange) -> dict[str, Any]:
    """Apply one date boundary to both supported generated-config shapes."""
    normalized = dict(config)
    start = split.start.isoformat()
    end = split.end.isoformat()
    normalized["start_date"] = start
    normalized["end_date"] = end
    period = dict(normalized.get("period") or {})
    period["start"] = start
    period["end"] = end
    normalized["period"] = period
    return normalized


def _load_config(path: Path) -> dict[str, Any]:
    config = yaml.safe_load(path.read_text(encoding="utf-8")) if path.exists() else {}
    if not isinstance(config, dict):
        raise ValidationServiceError("config.yaml must contain a mapping.")
    return config


def _normalize_frames(
    raw: Any,
    config: dict[str, Any],
    full_range: DateRange,
) -> tuple[dict[str, pd.DataFrame], Literal["dataframe", "mapping"]]:
    if isinstance(raw, pd.DataFrame):
        symbols = _config_symbols(config)
        frames = {symbols[0] if symbols else "primary": raw}
        output_type: Literal["dataframe", "mapping"] = "dataframe"
    elif isinstance(raw, dict) and raw and all(
        isinstance(frame, pd.DataFrame) for frame in raw.values()
    ):
        frames = {str(key): frame for key, frame in raw.items()}
        output_type = "mapping"
    else:
        raise ValidationServiceError(
            "fetch(config) must return a DataFrame or a non-empty mapping of DataFrames."
        )

    normalized: dict[str, pd.DataFrame] = {}
    start = pd.Timestamp(full_range.start)
    end = pd.Timestamp(full_range.end)
    for key, source in frames.items():
        frame = source.copy()
        if "date" in frame.columns:
            frame = frame.set_index("date")
        frame.index = pd.to_datetime(frame.index)
        frame = frame.sort_index().loc[start:end]
        if frame.empty:
            raise ValidationServiceError(f"Frozen dataset frame '{key}' is empty.")
        frame.index.name = "date"
        normalized[key] = frame
    return normalized, output_type


def _config_symbols(config: dict[str, Any]) -> list[str]:
    raw = config.get("symbols") or config.get("symbol") or []
    if isinstance(raw, str):
        return [raw]
    if isinstance(raw, list):
        return [str(item.get("code") if isinstance(item, dict) else item) for item in raw]
    return []


def _dataset_digest(
    root: Path,
    snapshot_files: dict[str, str],
    contract: dict[str, Any],
) -> str:
    digest = hashlib.sha256()
    digest.update(
        json.dumps(contract, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
    )
    for key, relative_path in sorted(snapshot_files.items()):
        digest.update(key.encode("utf-8"))
        digest.update((root / relative_path).read_bytes())
    return digest.hexdigest()


def _frozen_adapter_source(output_type: str, snapshot_files: dict[str, str]) -> str:
    files_json = json.dumps(snapshot_files, ensure_ascii=False, sort_keys=True)
    return f'''"""Generated immutable dataset adapter for one research experiment."""

from pathlib import Path

import pandas as pd

_FILES = {files_json}
_OUTPUT_TYPE = {output_type!r}


def _bounds(config):
    period = config.get("period") if isinstance(config.get("period"), dict) else {{}}
    start = config.get("start_date") or period.get("start")
    end = config.get("end_date") or period.get("end")
    return start, end


def fetch(config):
    """Read and date-filter the locked local snapshot without upstream access."""
    start, end = _bounds(config)
    frames = {{}}
    for key, filename in _FILES.items():
        frame = pd.read_csv(
            Path(__file__).parent / "frozen" / filename,
            index_col=0,
            parse_dates=[0],
        )
        frame.index.name = "date"
        if start:
            frame = frame.loc[pd.Timestamp(start):]
        if end:
            frame = frame.loc[:pd.Timestamp(end)]
        frames[key] = frame
    return next(iter(frames.values())) if _OUTPUT_TYPE == "dataframe" else frames
'''
