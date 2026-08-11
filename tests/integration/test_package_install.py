"""Release metadata and base-runtime smoke tests."""

import subprocess
import sys
import tomllib
from pathlib import Path

import autostrategy

PROJECT_ROOT = Path(__file__).parents[2]


def test_public_version_sources_agree():
    pyproject = tomllib.loads((PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    version_file = (PROJECT_ROOT / "VERSION").read_text(encoding="utf-8").strip()

    assert version_file == pyproject["project"]["version"] == autostrategy.__version__


def test_base_runtime_can_import_cli_and_backtest_engine():
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            "import autostrategy.cli.main; import autostrategy.core.backtest_engine",
        ],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr


def test_declared_license_file_exists():
    license_text = (PROJECT_ROOT / "LICENSE").read_text(encoding="utf-8")

    assert "MIT License" in license_text
    assert "Permission is hereby granted" in license_text
