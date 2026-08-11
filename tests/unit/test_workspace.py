"""Tests for workspace management."""

from autostrategy.core.workspace import Workspace


def test_workspace_root(tmp_path):
    ws = Workspace(root=tmp_path)
    assert ws.root == tmp_path
    assert ws.root.exists()


def test_create_strategy(tmp_path):
    ws = Workspace(root=tmp_path)
    strategy = ws.create_strategy("dual-ma", market="A股")
    assert strategy.name == "dual-ma"
    assert (tmp_path / "dual-ma" / "config.yaml").exists()
    assert (tmp_path / "dual-ma" / "STRATEGY_DESIGN.md").exists()
    assert strategy.version == 1
    assert strategy.content_digest


def test_list_strategies(tmp_path):
    ws = Workspace(root=tmp_path)
    ws.create_strategy("s1")
    ws.create_strategy("s2")
    strategies = ws.list_strategies()
    assert len(strategies) == 2


def test_get_strategy(tmp_path):
    ws = Workspace(root=tmp_path)
    created = ws.create_strategy("s1")
    fetched = ws.get_strategy("s1")
    assert fetched is not None
    assert fetched.slug == created.slug


def test_delete_strategy(tmp_path):
    ws = Workspace(root=tmp_path)
    ws.create_strategy("s1")
    ws.delete_strategy("s1")
    assert ws.get_strategy("s1") is None


def test_strategy_file_api(tmp_path):
    ws = Workspace(root=tmp_path)
    ws.create_strategy("s1")

    strategy_dir = ws.get_strategy_dir("s1")
    assert strategy_dir == tmp_path / "s1"

    output_path = ws.write_text_file("s1", "nested/file.txt", "hello")
    assert output_path == tmp_path / "s1" / "nested" / "file.txt"
    assert ws.read_text_file("s1", "nested/file.txt") == "hello"


def test_strategy_file_api_missing_strategy(tmp_path):
    ws = Workspace(root=tmp_path)

    try:
        ws.get_strategy_dir("missing")
    except FileNotFoundError as exc:
        assert "missing" in str(exc)
    else:
        raise AssertionError("Expected FileNotFoundError")


def test_workspace_digest_and_version_follow_artifact_changes(tmp_path):
    ws = Workspace(root=tmp_path)
    created = ws.create_strategy("versioned")
    original_digest = ws.compute_strategy_digest(created.slug)

    assert original_digest == created.content_digest
    assert ws.compute_strategy_digest(created.slug) == original_digest

    ws.write_text_file(created.slug, "strategy.py", "def run_backtest(config):\n    return {}\n")
    changed_digest = ws.compute_strategy_digest(created.slug)
    updated = ws.bump_strategy_version(created.slug)

    assert changed_digest != original_digest
    assert updated.version == 2
    assert updated.content_digest == changed_digest


def test_workspace_loads_legacy_metadata_with_version_defaults(tmp_path):
    strategy_dir = tmp_path / "legacy"
    strategy_dir.mkdir()
    (strategy_dir / "strategy.yaml").write_text(
        "name: Legacy\nslug: legacy\nmarket: A股\nstatus: draft\n", encoding="utf-8"
    )

    strategy = Workspace(root=tmp_path).get_strategy("legacy")

    assert strategy is not None
    assert strategy.version == 1
    assert strategy.content_digest == ""
