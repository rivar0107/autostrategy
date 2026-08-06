"""Tests for workspace path safety."""

import pytest

from autostrategy.core.workspace import Workspace


def test_resolve_strategy_path_allows_nested_file(tmp_path):
    workspace = Workspace(root=tmp_path)
    workspace.create_strategy("demo")

    resolved = workspace.resolve_strategy_path("demo", "nested/file.txt")

    assert resolved == (tmp_path / "demo" / "nested" / "file.txt").resolve()


@pytest.mark.parametrize(
    "relative_path",
    [
        "../evil.txt",
        "nested/../../evil.txt",
        "/tmp/evil.txt",
        "",
    ],
)
def test_resolve_strategy_path_rejects_unsafe_paths(tmp_path, relative_path):
    workspace = Workspace(root=tmp_path)
    workspace.create_strategy("demo")

    with pytest.raises(ValueError, match="Unsafe strategy file path"):
        workspace.resolve_strategy_path("demo", relative_path)


def test_write_text_file_rejects_path_traversal(tmp_path):
    workspace = Workspace(root=tmp_path)
    workspace.create_strategy("demo")

    with pytest.raises(ValueError, match="Unsafe strategy file path"):
        workspace.write_text_file("demo", "../evil.txt", "owned")

    assert not (tmp_path / "evil.txt").exists()
