"""Built-in strategy templates."""

from __future__ import annotations

import shutil
from pathlib import Path

TEMPLATES_DIR = Path(__file__).parent.parent / "templates"


class TemplateRegistry:
    """Registry of built-in strategy templates."""

    @staticmethod
    def list_templates() -> list[str]:
        """Return available template names."""
        if not TEMPLATES_DIR.exists():
            return []
        return sorted(
            entry.name
            for entry in TEMPLATES_DIR.iterdir()
            if entry.is_dir() and (entry / "STRATEGY_DESIGN.md").exists()
        )

    @classmethod
    def apply_template(cls, template_name: str, target_dir: Path) -> None:
        """Copy a template into a strategy workspace."""
        source = TEMPLATES_DIR / template_name
        if not source.exists():
            raise ValueError(f"Template '{template_name}' not found.")
        for item in source.iterdir():
            dest = target_dir / item.name
            if item.is_dir():
                shutil.copytree(item, dest, dirs_exist_ok=True)
            else:
                shutil.copy2(item, dest)
