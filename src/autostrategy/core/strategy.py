"""Strategy domain model."""

from __future__ import annotations

import re
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path

from pydantic import BaseModel, Field


class StrategyStatus(str, Enum):
    """Strategy lifecycle status."""

    DRAFT = "draft"
    DESIGNED = "designed"
    CODED = "coded"
    BACKTESTED = "backtested"
    PAPER_RUNNING = "paper_running"
    OPTIMIZED = "optimized"
    ACTIVE = "active"
    ARCHIVED = "archived"


class Strategy(BaseModel):
    """A single trading strategy."""

    name: str
    slug: str = ""
    description: str = ""
    market: str = "A股"
    status: StrategyStatus = StrategyStatus.DRAFT
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    template: str | None = None
    tags: list[str] = Field(default_factory=list)

    def model_post_init(self, __context) -> None:
        if not self.slug:
            self.slug = self._to_slug(self.name)

    @staticmethod
    def _to_slug(name: str) -> str:
        """Convert a human-readable name to a URL-safe slug."""
        slug = name.lower().strip()
        slug = re.sub(r"[^\w\s-]", "", slug)
        slug = re.sub(r"[-\s]+", "-", slug)
        return slug.strip("-")

    @property
    def workspace_dir(self) -> Path:
        """Return the relative workspace directory name."""
        return Path(self.slug)
