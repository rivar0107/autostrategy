"""Strategy domain model."""

from __future__ import annotations

import re
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path

from pydantic import BaseModel, Field


class StrategyStatus(StrEnum):
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
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    template: str | None = None
    tags: list[str] = Field(default_factory=list)
    version: int = Field(default=1, ge=1)
    content_digest: str = ""
    current_version_id: str | None = None
    active_version_id: str | None = None

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
