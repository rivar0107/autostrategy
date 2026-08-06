"""Quality checks for STRATEGY_DESIGN.md."""

from __future__ import annotations

from dataclasses import dataclass, field


REQUIRED_SECTIONS = [
    "策略概述",
    "买入条件",
    "卖出条件",
    "止损",
    "仓位管理",
]


@dataclass
class QualityReport:
    """Result of a design document quality check."""

    passed: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


class DesignQualityCheck:
    """Check whether a STRATEGY_DESIGN.md meets the required structure."""

    @classmethod
    def check(cls, design_text: str) -> QualityReport:
        """Run all checks and return a report."""
        errors = []
        warnings = []

        for section in REQUIRED_SECTIONS:
            if f"## {section}" not in design_text:
                errors.append(f"Missing required section: {section}")

        if len(design_text.strip()) < 50:
            errors.append("Design document is too short.")

        if "未来函数" in design_text:
            warnings.append("Document mentions future functions — review carefully.")

        return QualityReport(passed=len(errors) == 0, errors=errors, warnings=warnings)
