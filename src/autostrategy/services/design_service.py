"""Design generation service."""

from __future__ import annotations

from pathlib import Path

from autostrategy.agents.design_agent import DesignAgent
from autostrategy.config import LLMConfig
from autostrategy.services.exceptions import (
    AutostrategyServiceError,
    LLMConfigurationRequiredError,
    ValidationServiceError,
)
from autostrategy.services.models import DesignResult, StrategySummary
from autostrategy.services.strategy_service import StrategyService


class DesignService:
    """Application service for natural-language strategy design."""

    def __init__(self, workspace_root: Path | None = None, llm_config: LLMConfig | None = None) -> None:
        self.strategy_service = StrategyService(workspace_root=workspace_root)
        self.agent = DesignAgent(llm_config=llm_config)

    def create_design(
        self,
        name: str,
        prompt: str,
        market: str = "A股",
        template: str | None = None,
    ) -> DesignResult:
        """Create a strategy workspace and generate STRATEGY_DESIGN.md."""
        try:
            strategy = self.agent.design_and_save(
                workspace=self.strategy_service.workspace,
                name=name,
                prompt=prompt,
                market=market,
                template=template,
            )
        except LLMConfigurationRequiredError:
            raise
        except ValueError as exc:
            raise ValidationServiceError(str(exc)) from exc
        except RuntimeError as exc:
            raise ValidationServiceError(str(exc)) from exc
        except Exception as exc:
            raise AutostrategyServiceError(f"Design generation failed: {exc}") from exc
        paths = self.strategy_service.get_strategy_paths(strategy.slug)
        return DesignResult(
            strategy=StrategySummary.from_strategy(strategy),
            design_path=paths.design,
        )
