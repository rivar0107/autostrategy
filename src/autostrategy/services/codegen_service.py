"""Code generation service."""

from __future__ import annotations

from pathlib import Path

from autostrategy.agents.codegen_agent import REQUIRED_GENERATED_FILES, CodegenAgent
from autostrategy.config import LLMConfig
from autostrategy.services.exceptions import (
    AutostrategyServiceError,
    LLMConfigurationRequiredError,
    StrategyNotFoundError,
    ValidationServiceError,
)
from autostrategy.services.models import CodegenResult, StrategySummary
from autostrategy.services.strategy_service import StrategyService


class CodegenService:
    """Application service for generating executable strategy files."""

    def __init__(self, workspace_root: Path | None = None, llm_config: LLMConfig | None = None) -> None:
        self.strategy_service = StrategyService(workspace_root=workspace_root)
        self.agent = CodegenAgent(llm_config=llm_config)

    def generate_code(self, slug: str, force: bool = False) -> CodegenResult:
        """Generate strategy implementation files for a strategy."""
        try:
            strategy = self.agent.codegen_and_save(
                workspace=self.strategy_service.workspace,
                slug=slug,
                force=force,
            )
        except FileNotFoundError as exc:
            raise StrategyNotFoundError(str(exc)) from exc
        except LLMConfigurationRequiredError:
            raise
        except (FileExistsError, ValueError, RuntimeError) as exc:
            raise ValidationServiceError(str(exc)) from exc
        except Exception as exc:
            raise AutostrategyServiceError(f"Code generation failed: {exc}") from exc
        return CodegenResult(
            strategy=StrategySummary.from_strategy(strategy),
            generated_files=sorted(REQUIRED_GENERATED_FILES),
        )
