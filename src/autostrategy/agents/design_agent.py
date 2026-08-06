"""Design Agent: generate STRATEGY_DESIGN.md from natural language."""

from __future__ import annotations

from autostrategy.agents.prompts.design import DESIGN_SYSTEM_PROMPT
from autostrategy.config import LLMConfig
from autostrategy.core.quality_check import DesignQualityCheck
from autostrategy.core.strategy import Strategy, StrategyStatus
from autostrategy.core.template_registry import TemplateRegistry
from autostrategy.core.workspace import Workspace
from autostrategy.llm.client import ChatMessage, LLMClient


class DesignAgent:
    """Agent that generates strategy design documents."""

    def __init__(self, llm_config: LLMConfig | None = None) -> None:
        self.llm_config = llm_config or LLMConfig()
        self.llm_client = LLMClient(self.llm_config)

    def design(
        self,
        prompt: str,
        market: str = "A股",
        template: str | None = None,
    ) -> str:
        """Generate a STRATEGY_DESIGN.md string from a user prompt."""
        messages = [
            ChatMessage(role="system", content=DESIGN_SYSTEM_PROMPT),
            ChatMessage(
                role="user",
                content=self._build_user_prompt(prompt, market, template),
            ),
        ]
        design_text = self.llm_client.chat(messages)
        return self._normalize(design_text)

    def _build_user_prompt(
        self,
        prompt: str,
        market: str,
        template: str | None,
    ) -> str:
        parts = [
            f"目标市场：{market}",
            f"用户需求：{prompt}",
        ]
        if template:
            template_names = TemplateRegistry.list_templates()
            if template in template_names:
                parts.append(f"参考模板：{template}")
        return "\n".join(parts)

    @staticmethod
    def _normalize(text: str) -> str:
        """Clean up LLM output."""
        text = text.strip()
        if text.startswith("```markdown"):
            text = text[len("```markdown") :]
        if text.startswith("```"):
            text = text[len("```") :]
        if text.endswith("```"):
            text = text[: -len("```")]
        return text.strip()

    def design_and_save(
        self,
        workspace: Workspace,
        name: str,
        prompt: str,
        market: str = "A股",
        template: str | None = None,
    ) -> Strategy:
        """Create a strategy workspace, generate design, and save."""
        strategy = workspace.create_strategy(name, market=market, template=template)
        design_text = self.design(prompt, market=market, template=template)

        report = DesignQualityCheck.check(design_text)
        if not report.passed:
            workspace.delete_strategy(strategy.slug)
            raise ValueError(
                f"Generated design failed quality check: {'; '.join(report.errors)}"
            )

        design_path = workspace._strategy_dir(strategy.slug) / "STRATEGY_DESIGN.md"
        design_path.write_text(design_text, encoding="utf-8")
        workspace.update_strategy_status(strategy.slug, StrategyStatus.DESIGNED)
        updated = workspace.get_strategy(strategy.slug)
        if updated is None:
            raise RuntimeError(f"Strategy '{strategy.slug}' disappeared after saving design.")
        return updated
