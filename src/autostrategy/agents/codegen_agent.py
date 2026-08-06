"""Codegen Agent: generate executable strategy files from STRATEGY_DESIGN.md."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import yaml

from autostrategy.agents.prompts.codegen import CODEGEN_SYSTEM_PROMPT
from autostrategy.config import LLMConfig
from autostrategy.core.strategy import Strategy, StrategyStatus
from autostrategy.core.workspace import Workspace
from autostrategy.llm.client import ChatMessage, LLMClient


ALLOWED_GENERATED_FILES = {
    "strategy.py",
    "config.yaml",
    "README.md",
    "requirements.txt",
    "data/fetch_data.py",
}

REQUIRED_GENERATED_FILES = {
    "strategy.py",
    "config.yaml",
    "README.md",
    "requirements.txt",
    "data/fetch_data.py",
}

DANGEROUS_CODE_PATTERNS = {
    "os.system": "shell command execution",
    "subprocess": "subprocess execution",
    "socket": "raw network access",
    "eval(": "dynamic code evaluation",
    "exec(": "dynamic code execution",
    "shutil.rmtree": "recursive file deletion",
}


@dataclass
class CodegenQualityReport:
    """Static validation result for generated strategy files."""

    passed: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


class CodegenAgent:
    """Agent that generates strategy implementation files."""

    def __init__(self, llm_config: LLMConfig | None = None) -> None:
        self.llm_config = llm_config or LLMConfig()
        self.llm_client = LLMClient(self.llm_config)

    def codegen(self, design_text: str, market: str = "A股") -> dict[str, str]:
        """Generate file contents from a strategy design document."""
        messages = [
            ChatMessage(role="system", content=CODEGEN_SYSTEM_PROMPT),
            ChatMessage(role="user", content=self._build_user_prompt(design_text, market)),
        ]
        response = self.llm_client.chat(messages)
        return self._parse_generated_files(response)

    def codegen_and_save(self, workspace: Workspace, slug: str, force: bool = False) -> Strategy:
        """Generate and save code files for an existing strategy workspace."""
        strategy = workspace.get_strategy(slug)
        if strategy is None:
            raise FileNotFoundError(f"Strategy '{slug}' not found.")

        design_text = workspace.read_text_file(slug, "STRATEGY_DESIGN.md")
        if not design_text.strip() or "待补充" in design_text:
            raise ValueError("STRATEGY_DESIGN.md is missing or incomplete.")

        generated_files = self.codegen(design_text, market=strategy.market)
        report = self.check_generated_files(generated_files)
        if not report.passed:
            raise ValueError(f"Generated files failed quality check: {'; '.join(report.errors)}")

        strategy_dir = workspace.get_strategy_dir(slug)
        for relative_path, content in generated_files.items():
            output_path = strategy_dir / relative_path
            if relative_path == "strategy.py" and output_path.exists() and not force:
                raise FileExistsError(
                    f"Generated file already exists: {relative_path}. Use force=True to overwrite."
                )

        for relative_path, content in generated_files.items():
            workspace.write_text_file(slug, relative_path, content)

        workspace.update_strategy_status(slug, StrategyStatus.CODED)
        updated = workspace.get_strategy(slug)
        if updated is None:
            raise RuntimeError(f"Strategy '{slug}' disappeared after codegen.")
        return updated

    def _build_user_prompt(self, design_text: str, market: str) -> str:
        return "\n".join([
            f"目标市场：{market}",
            "请严格根据以下 STRATEGY_DESIGN.md 生成文件：",
            design_text,
        ])

    def _parse_generated_files(self, text: str) -> dict[str, str]:
        """Parse LLM output into a filename -> content mapping."""
        pattern = re.compile(
            r"^=== FILE: (?P<path>.+?) ===\s*\n```(?:\w+)?\s*\n(?P<content>.*?)\n```",
            re.MULTILINE | re.DOTALL,
        )
        files: dict[str, str] = {}
        for match in pattern.finditer(text):
            relative_path = match.group("path").strip()
            if relative_path not in ALLOWED_GENERATED_FILES:
                raise ValueError(f"Generated file path is not allowed: {relative_path}")
            files[relative_path] = match.group("content").strip() + "\n"
        return files

    def check_generated_files(self, files: dict[str, str]) -> CodegenQualityReport:
        """Validate generated files before writing them to disk."""
        errors: list[str] = []
        warnings: list[str] = []

        missing = REQUIRED_GENERATED_FILES - set(files)
        for relative_path in sorted(missing):
            errors.append(f"Missing generated file: {relative_path}")

        self._check_python_code_safety(files.get("strategy.py", ""), "strategy.py", errors)
        self._check_python_code_safety(files.get("data/fetch_data.py", ""), "data/fetch_data.py", errors)

        strategy_py = files.get("strategy.py", "")
        if strategy_py:
            try:
                compile(strategy_py, "strategy.py", "exec")
            except SyntaxError as exc:
                errors.append(f"strategy.py has syntax error: {exc}")
            if "def run_backtest(" not in strategy_py and "class Strategy" not in strategy_py:
                errors.append("strategy.py must expose run_backtest(config) or Strategy class.")
            if "def run_backtest(" not in strategy_py:
                warnings.append("strategy.py does not expose the recommended run_backtest(config) API.")

        config_yaml = files.get("config.yaml", "")
        if config_yaml:
            try:
                parsed = yaml.safe_load(config_yaml) or {}
            except yaml.YAMLError as exc:
                errors.append(f"config.yaml is invalid YAML: {exc}")
                parsed = {}
            required_config_keys = [
                "initial_cash",
                "start_date",
                "end_date",
                "commission",
                "slippage",
                "market",
            ]
            for key in required_config_keys:
                if key not in parsed:
                    errors.append(f"config.yaml missing required key: {key}")

        readme = files.get("README.md", "")
        if readme and "# " not in readme:
            errors.append("README.md must contain a title.")
        if readme and "策略概述" not in readme and "核心逻辑" not in readme:
            errors.append("README.md must contain 策略概述 or 核心逻辑.")

        requirements = files.get("requirements.txt", "")
        if requirements:
            lower_requirements = requirements.lower()
            for package in ["pandas", "numpy"]:
                if package not in lower_requirements:
                    errors.append(f"requirements.txt missing required package: {package}")
            if "backtrader" in strategy_py.lower() and "backtrader" not in lower_requirements:
                errors.append("requirements.txt missing backtrader used by strategy.py")

        fetch_data = files.get("data/fetch_data.py", "")
        if fetch_data:
            try:
                compile(fetch_data, "data/fetch_data.py", "exec")
            except SyntaxError as exc:
                errors.append(f"data/fetch_data.py has syntax error: {exc}")
            if "def fetch(" not in fetch_data:
                errors.append("data/fetch_data.py must expose fetch(config).")

        return CodegenQualityReport(passed=not errors, errors=errors, warnings=warnings)

    def _check_python_code_safety(self, code: str, relative_path: str, errors: list[str]) -> None:
        """Reject generated Python code with clearly dangerous primitives."""
        if not code:
            return
        for pattern, reason in DANGEROUS_CODE_PATTERNS.items():
            if pattern in code:
                errors.append(f"{relative_path} contains dangerous pattern '{pattern}': {reason}")
