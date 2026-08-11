"""Codegen Agent: generate executable strategy files from STRATEGY_DESIGN.md."""

from __future__ import annotations

import ast
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

GENERATION_ORDER = (
    "config.yaml",
    "data/fetch_data.py",
    "strategy.py",
    "requirements.txt",
    "README.md",
)

FILE_INSTRUCTIONS = {
    "config.yaml": (
        "生成完整 YAML 配置。initial_cash、start_date、end_date、commission、slippage、"
        "market、symbols 必须位于 YAML 顶层；symbols 是策略可能下单的有限证券列表，"
        "仅作行情基准的指数放在 benchmark，不得放入 symbols。"
    ),
    "data/fetch_data.py": (
        "生成精简的数据适配器，公开 def fetch(config)，并且只调用 "
        "autostrategy.data.ftshare.fetch_daily_ohlc；只允许参数 symbol、limit、type_、"
        "start_date、end_date、client，禁止 fields 和 market。"
    ),
    "strategy.py": (
        "生成可编译、精简且完整的策略实现，公开 def run_backtest(config: dict) -> dict，"
        "本地回放入口 def run_paper(config: dict)，以及纯计算的 "
        "def generate_intents(context: dict) -> dict；generate_intents 不得发起"
        "网络请求或接触交易凭证，"
        "返回约定的全部指标字段；annual_return、max_drawdown、win_rate 使用百分数口径，"
        "数据失败时返回 error 或抛出异常，不得伪装成全零成功结果。"
    ),
    "requirements.txt": "生成最小依赖清单，至少包含 pandas 和 numpy。",
    "README.md": (
        "生成中文说明文档，必须包含 Markdown 标题、`## 策略概述` 和 `## 核心逻辑`。"
    ),
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
        """Generate and validate files one at a time in dependency order."""
        generated_files: dict[str, str] = {}
        for relative_path in GENERATION_ORDER:
            validation_errors: list[str] = []
            previous_content = ""
            for attempt in range(2):
                messages = [
                    ChatMessage(role="system", content=CODEGEN_SYSTEM_PROMPT),
                    ChatMessage(
                        role="user",
                        content=self._build_file_prompt(
                            design_text=design_text,
                            market=market,
                            relative_path=relative_path,
                            generated_files=generated_files,
                            validation_errors=validation_errors,
                            previous_content=previous_content,
                        ),
                    ),
                ]
                response = self.llm_client.chat(messages, temperature=0.2)
                try:
                    parsed_files = self._parse_generated_files(response)
                    content = parsed_files.get(relative_path)
                    if not content:
                        raise ValueError(f"Missing generated file: {relative_path}")
                except ValueError as exc:
                    validation_errors = [str(exc)]
                    previous_content = response
                else:
                    generated_files[relative_path] = content
                    report = self._check_generated_file(relative_path, generated_files)
                    if report.passed:
                        break
                    validation_errors = report.errors
                    previous_content = content

                if attempt == 1:
                    generated_files.pop(relative_path, None)
                    raise ValueError(
                        f"Generated file failed quality check after repair: {relative_path}: "
                        f"{'; '.join(validation_errors)}"
                    )

        return generated_files

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
        return workspace.bump_strategy_version(slug)

    def _build_file_prompt(
        self,
        design_text: str,
        market: str,
        relative_path: str,
        generated_files: dict[str, str],
        validation_errors: list[str],
        previous_content: str,
    ) -> str:
        """Build a focused prompt for one generated file or its repair."""
        parts = [
            f"目标市场：{market}",
            f"目标文件：{relative_path}",
            f"文件要求：{FILE_INSTRUCTIONS[relative_path]}",
            "只输出这个文件，使用 `=== FILE: path ===` 加 fenced code block 的格式。",
            "以下 STRATEGY_DESIGN.md 是策略逻辑的唯一来源：",
            design_text,
        ]
        if relative_path != "config.yaml" and "config.yaml" in generated_files:
            parts.extend(
                [
                    "已通过校验的 config.yaml（保持接口一致）：",
                    generated_files["config.yaml"],
                ]
            )
        if validation_errors:
            parts.extend(
                [
                    "上一次输出未通过校验。只修复当前目标文件，不要修改策略逻辑。",
                    "校验错误：",
                    "\n".join(f"- {error}" for error in validation_errors),
                    "上一次输出：",
                    previous_content,
                ]
            )
        return "\n".join(parts)

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

    def _check_generated_file(
        self, relative_path: str, files: dict[str, str]
    ) -> CodegenQualityReport:
        """Validate one file without requiring later generation steps to exist."""
        errors: list[str] = []
        warnings: list[str] = []

        if relative_path == "config.yaml":
            config_yaml = files.get(relative_path, "")
            try:
                parsed = yaml.safe_load(config_yaml) or {}
            except yaml.YAMLError as exc:
                errors.append(f"config.yaml is invalid YAML: {exc}")
                parsed = {}
            if not isinstance(parsed, dict):
                errors.append("config.yaml must contain a YAML mapping at the top level.")
                parsed = {}
            for key in [
                "initial_cash",
                "start_date",
                "end_date",
                "commission",
                "slippage",
                "market",
                "symbols",
            ]:
                if key not in parsed:
                    errors.append(f"config.yaml missing required key: {key}")
            symbols = parsed.get("symbols")
            if "symbols" in parsed and (
                not isinstance(symbols, list)
                or not symbols
                or not all(isinstance(item, str) and item.strip() for item in symbols)
            ):
                errors.append("config.yaml symbols must be a non-empty list of security codes.")

        elif relative_path == "data/fetch_data.py":
            fetch_data = files.get(relative_path, "")
            self._check_python_code_safety(fetch_data, relative_path, errors)
            try:
                tree = ast.parse(fetch_data, relative_path, "exec")
            except SyntaxError as exc:
                errors.append(f"data/fetch_data.py has syntax error: {exc}")
            else:
                self._check_ftshare_calls(tree, errors)
            if "def fetch(" not in fetch_data:
                errors.append("data/fetch_data.py must expose fetch(config).")

        elif relative_path == "strategy.py":
            strategy_py = files.get(relative_path, "")
            self._check_python_code_safety(strategy_py, relative_path, errors)
            try:
                compile(strategy_py, relative_path, "exec")
            except SyntaxError as exc:
                errors.append(f"strategy.py has syntax error: {exc}")
            if "def run_backtest(" not in strategy_py and "class Strategy" not in strategy_py:
                errors.append("strategy.py must expose run_backtest(config) or Strategy class.")
            if "def run_backtest(" not in strategy_py:
                warnings.append(
                    "strategy.py does not expose the recommended run_backtest(config) API."
                )
            if "def run_paper(" not in strategy_py:
                errors.append(
                    "strategy.py must expose run_paper(config) for local paper replay."
                )
            if "def generate_intents(" not in strategy_py:
                errors.append(
                    "strategy.py must expose generate_intents(context) for client simulation."
                )
            self._check_intent_entry_network_safety(strategy_py, errors)

        elif relative_path == "requirements.txt":
            requirements = files.get(relative_path, "")
            lower_requirements = requirements.lower()
            for package in ["pandas", "numpy"]:
                if package not in lower_requirements:
                    errors.append(f"requirements.txt missing required package: {package}")
            strategy_py = files.get("strategy.py", "")
            if "backtrader" in strategy_py.lower() and "backtrader" not in lower_requirements:
                errors.append("requirements.txt missing backtrader used by strategy.py")

        elif relative_path == "README.md":
            readme = files.get(relative_path, "")
            if "# " not in readme:
                errors.append("README.md must contain a title.")
            if "策略概述" not in readme and "核心逻辑" not in readme:
                errors.append("README.md must contain 策略概述 or 核心逻辑.")

        return CodegenQualityReport(passed=not errors, errors=errors, warnings=warnings)

    @staticmethod
    def _check_ftshare_calls(tree: ast.AST, errors: list[str]) -> None:
        """Reject keyword arguments that are not part of fetch_daily_ohlc's public API."""
        supported = {"limit", "type_", "start_date", "end_date", "client"}
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            function_name = ""
            if isinstance(node.func, ast.Name):
                function_name = node.func.id
            elif isinstance(node.func, ast.Attribute):
                function_name = node.func.attr
            if function_name != "fetch_daily_ohlc":
                continue
            unsupported = sorted(
                keyword.arg
                for keyword in node.keywords
                if keyword.arg is not None and keyword.arg not in supported
            )
            if unsupported:
                errors.append(
                    "data/fetch_data.py passes unsupported fetch_daily_ohlc arguments: "
                    + ", ".join(unsupported)
                )

    def check_generated_files(self, files: dict[str, str]) -> CodegenQualityReport:
        """Validate generated files before writing them to disk."""
        errors: list[str] = []
        warnings: list[str] = []

        missing = REQUIRED_GENERATED_FILES - set(files)
        for relative_path in sorted(missing):
            errors.append(f"Missing generated file: {relative_path}")

        self._check_python_code_safety(files.get("strategy.py", ""), "strategy.py", errors)
        self._check_python_code_safety(
            files.get("data/fetch_data.py", ""), "data/fetch_data.py", errors
        )

        strategy_py = files.get("strategy.py", "")
        if strategy_py:
            try:
                compile(strategy_py, "strategy.py", "exec")
            except SyntaxError as exc:
                errors.append(f"strategy.py has syntax error: {exc}")
            if "def run_backtest(" not in strategy_py and "class Strategy" not in strategy_py:
                errors.append("strategy.py must expose run_backtest(config) or Strategy class.")
            if "def run_backtest(" not in strategy_py:
                warnings.append(
                    "strategy.py does not expose the recommended run_backtest(config) API."
                )
            if "def run_paper(" not in strategy_py:
                errors.append(
                    "strategy.py must expose run_paper(config) for local paper replay."
                )
            if "def generate_intents(" not in strategy_py:
                errors.append(
                    "strategy.py must expose generate_intents(context) for client simulation."
                )
            self._check_intent_entry_network_safety(strategy_py, errors)

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
                "symbols",
            ]
            for key in required_config_keys:
                if key not in parsed:
                    errors.append(f"config.yaml missing required key: {key}")
            symbols = parsed.get("symbols")
            if "symbols" in parsed and (
                not isinstance(symbols, list)
                or not symbols
                or not all(isinstance(item, str) and item.strip() for item in symbols)
            ):
                errors.append("config.yaml symbols must be a non-empty list of security codes.")

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
                tree = ast.parse(fetch_data, "data/fetch_data.py", "exec")
            except SyntaxError as exc:
                errors.append(f"data/fetch_data.py has syntax error: {exc}")
            else:
                self._check_ftshare_calls(tree, errors)
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

    @staticmethod
    def _check_intent_entry_network_safety(code: str, errors: list[str]) -> None:
        """Keep generated strategy execution detached from broker/network concerns."""
        for pattern in ("import requests", "import httpx", "urllib.request", "websocket"):
            if pattern in code.lower():
                errors.append(
                    f"strategy.py generate_intents contract forbids network primitive '{pattern}'."
                )
