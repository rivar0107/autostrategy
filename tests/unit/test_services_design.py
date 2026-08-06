"""Tests for design service error handling."""

from autostrategy.services.design_service import DesignService
from autostrategy.services.exceptions import ValidationServiceError


class FailingAgent:
    """Agent that simulates an LLM runtime failure."""

    def design_and_save(self, **kwargs):
        raise RuntimeError("No LLM API key found.")


def test_design_service_maps_runtime_error_to_validation(tmp_path):
    service = DesignService(workspace_root=tmp_path)
    service.agent = FailingAgent()

    try:
        service.create_design(name="demo", prompt="帮我做一个策略")
    except ValidationServiceError as exc:
        assert "No LLM API key" in exc.message
    else:
        raise AssertionError("Expected ValidationServiceError")
