# Optimization Ratchet Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:test-driven-development to implement each task and superpowers:verification-before-completion before claiming success.

**Goal:** Provide a minimal, safe optimization loop that evaluates configuration candidates and never silently degrades or overwrites a strategy.

**Architecture:** Evaluate caller-supplied config overrides in isolated temporary copies, compare validated scores against the baseline, persist a report, and require an explicit accept operation before updating configuration.

**Tech Stack:** Python 3.11, tempfile, PyYAML, Pydantic 2, FastAPI, pytest.

---

### Task 1: Specify the ratchet contract

**Files:**
- Create: `tests/unit/test_optimization_service.py`
- Create: `tests/integration/test_api_optimization.py`

- [ ] Test baseline and candidate isolation.
- [ ] Test rejection of equal/worse candidates and selection above a minimum improvement.
- [ ] Test report-only default behavior and explicit acceptance.
- [ ] Test that invalid candidates cannot corrupt the source config.
- [ ] Run focused tests and confirm RED.

### Task 2: Implement candidate evaluation

**Files:**
- Create: `src/autostrategy/services/optimization_service.py`
- Modify: `src/autostrategy/services/models.py`
- Modify: `src/autostrategy/services/__init__.py`

- [ ] Define typed candidate and report models.
- [ ] Deep-merge overrides only into copied workspaces.
- [ ] Evaluate baseline and candidates using the strict backtest workflow.
- [ ] Persist a full report while leaving the live strategy untouched.

### Task 3: Add explicit acceptance and API

**Files:**
- Create: `src/autostrategy/api/routers/optimization.py`
- Modify: `src/autostrategy/api/schemas.py`
- Modify: `src/autostrategy/api/dependencies.py`
- Modify: `src/autostrategy/api/app.py`

- [ ] Add evaluate, latest-report, and accept endpoints.
- [ ] On acceptance, verify report/digest freshness, update config, bump version, and set optimized status.
- [ ] Run focused tests and confirm GREEN.
- [ ] Do not invoke an LLM, alter strategy design/code automatically, or touch simulation behavior.
- [ ] Do not commit unless the user explicitly requests it.
