# Backtest Contract and Research Quality Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:test-driven-development to implement each task and superpowers:verification-before-completion before claiming success.

**Goal:** Make every successful backtest expose validated, finite core metrics and honest research-quality warnings without fabricating missing evidence.

**Architecture:** Add Pydantic result models at the engine boundary, normalize legacy strategy dictionaries into that contract, and attach a separate research-quality section describing sample size and missing equity, benchmark, and out-of-sample evidence.

**Tech Stack:** Python 3.11, Pydantic 2, pytest, Ruff.

---

### Task 1: Lock the result contract with failing tests

**Files:**
- Create: `tests/unit/test_backtest_result.py`
- Modify: `tests/unit/test_backtest_engine.py`

- [ ] Test rejection of NaN/Infinity, invalid percentage ranges, negative drawdown, and zero trades.
- [ ] Test normalization of a valid legacy result.
- [ ] Test limited-sample and missing-evidence warnings.
- [ ] Run the focused tests and confirm RED before production changes.

### Task 2: Implement and integrate the contract

**Files:**
- Create: `src/autostrategy/core/backtest_result.py`
- Modify: `src/autostrategy/core/backtest_engine.py`
- Modify: `src/autostrategy/api/schemas.py`
- Modify: `src/autostrategy/services/models.py`

- [ ] Implement strict metric, equity-point, trade-record, research-quality, and workflow-result models.
- [ ] Validate strategy output before scoring or persistence and return an actionable error when invalid.
- [ ] Preserve extra legacy metrics while guaranteeing the required fields.
- [ ] Expose the typed workflow result through service/API schemas.
- [ ] Run focused tests and confirm GREEN.

### Task 3: Regression verification

- [ ] Run all backtest unit and integration tests.
- [ ] Run Ruff on touched Python files.
- [ ] Do not change paper-run behavior or any FT client simulation contract.
- [ ] Do not commit unless the user explicitly requests it.
