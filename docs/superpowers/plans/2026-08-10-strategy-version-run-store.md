# Strategy Version and Run Store Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:test-driven-development to implement each task and superpowers:verification-before-completion before claiming success.

**Goal:** Tie each generated strategy revision to immutable backtest history that survives service restarts.

**Architecture:** Extend strategy metadata with a monotonic version and content digest, update both after successful code generation, and store completed backtest snapshots in a workspace-level SQLite database using the standard library.

**Tech Stack:** Python 3.11, sqlite3, hashlib, Pydantic 2, FastAPI, pytest.

---

### Task 1: Specify version and digest behavior

**Files:**
- Modify: `tests/unit/test_workspace.py`
- Modify: `tests/unit/test_services_backtest.py`
- Modify: `tests/integration/test_api_backtest.py`

- [ ] Test backward-compatible defaults for old `strategy.yaml` files.
- [ ] Test deterministic digest computation and version bumping.
- [ ] Test one immutable run record per successful backtest.
- [ ] Test list/detail run-history API responses.
- [ ] Run focused tests and confirm RED.

### Task 2: Implement metadata and SQLite persistence

**Files:**
- Modify: `src/autostrategy/core/strategy.py`
- Modify: `src/autostrategy/core/workspace.py`
- Modify: `src/autostrategy/services/codegen_service.py`
- Create: `src/autostrategy/persistence/__init__.py`
- Create: `src/autostrategy/persistence/run_store.py`
- Modify: `src/autostrategy/services/backtest_service.py`

- [ ] Add version and content digest fields with safe defaults.
- [ ] Hash the versioned strategy artifacts and bump only after successful generation.
- [ ] Create and query the SQLite backtest run table.
- [ ] Persist the validated result snapshot after every successful backtest.

### Task 3: Expose read-only history

**Files:**
- Modify: `src/autostrategy/services/models.py`
- Modify: `src/autostrategy/api/schemas.py`
- Modify: `src/autostrategy/api/routers/backtest.py`

- [ ] Add run summary/detail models and list/detail endpoints.
- [ ] Keep latest-result compatibility intact.
- [ ] Run focused tests and confirm GREEN.
- [ ] Do not add SimulationSession or change paper-run state.
- [ ] Do not commit unless the user explicitly requests it.
