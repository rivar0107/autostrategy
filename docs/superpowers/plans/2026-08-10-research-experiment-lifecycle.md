# Research Experiment Lifecycle Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [x]`) syntax for tracking.

**Goal:** Build a reproducible strategy research lifecycle from an immutable strategy version and frozen dataset through baseline, diagnosis, isolated optimization, one-time out-of-sample validation, and explicit accept/reject/rollback.

**Architecture:** Add immutable artifact snapshots and active-version pointers, capture strategy data into a locked dataset manifest, then persist a separate `ExperimentSession` state machine in SQLite. Every backtest run is linked to a session, version, manifest, phase, and candidate; optimization never edits the live workspace, and acceptance or rollback restores a selected immutable snapshot atomically. FT client `SimulationSession` remains a separate subsystem.

**Tech Stack:** Python 3.11+, Pydantic 2, sqlite3, pandas, PyYAML, FastAPI, React 19, TypeScript, pytest, Vitest.

---

### Task 1: Define immutable research contracts and persistence

**Files:**
- Create: `src/autostrategy/core/research.py`
- Create: `src/autostrategy/persistence/research_store.py`
- Modify: `src/autostrategy/core/strategy.py`
- Modify: `src/autostrategy/services/models.py`
- Test: `tests/unit/test_research_store.py`

- [x] Write failing tests for `StrategyVersion`, non-overlapping `DatasetSplit`, `DatasetManifest`, structured `DiagnosticFinding`, and valid/invalid `ExperimentStatus` transitions.
- [x] Write failing SQLite tests proving versions, manifests, and sessions survive a new store instance and strategy/version uniqueness is enforced.
- [x] Run `.venv/bin/pytest -q tests/unit/test_research_store.py` and confirm RED because the contracts/store do not exist.
- [x] Implement Pydantic contracts with these required identifiers: `version_id`, `manifest_id`, `session_id`, `strategy_slug`, immutable digests, timestamps, parent/base relationships, and decision fields.
- [x] Add `current_version_id` and `active_version_id` to `Strategy` with backward-compatible `None` defaults.
- [x] Create SQLite tables `strategy_versions`, `dataset_manifests`, and `experiment_sessions`; serialize nested models as JSON and use one connection per operation.
- [x] Run the focused test and confirm GREEN with no `ResourceWarning`.

### Task 2: Snapshot, activate, and roll back strategy versions

**Files:**
- Create: `src/autostrategy/services/version_service.py`
- Modify: `src/autostrategy/core/workspace.py`
- Modify: `src/autostrategy/agents/codegen_agent.py`
- Modify: `src/autostrategy/services/codegen_service.py`
- Test: `tests/unit/test_version_service.py`
- Test: `tests/integration/test_api_strategy_versions.py`

- [x] Write failing tests that snapshot every path in `VERSIONED_ARTIFACTS`, reject mutation of an existing snapshot, and lazily migrate a legacy strategy to an initial version.
- [x] Write failing tests that create a config-only child version without touching live files, activate it, and restore its parent on rollback.
- [x] Implement snapshots under `<strategy>/.autostrategy/versions/<version_id>/`, including `version.json` plus exact artifact copies and SHA-256 verification.
- [x] Implement `ensure_current_version`, `create_candidate_version`, `activate_version`, `reject_version`, `list_versions`, and `get_version`.
- [x] Before code generation overwrites artifacts, ensure the current snapshot exists; after successful generation create and activate a new accepted snapshot.
- [x] Run focused tests and confirm GREEN.

### Task 3: Freeze data and enforce train/validation/test boundaries

**Files:**
- Create: `src/autostrategy/services/dataset_manifest_service.py`
- Test: `tests/unit/test_dataset_manifest_service.py`

- [x] Write failing tests for chronological, non-overlapping train/validation/test ranges; reject missing benchmark, negative costs, and mutable manifests.
- [x] Write a local `data/fetch_data.py` fixture returning both DataFrame and dict-of-DataFrame forms, then test that one capture writes immutable CSV snapshots and a digest.
- [x] Test that materializing a manifest into a temporary version workspace replaces only the temporary fetch adapter and filters every frame to the requested split.
- [x] Implement capture under `<strategy>/.autostrategy/datasets/<manifest_id>/`, persisting canonical manifest JSON and CSV bytes before computing `data_digest`.
- [x] Implement a generated local adapter that never calls the upstream source during experiment backtests.
- [x] Implement split config normalization that sets both `start_date/end_date` and `period.start/period.end`.
- [x] Run focused tests and confirm GREEN.

### Task 4: Persist and execute the experiment state machine

**Files:**
- Create: `src/autostrategy/services/experiment_service.py`
- Modify: `src/autostrategy/persistence/run_store.py`
- Modify: `src/autostrategy/services/backtest_service.py`
- Test: `tests/unit/test_experiment_service.py`

- [x] Write failing tests for `created → baseline_completed → diagnosed → optimized → oos_validated → awaiting_decision → accepted/rejected` and reject skipped or repeated transitions.
- [x] Extend backtest runs with nullable `session_id`, `manifest_id`, `version_id`, `phase`, and `candidate_id`, including migration of existing SQLite databases.
- [x] Implement experiment creation from a base version and locked manifest.
- [x] Run baseline training and validation from immutable artifacts and frozen data; persist both run IDs.
- [x] Convert engine diagnostics and research-quality warnings into structured findings with severity, evidence, hypothesis, suggested actions, and `auto_fixable`.
- [x] Persist every transition before returning so a new service instance resumes the same session.
- [x] Run focused tests and confirm GREEN.

### Task 5: Optimize without test leakage and reveal OOS once

**Files:**
- Modify: `src/autostrategy/services/optimization_service.py`
- Modify: `src/autostrategy/services/experiment_service.py`
- Test: `tests/unit/test_experiment_optimization.py`

- [x] Write failing tests proving candidates see train/validation ranges but never test dates, each candidate changes one primary parameter, and no live artifact changes during optimization.
- [x] Add deterministic candidate generation for allowlisted numeric strategy parameters when callers provide no candidates; cap at five candidates and exclude dates, cash, data limits, commission, and slippage.
- [x] Evaluate every candidate on train then validation, enforcing finite metrics, minimum validation trades, maximum drawdown, score improvement, and complexity budget.
- [x] Persist selected config overrides and create a non-active candidate `StrategyVersion` snapshot.
- [x] Write failing tests that allow exactly one final OOS reveal, run the base and selected versions on the same test split, and reject repeated reveals.
- [x] Mark OOS passed only when hard gates pass and the candidate does not exceed the configured score-degradation allowance.
- [x] Run focused tests and confirm GREEN.

### Task 6: Accept, reject, and pointer-based rollback

**Files:**
- Modify: `src/autostrategy/services/experiment_service.py`
- Modify: `src/autostrategy/services/version_service.py`
- Test: `tests/unit/test_experiment_decisions.py`

- [x] Write failing tests that acceptance requires successful OOS, matching base digests, and explicit user intent.
- [x] Atomically activate the selected immutable snapshot, set both strategy pointers, mark the version accepted, and persist the decision.
- [x] Implement rejection without modifying live artifacts and prohibit decisions after a terminal decision.
- [x] Implement rollback to any accepted ancestor, restoring exact artifact bytes while retaining all newer versions and run history.
- [x] Inject a restore failure in a test and prove live files and pointers remain unchanged.
- [x] Run focused tests and confirm GREEN.

### Task 7: Expose APIs and the workbench research flow

**Files:**
- Create: `src/autostrategy/api/routers/research.py`
- Modify: `src/autostrategy/api/schemas.py`
- Modify: `src/autostrategy/api/dependencies.py`
- Modify: `src/autostrategy/api/app.py`
- Modify: `src/autostrategy/web/frontend/src/types.ts`
- Modify: `src/autostrategy/web/frontend/src/api/client.ts`
- Modify: `src/autostrategy/web/frontend/src/StrategyWorkbench.tsx`
- Modify: `src/autostrategy/web/frontend/src/App.test.tsx`
- Test: `tests/integration/test_api_research_flow.py`

- [x] Add create/read/list endpoints for versions, manifests, and experiments plus explicit baseline, diagnose, optimize, validate-OOS, accept, reject, and rollback actions.
- [x] Add API integration coverage for the complete happy path, illegal transitions, stale digests, repeated OOS reveal, and rollback.
- [x] Add a workbench research-flow panel showing current session state, immutable base/data identifiers, train/validation/test dates, findings, candidates, OOS result, and decision buttons.
- [x] Require confirmation only for accept, reject, and rollback; ordinary research steps remain reversible or read-only.
- [x] Run focused API/frontend tests and confirm GREEN.

### Task 8: Migration and final verification

**Files:**
- Modify: `README.md`
- Modify: `.github/workflows/ci.yml`
- Test: all tests

- [x] Verify a legacy `strategy.yaml` and existing `runs.sqlite3` migrate without losing current results.
- [x] Document experiment semantics, one-time OOS rule, hard acceptance gates, and the distinction from FT `SimulationSession`.
- [x] Run `.venv/bin/pytest -q -W error::ResourceWarning`.
- [x] Run `.venv/bin/ruff check src tests`.
- [x] Run `npm test -- --reporter=dot`, `npm run typecheck`, and `npm run build`.
- [x] Run `npm audit --audit-level=high` and `git diff --check`.
- [x] Restart the local backend, execute one fixture-backed experiment through the API, and inspect the real workbench and browser console.
- [x] Do not commit, branch, or modify FT client simulation contracts unless the user explicitly requests it.
