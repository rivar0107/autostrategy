# Strategy Workbench Next Action Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make a designed strategy expose an obvious code-generation action and prove the real UI advances to coded with a backtest next step.

**Architecture:** Keep status-to-action mapping inside `StrategyWorkbench`, reuse the existing `api.codegen` request, and refresh strategy/artifact state after success. Add focused component tests for the visible action, pending state, successful transition, and localized failure before changing production behavior.

**Tech Stack:** React 19, TypeScript, Ant Design 6, Vitest, Testing Library, FastAPI, pytest, Ruff.

---

### Task 1: Lock the designed-to-coded UI contract

**Files:**
- Modify: `src/autostrategy/web/frontend/src/App.test.tsx`
- Test: `src/autostrategy/web/frontend/src/App.test.tsx`

- [ ] Add a focused test whose initial detail response has `status: 'designed'` and assert the page renders `下一步：生成策略代码` plus the primary button `生成策略代码`.
- [ ] Add a deferred codegen response and assert clicking the primary button sends `POST /api/v1/strategies/demo/codegen` with `{"force":false}` and makes the button busy/disabled while pending.
- [ ] Resolve codegen with a `coded` strategy, make the refreshed detail response return `coded`, and assert the next action becomes `下一步：运行回测`.
- [ ] Return an API error from codegen and assert a clear Chinese message beginning with `生成策略代码失败：` is visible.
- [ ] Run `npm test -- src/App.test.tsx --reporter=dot` and verify the new localization assertion fails before implementation.

### Task 2: Implement the minimal state transition and localized error

**Files:**
- Modify: `src/autostrategy/web/frontend/src/StrategyWorkbench.tsx`
- Test: `src/autostrategy/web/frontend/src/App.test.tsx`

- [ ] Keep the `designed` mapping as title `下一步：生成策略代码`, button `生成策略代码`, and `runCodegen(false)`.
- [ ] On a successful codegen response, immediately apply `result.strategy`, refresh artifacts/results, and preserve the loading state until refresh completes.
- [ ] On failure, show `生成策略代码失败：${readableError}`; map LLM configuration and validation errors to clear Chinese guidance while retaining useful backend details.
- [ ] Replace deprecated Ant Design props touched by the page: `iconPosition` → `iconPlacement`, `Alert message` → `Alert title`, and Modal mask closability with the Ant Design 6 API.
- [ ] Run the focused frontend test and confirm all assertions pass without Ant Design deprecation warnings.

### Task 3: Repair previously discovered Python quality failures

**Files:**
- Modify: `tests/unit/test_llm_client.py`
- Modify: `src/autostrategy/core/backtest_engine.py`
- Modify: `src/autostrategy/services/design_job_service.py`

- [ ] Update `test_resolve_api_key_missing` to remove every supported environment fallback and stub Codex defaults, preventing local `.env`/Codex state from leaking a secret into assertion output.
- [ ] Reformat the six Ruff E501 violations without changing behavior.
- [ ] Remove the two Ruff F401 unused imports.
- [ ] Run `.venv/bin/pytest -q` and `.venv/bin/ruff check src tests` and confirm both pass.

### Task 4: Build and perform real browser verification

**Files:**
- Generated: `src/autostrategy/web/static/index.html`
- Generated: `src/autostrategy/web/static/assets/*`

- [ ] Run `npm run typecheck`.
- [ ] Run `npm run build` and let Vite update only its generated static output.
- [ ] Run `npm test -- --reporter=dot`, `.venv/bin/pytest -q`, `.venv/bin/ruff check src tests`, and `git diff --check`.
- [ ] Open `http://127.0.0.1:3001/static/#/strategy/中证500指数增强策略` and visibly confirm the designed next-action card/button.
- [ ] Click the actual button, confirm the POST request and loading state, wait for success, then visibly confirm status `coded` and `下一步：运行回测`.
- [ ] Inspect browser console/API errors and report exact evidence; do not commit, branch, or revert unrelated changes.
