# Release Hardening Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:test-driven-development for behavior changes and superpowers:verification-before-completion for the release checks.

**Goal:** Make repository claims, installation dependencies, and automated checks consistent with the actual product.

**Architecture:** Keep the current package layout, repair metadata and documentation drift, pin frontend dependency ranges, and add a CI workflow that executes the same Python and frontend gates used locally.

**Tech Stack:** Hatchling, GitHub Actions, pytest, Ruff, npm, Vitest, TypeScript, Vite.

---

### Task 1: Add release contract checks

**Files:**
- Create: `tests/integration/test_package_install.py`
- Modify: `tests/integration/test_dashboard_static.py`

- [ ] Test that base installation can import the CLI and backtest engine dependencies.
- [ ] Test that all public version sources agree.
- [ ] Run focused tests and confirm any current failure.

### Task 2: Align packaging and documentation

**Files:**
- Create: `LICENSE`
- Modify: `pyproject.toml`
- Modify: `README.md`
- Modify: `package.json`

- [ ] Add the declared MIT license text.
- [ ] Ensure NumPy/Pandas are available wherever mandatory imports require them.
- [ ] Remove the inaccurate local-CSV precedence statement and describe current market support honestly.
- [ ] Replace unbounded `latest` frontend dependencies with reproducible compatible ranges.

### Task 3: Add CI and verify a release build

**Files:**
- Create: `.github/workflows/ci.yml`

- [ ] Run pytest and Ruff for supported Python versions.
- [ ] Run frontend tests, typecheck, and build on the declared Node version.
- [ ] Run a local wheel/install smoke check where feasible.
- [ ] Run all repository verification commands and `git diff --check`.
- [ ] Do not change any FT client simulation API or commit unless explicitly requested.
