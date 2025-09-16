(The file `d:\repos\splurge-lazyframe-compare\CHANGELOG.md` exists, but is empty)
# Changelog

All notable changes to this project are documented in this file.

### 2025.2.0 (2025-09-03)
- Added domain exceptions at CLI boundary: `ConfigError`, `DataSourceError`.
- Standardized CLI exit codes: `2` for domain errors, `1` for unexpected errors.
- CLI now catches `ComparisonError` first, preserving clear user-facing messages.
- Services error handling preserves exception type and chains the original cause; messages now include service name and context.
- Documentation updates: README now documents CLI errors/exit codes and new exceptions.
- New CLI capabilities and flags: `compare`, `report`, `export`, `--dry-run`, `--format`, `--output-dir`, `--log-level`.
- Logging improvements: introduced `configure_logging()`; removed import-time handler side-effects; consistent log formatting.
- Packaging/entrypoints: ensured `slc` console script and `__main__.py` entry for `python -m splurge_lazyframe_compare`.
- Docs: added CLI usage, logging configuration, and large-data export guidance.

### 2025.3.0 (2025-09-16) - chore/update-tests & feature work
- Added per-Python-version CI workflows and badges (3.10–3.13) and updated the central `ci.yml` lint/typecheck workflow.
- Implemented an in-repo coverage badge generator workflow and accompanying script that creates `docs/coverage-badge.svg` and opens/updates an idempotent PR (`coverage-badge-update`).
- Added smoke checks and robust parsing for coverage generation; fallback badge shows `unknown` when parsing fails.
- Switched CI trigger policy: lint/typecheck and per-version tests now run on pushes/PRs for all branches; coverage badge generation restricted to `main`.
- Implemented idempotent badge updates (fixed branch) and PR-update behavior to avoid duplicate PRs.
- Added/updated GitHub Actions workflow files and helper script: `.github/workflows/*` and `.github/scripts/generate_coverage_badge.sh`.
- Added many unit & integration tests covering schema validation, file operations, formatting utilities, logging helpers, and type helpers (multiple `tests/unit/*` and `tests/integration/*` files).
- Enhanced documentation and coding standards guidance; added plans and rules under `.cursor/` and `.github/copilot-instructions.md`.
- Began implementing feature scaffolding and documentation for a Polars LazyFrame comparison framework (new/updated modules and examples).
- Misc: improved packaging metadata, pre-commit config, and general code cleanups across modules and examples.

### 2025.1.1 (2025-08-29)
- Removed extraneous folders and plan documents.

### 2025.1.0 (2025-08-29)
- Initial Commit

