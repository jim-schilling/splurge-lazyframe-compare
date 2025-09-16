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

### 2025.1.1 (2025-08-29)
- Removed extraneous folders and plan documents.

### 2025.1.0 (2025-08-29)
- Initial Commit

