# splurge-lazyframe-compare

[![PyPI - Version](https://img.shields.io/pypi/v/splurge-lazyframe-compare.svg)](https://pypi.org/project/splurge-lazyframe-compare)
[![License](https://img.shields.io/pypi/l/splurge-lazyframe-compare.svg)](LICENSE)
[![CI (general)](https://img.shields.io/github/actions/workflow/status/jim-schilling/splurge-lazyframe-compare/ci.yml?branch=main&label=CI)](https://github.com/jim-schilling/splurge-lazyframe-compare/actions)

[![Py 3.10](https://img.shields.io/github/actions/workflow/status/jim-schilling/splurge-lazyframe-compare/test-py310.yml?branch=main&label=py3.10)](https://github.com/jim-schilling/splurge-lazyframe-compare/actions/workflows/test-py310.yml)
[![Py 3.11](https://img.shields.io/github/actions/workflow/status/jim-schilling/splurge-lazyframe-compare/test-py311.yml?branch=main&label=py3.11)](https://github.com/jim-schilling/splurge-lazyframe-compare/actions/workflows/test-py311.yml)
[![Py 3.12](https://img.shields.io/github/actions/workflow/status/jim-schilling/splurge-lazyframe-compare/test-py312.yml?branch=main&label=py3.12)](https://github.com/jim-schilling/splurge-lazyframe-compare/actions/workflows/test-py312.yml)
[![Py 3.13](https://img.shields.io/github/actions/workflow/status/jim-schilling/splurge-lazyframe-compare/test-py313.yml?branch=main&label=py3.13)](https://github.com/jim-schilling/splurge-lazyframe-compare/actions/workflows/test-py313.yml)

[![Coverage](docs/coverage-badge.svg)](docs/coverage-badge.svg)
[![ruff](https://img.shields.io/badge/ruff-passed-brightgreen)](https://github.com/charliermarsh/ruff)
[![mypy](https://img.shields.io/badge/mypy-passed-brightgreen)](https://mypy-lang.org/)

A compact, high-performance toolkit for comparing Polars LazyFrames and generating structured difference reports.

See the detailed documentation for usage, configuration, API reference, examples, and advanced topics:

- Detailed docs: docs/README-DETAILS.md
- Changelog: CHANGELOG.md

## Quick install

```bash
pip install splurge-lazyframe-compare
```

## Minimal example

```python
from splurge_lazyframe_compare import LazyFrameComparator, ComparisonConfig
import polars as pl

left = pl.LazyFrame({"id": [1,2], "value": [10, 20]})
right = pl.LazyFrame({"id": [1,2], "value": [10, 21]})

config = ComparisonConfig(pk_columns=["id"])  # basic config for identical columns
comp = LazyFrameComparator(config)
results = comp.compare(left, right)
print(results.summary)
```

For full details, examples, and troubleshooting, see docs/README-DETAILS.md.

Contributions welcome — see CONTRIBUTING.md for guidelines.

