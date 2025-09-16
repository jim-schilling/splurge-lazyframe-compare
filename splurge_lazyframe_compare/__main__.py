"""Package entry point.

Copyright (c) 2025 Jim Schilling.
Licensed under the MIT License. See the LICENSE file for details.
"""

import sys

from splurge_lazyframe_compare.cli import main

DOMAINS: list[str] = ["cli", "entrypoint"]


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
