#!/usr/bin/env python3
"""Generate coverage-percent.txt from coverage.xml.

This script is intentionally small and robust so it can be invoked from
GitHub Actions without relying on YAML heredocs. It parses the coverage.xml
produced by coverage/pytest-cov and writes a single-line file `coverage-percent.txt`.

Behavior:
 - If a parsable line-rate is found, writes a float percent with one decimal (e.g. 87.6)
 - On any parse error or missing file, writes 'unknown'

It exits with code 0 in all cases to avoid failing the job; the workflow
can perform additional smoke checks later.
"""
from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    cov_file = repo_root / "coverage.xml"
    out_file = repo_root / "coverage-percent.txt"

    if not cov_file.exists():
        out_file.write_text("unknown")
        print("coverage.xml not found; wrote 'unknown' to coverage-percent.txt")
        return 0

    try:
        tree = ET.parse(cov_file)
        root = tree.getroot()

        # coverage.py typically writes line-rate on the root element
        line_rate = root.attrib.get("line-rate")
        if line_rate is None:
            # try to find a child element with line-rate attribute
            for elem in root.iter():
                if "line-rate" in elem.attrib:
                    line_rate = elem.attrib.get("line-rate")
                    break

        if line_rate is None:
            out_file.write_text("unknown")
            print("line-rate attribute not found; wrote 'unknown'")
            return 0

        try:
            percent = float(line_rate) * 100.0
        except Exception:
            out_file.write_text("unknown")
            print("line-rate value not parseable as float; wrote 'unknown'")
            return 0

        out_file.write_text(f"{percent:.1f}")
        print(f"Wrote coverage percent: {percent:.1f}")
        return 0

    except ET.ParseError:
        out_file.write_text("unknown")
        print("Failed to parse coverage.xml; wrote 'unknown'")
        return 0
    except Exception as exc:  # pragma: no cover - robust logging
        out_file.write_text("unknown")
        print(f"Unexpected error while parsing coverage.xml: {exc}; wrote 'unknown'")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
