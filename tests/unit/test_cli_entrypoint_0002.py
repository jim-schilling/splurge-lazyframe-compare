import importlib
import sys
from types import ModuleType

import pytest


def _reload_main() -> ModuleType:
    if "splurge_lazyframe_compare.__main__" in sys.modules:
        del sys.modules["splurge_lazyframe_compare.__main__"]
    return importlib.import_module("splurge_lazyframe_compare.__main__")


def test_report_dry_run(capsys: pytest.CaptureFixture[str]) -> None:
    mod = _reload_main()
    main = mod.main
    code = main(["report", "--dry-run"])
    out = capsys.readouterr().out
    assert code == 0
    assert "Dry run: report" in out


def test_export_dry_run(capsys: pytest.CaptureFixture[str]) -> None:
    mod = _reload_main()
    main = mod.main
    code = main(["export", "--dry-run"])
    out = capsys.readouterr().out
    assert code == 0
    assert "Dry run: export" in out
