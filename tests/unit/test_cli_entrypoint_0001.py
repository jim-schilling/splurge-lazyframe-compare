import importlib
import sys
from types import ModuleType

import pytest


def load_main_module() -> ModuleType:
    # Reload to ensure test isolation across runs
    if "splurge_lazyframe_compare.__main__" in sys.modules:
        del sys.modules["splurge_lazyframe_compare.__main__"]
    return importlib.import_module("splurge_lazyframe_compare.__main__")


def test_cli_help_displays_usage_and_exits() -> None:
    main_mod = load_main_module()
    main = getattr(main_mod, "main")

    with pytest.raises(SystemExit) as exc:
        main(["--help"])  # Argparse help should exit

    assert exc.value.code == 0


def test_cli_compare_dry_run_succeeds_and_outputs_message(capsys: pytest.CaptureFixture[str]) -> None:
    main_mod = load_main_module()
    main = getattr(main_mod, "main")

    exit_code = main(["compare", "--dry-run"])  # Should not require dataframes
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "Dry run" in captured.out


def test_cli_namespace_and_program_name_in_help(capsys: pytest.CaptureFixture[str]) -> None:
    main_mod = load_main_module()
    main = getattr(main_mod, "main")

    with pytest.raises(SystemExit):
        main(["-h"])  # triggers help

    captured = capsys.readouterr()
    assert "Splurge LazyFrame Compare CLI" in captured.out


def test_cli_report_and_export_help() -> None:
    main_mod = load_main_module()
    main = getattr(main_mod, "main")

    with pytest.raises(SystemExit):
        main(["report", "-h"])  # subcommand help exits

    with pytest.raises(SystemExit):
        main(["export", "-h"])  # subcommand help exits
