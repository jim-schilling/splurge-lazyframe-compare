import importlib
import json
import sys
from types import ModuleType

import polars as pl
import pytest


def load_main_module() -> ModuleType:
    if "splurge_lazyframe_compare.__main__" in sys.modules:
        del sys.modules["splurge_lazyframe_compare.__main__"]
    return importlib.import_module("splurge_lazyframe_compare.__main__")


def _write_parquet(df: pl.DataFrame, path: str) -> None:
    df.write_parquet(path)


def _make_small_frames() -> tuple[pl.DataFrame, pl.DataFrame]:
    left = pl.DataFrame({"id": [1, 2], "value": [10, 20]})
    right = pl.DataFrame({"cust_id": [1, 2], "val": [10, 20]})
    return left, right


def _make_basic_config(tmp_path) -> str:
    cfg = {
        "primary_key_columns": ["id"],
        "column_mappings": [
            {"left": "id", "right": "cust_id", "name": "id"},
            {"left": "value", "right": "val", "name": "value"},
        ],
        "ignore_case": False,
        "null_equals_null": True,
        "tolerance": {},
    }
    cfg_path = tmp_path / "cfg.json"
    cfg_path.write_text(json.dumps(cfg))
    return str(cfg_path)


def test_cli_compare_requires_paths_or_config(capsys: pytest.CaptureFixture[str]) -> None:
    main_mod = load_main_module()
    main = main_mod.main

    code = main(["compare"])  # missing left/right
    out = capsys.readouterr().out
    assert code == 2
    assert "required" in out.lower()


def test_cli_compare_with_config_and_parquet(tmp_path, capsys: pytest.CaptureFixture[str]) -> None:
    left_df, right_df = _make_small_frames()
    left_path = tmp_path / "left.parquet"
    right_path = tmp_path / "right.parquet"
    _write_parquet(left_df, str(left_path))
    _write_parquet(right_df, str(right_path))

    cfg_path = _make_basic_config(tmp_path)

    main_mod = load_main_module()
    main = main_mod.main
    code = main(["compare", "--config", cfg_path, "--left", str(left_path), "--right", str(right_path)])
    out = capsys.readouterr().out

    assert code == 0
    assert "SPLURGE LAZYFRAME COMPARISON SUMMARY" in out


def test_cli_export_writes_files_and_prints_paths(tmp_path, capsys: pytest.CaptureFixture[str]) -> None:
    left_df, right_df = _make_small_frames()
    left_path = tmp_path / "left.parquet"
    right_path = tmp_path / "right.parquet"
    out_dir = tmp_path / "out"
    _write_parquet(left_df, str(left_path))
    _write_parquet(right_df, str(right_path))

    cfg_path = _make_basic_config(tmp_path)

    main_mod = load_main_module()
    main = main_mod.main
    code = main(
        [
            "export",
            "--config",
            cfg_path,
            "--left",
            str(left_path),
            "--right",
            str(right_path),
            "--format",
            "parquet",
            "--output-dir",
            str(out_dir),
        ]
    )
    out = capsys.readouterr().out
    assert code == 0

    exported = json.loads(out)
    assert "summary" in exported
    # Summary JSON should exist
    from pathlib import Path

    summary_path = Path(exported["summary"])
    assert summary_path.exists()
    assert summary_path.suffix == ".json"


def test_cli_invalid_config_fails_gracefully(tmp_path, capsys: pytest.CaptureFixture[str]) -> None:
    bad_cfg = tmp_path / "bad.json"
    bad_cfg.write_text("{not json}")

    main_mod = load_main_module()
    main = main_mod.main
    code = main(["compare", "--config", str(bad_cfg)])
    out = capsys.readouterr().out
    assert code == 2
    assert "configuration error" in out.lower()


def test_cli_unsupported_extension_returns_error(tmp_path, capsys: pytest.CaptureFixture[str]) -> None:
    data = tmp_path / "data.txt"
    data.write_text("dummy")
    cfg_path = _make_basic_config(tmp_path)

    main_mod = load_main_module()
    main = main_mod.main
    code = main(["compare", "--config", cfg_path, "--left", str(data), "--right", str(data)])
    out = capsys.readouterr().out
    assert code == 2
    assert "unsupported file extension" in out.lower()
