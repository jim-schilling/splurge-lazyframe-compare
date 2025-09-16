from pathlib import Path

import polars as pl
import pytest

from splurge_lazyframe_compare.utils.file_operations import (
    FileOperationConstants,
    ensure_directory_exists,
    export_lazyframe,
    get_export_file_paths,
    get_file_extension,
    import_lazyframe,
    list_files_by_pattern,
    validate_file_path,
)


def test_get_file_extension_supported_and_errors():
    assert get_file_extension("parquet") == FileOperationConstants.PARQUET_EXT
    assert get_file_extension("csv") == FileOperationConstants.CSV_EXT
    assert get_file_extension("json") == FileOperationConstants.JSON_EXT
    with pytest.raises(ValueError):
        get_file_extension("xml")


def test_export_and_import_parquet_csv_json(tmp_path: Path):
    df = pl.LazyFrame({"id": [1, 2], "name": ["a", "b"]})

    for fmt in ("parquet", "csv", "json"):
        out_file = tmp_path / f"data.{fmt if fmt != 'json' else 'ndjson'}"
        # export
        export_lazyframe(df, out_file, fmt)
        assert out_file.exists() and out_file.stat().st_size > 0

        # import (auto-detect by extension)
        lf = import_lazyframe(out_file)
        assert isinstance(lf, pl.LazyFrame)
        collected = lf.collect()
        assert collected.shape[0] == 2


def test_import_errors_and_validation(tmp_path: Path):
    # non-existing
    with pytest.raises(FileNotFoundError):
        import_lazyframe(tmp_path / "missing.parquet")

    # directory path
    ensure_directory_exists(tmp_path)
    with pytest.raises(ValueError):
        import_lazyframe(tmp_path)

    # empty file
    empty_file = tmp_path / "empty.csv"
    empty_file.write_text("")
    with pytest.raises(ValueError):
        import_lazyframe(empty_file)


def test_validate_file_path_and_list_patterns(tmp_path: Path):
    # valid path ensures parent exists
    p = tmp_path / "sub" / "file.parquet"
    validate_file_path(p)
    assert (tmp_path / "sub").exists()

    # list patterns
    (tmp_path / "one.txt").write_text("1")
    (tmp_path / "two.txt").write_text("2")
    matches = list_files_by_pattern(tmp_path, "*.txt")
    assert len(matches) == 2


def test_get_export_file_paths_map(tmp_path: Path):
    paths = get_export_file_paths("base", tmp_path, formats=["parquet", "csv", "json"])
    assert set(paths.keys()) == {"parquet", "csv", "json"}
    assert paths["parquet"].name.endswith(".parquet")
    assert paths["csv"].name.endswith(".csv")
    assert paths["json"].name.endswith(".json")
