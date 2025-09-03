import polars as pl
import pytest

from splurge_lazyframe_compare.models.schema import ComparisonConfig
from splurge_lazyframe_compare.utils import create_comparison_config_from_lazyframes


def test_create_comparison_config_from_lazyframes_keyword_names() -> None:
    left_df = pl.LazyFrame({"id": [1, 2], "name": ["a", "b"]})
    right_df = pl.LazyFrame({"id": [1, 3], "name": ["a", "c"]})

    cfg = create_comparison_config_from_lazyframes(
        left=left_df,
        right=right_df,
        pk_columns=["id"],
    )

    assert isinstance(cfg, ComparisonConfig)
    assert cfg.pk_columns == ["id"]
    assert len(cfg.column_mappings) == 2


def test_create_comparison_config_from_lazyframes_missing_pk_raises() -> None:
    left_df = pl.LazyFrame({"id": [1, 2], "name": ["a", "b"]})
    right_df = pl.LazyFrame({"id": [1, 3], "name": ["a", "c"]})

    with pytest.raises(ValueError) as exc_info:
        create_comparison_config_from_lazyframes(
            left=left_df,
            right=right_df,
            pk_columns=["missing"]
        )

    msg = str(exc_info.value)
    assert "missing from left LazyFrame" in msg
    assert "missing" in msg
