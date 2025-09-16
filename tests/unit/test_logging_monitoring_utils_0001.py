import importlib
import logging
import os
import sys
from types import ModuleType

import pytest


def reload_logging_helpers() -> ModuleType:
    if "splurge_lazyframe_compare.utils.logging_helpers" in sys.modules:
        del sys.modules["splurge_lazyframe_compare.utils.logging_helpers"]
    return importlib.import_module("splurge_lazyframe_compare.utils.logging_helpers")


def test_import_has_no_module_level_handlers() -> None:
    # Capture handler counts before import
    pre_root_handlers = list(logging.getLogger().handlers)

    mod = reload_logging_helpers()
    logger = logging.getLogger(mod.__name__)

    # No new handlers should be attached by merely importing the module
    assert list(logging.getLogger().handlers) == pre_root_handlers
    assert len(logger.handlers) == 0


def test_configure_logging_sets_level_and_format(capsys: pytest.CaptureFixture[str]) -> None:
    mod = reload_logging_helpers()

    configure_logging = getattr(mod, "configure_logging")
    get_logger = getattr(mod, "get_logger")

    # Configure logging to INFO with a simple format
    fmt = "[%(levelname)s] %(name)s: %(message)s"
    configure_logging(level=logging.INFO, fmt=fmt)

    test_logger = get_logger("TestService")
    test_logger.info("hello")

    # Ensure format and level applied
    out = capsys.readouterr().err or capsys.readouterr().out
    assert "[INFO]" in out
    assert "splurge_lazyframe_compare.TestService" in out
    assert "hello" in out


def test_configure_logging_honors_env_vars(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    mod = reload_logging_helpers()
    configure_logging = getattr(mod, "configure_logging")
    get_logger = getattr(mod, "get_logger")

    monkeypatch.setenv("SLC_LOG_LEVEL", "DEBUG")
    monkeypatch.setenv("SLC_LOG_FORMAT", "[%(levelname)s] %(message)s")

    # Simulate env-aware bootstrap (we will call configure_logging using env)
    level = os.getenv("SLC_LOG_LEVEL", "INFO")
    fmt = os.getenv("SLC_LOG_FORMAT", None)
    configure_logging(level=level, fmt=fmt)

    logger = get_logger("EnvTest")
    logger.debug("dbg")

    out = capsys.readouterr().err or capsys.readouterr().out
    assert "[DEBUG]" in out
    assert "dbg" in out
