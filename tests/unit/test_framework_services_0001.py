import pytest

from splurge_lazyframe_compare.services.base_service import BaseService


class _FailingService(BaseService):
    def __init__(self) -> None:
        super().__init__("FailingService")

    def _validate_inputs(self, **kwargs) -> None:  # noqa: ANN001
        pass

    def do_work(self) -> None:
        try:
            raise ValueError("original failure message")
        except Exception as e:  # noqa: BLE001
            self._handle_error(e, {"operation": "unit_test_op", "detail": 123})


def test_handle_error_preserves_type_and_context() -> None:
    service = _FailingService()

    with pytest.raises(ValueError) as exc_info:
        service.do_work()

    # Type is preserved
    assert isinstance(exc_info.value, ValueError)

    # Message contains service name, original message, and context snippet
    message = str(exc_info.value)
    assert "FailingService" in message
    assert "original failure message" in message
    assert "unit_test_op" in message

    # Cause is chained to preserve traceback context
    assert exc_info.value.__cause__ is not None
