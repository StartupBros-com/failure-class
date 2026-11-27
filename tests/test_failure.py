from __future__ import annotations

from urllib.error import URLError

import pytest

from failure_class import FailureClass, classify_failure


def test_connection_timeout_auth_and_protocol_classes():
    assert classify_failure(URLError("connection refused")) == FailureClass.CONNECTION
    assert classify_failure(TimeoutError("timed out")) == FailureClass.TIMEOUT_SLA
    assert classify_failure(status_code=401, message="unauthorized") == FailureClass.AUTH
    assert classify_failure(message="missing choices", malformed_response=True) == FailureClass.ENDPOINT_PROTOCOL


def test_context_and_mlx_lazy_load_failures_are_provider_classes():
    assert classify_failure(status_code=400, message="prompt too long: exceeds max context window") == FailureClass.CONTEXT_OR_MODEL_FIT
    assert classify_failure(message="mlx lazy-load failed during eager-load probe") == FailureClass.RUNTIME_PROVIDER_WEDGED


def test_scorer_failure_after_success_is_model_quality():
    assert classify_failure(successful_model_response=True, scorer_failed=True) == FailureClass.MODEL_QUALITY


@pytest.mark.parametrize("status", [None, 400, 413, 422, 503])
def test_context_markers_classify_independently_of_status(status):
    assert (
        classify_failure(status_code=status, message="prompt too long")
        == FailureClass.CONTEXT_OR_MODEL_FIT
    )


@pytest.mark.parametrize(
    ("status", "message", "expected"),
    [
        (413, "request rejected", FailureClass.NONE),
        (401, "prompt too long", FailureClass.AUTH),
        (429, "prompt too long", FailureClass.TIMEOUT_SLA),
        (422, "invalid json: prompt too long", FailureClass.ENDPOINT_PROTOCOL),
    ],
)
def test_context_check_keeps_status_and_protocol_precedence(status, message, expected):
    assert classify_failure(status_code=status, message=message) == expected
