"""Tests for core/retry.py — §3.4 API Resilience invariant"""
import pytest
from unittest.mock import MagicMock

from core.retry import is_transient_error, with_retry, _MAX_RETRIES


class TestIsTransientError:
    def test_builtin_timeout(self):
        assert is_transient_error(TimeoutError("timed out"))

    def test_builtin_connection_error(self):
        assert is_transient_error(ConnectionError("refused"))

    def test_generic_exception_non_transient(self):
        assert not is_transient_error(ValueError("bad input"))

    def test_zero_division_non_transient(self):
        assert not is_transient_error(ZeroDivisionError("div by zero"))

    def test_regex_429_in_message(self):
        assert is_transient_error(Exception("status 429"))

    def test_regex_500_in_message(self):
        assert is_transient_error(Exception("HTTP 500 Internal Server Error"))

    def test_regex_503_in_message(self):
        assert is_transient_error(Exception("503 Service Unavailable"))

    def test_regex_rate_limit_in_message(self):
        assert is_transient_error(Exception("rate limit exceeded"))

    def test_regex_too_many_requests(self):
        assert is_transient_error(Exception("too many requests"))

    def test_regex_timeout_in_message(self):
        assert is_transient_error(Exception("Connection timeout occurred"))

    def test_regex_temporarily_unavailable(self):
        assert is_transient_error(Exception("temporarily unavailable"))

    def test_regex_service_unavailable(self):
        assert is_transient_error(Exception("service unavailable"))

    def test_non_transient_404_in_message(self):
        assert not is_transient_error(Exception("404 Not Found"))

    def test_non_transient_401_in_message(self):
        assert not is_transient_error(Exception("401 Unauthorized"))

    def test_requests_http_error_with_429(self):
        """status_code path only applies to HTTPError subclasses, not plain Exception"""
        import requests
        mock_resp = MagicMock()
        mock_resp.status_code = 429
        exc = requests.exceptions.HTTPError(response=mock_resp)
        assert is_transient_error(exc)

    def test_requests_http_error_with_500(self):
        import requests
        mock_resp = MagicMock()
        mock_resp.status_code = 500
        exc = requests.exceptions.HTTPError(response=mock_resp)
        assert is_transient_error(exc)

    def test_requests_http_error_with_502(self):
        import requests
        mock_resp = MagicMock()
        mock_resp.status_code = 502
        exc = requests.exceptions.HTTPError(response=mock_resp)
        assert is_transient_error(exc)

    def test_requests_http_error_with_504(self):
        import requests
        mock_resp = MagicMock()
        mock_resp.status_code = 504
        exc = requests.exceptions.HTTPError(response=mock_resp)
        assert is_transient_error(exc)

    def test_requests_http_error_with_404_non_transient(self):
        import requests
        mock_resp = MagicMock()
        mock_resp.status_code = 404
        exc = requests.exceptions.HTTPError("Not Found", response=mock_resp)
        assert not is_transient_error(exc)

    def test_httpx_timeout_is_transient(self):
        import httpx
        exc = httpx.TimeoutException("timed out")
        assert is_transient_error(exc)

    def test_httpx_connect_error_is_transient(self):
        import httpx
        exc = httpx.ConnectError("connection refused")
        assert is_transient_error(exc)

    def test_httpx_http_status_error_with_429(self):
        import httpx
        mock_req = MagicMock()
        mock_resp = MagicMock()
        mock_resp.status_code = 429
        exc = httpx.HTTPStatusError("rate limited", request=mock_req, response=mock_resp)
        assert is_transient_error(exc)

    def test_httpx_http_status_error_with_503(self):
        import httpx
        mock_req = MagicMock()
        mock_resp = MagicMock()
        mock_resp.status_code = 503
        exc = httpx.HTTPStatusError("service unavailable", request=mock_req, response=mock_resp)
        assert is_transient_error(exc)

    def test_httpx_http_status_error_with_404_non_transient(self):
        import httpx
        mock_req = MagicMock()
        mock_resp = MagicMock()
        mock_resp.status_code = 404
        exc = httpx.HTTPStatusError("not found", request=mock_req, response=mock_resp)
        assert not is_transient_error(exc)


class TestWithRetry:
    def test_success_first_try(self, monkeypatch):
        slept = []
        monkeypatch.setattr("core.retry.time.sleep", lambda s: slept.append(s))
        fn = MagicMock(return_value="ok")
        assert with_retry(fn, 1, key="v") == "ok"
        fn.assert_called_once_with(1, key="v")
        assert slept == []

    def test_success_on_second_try(self, monkeypatch):
        slept = []
        monkeypatch.setattr("core.retry.time.sleep", lambda s: slept.append(s))
        calls = [0]

        def fn():
            calls[0] += 1
            if calls[0] == 1:
                raise TimeoutError("transient")
            return "done"

        assert with_retry(fn) == "done"
        assert calls[0] == 2
        assert slept == [1]  # 2^0

    def test_non_transient_raises_immediately(self, monkeypatch):
        slept = []
        monkeypatch.setattr("core.retry.time.sleep", lambda s: slept.append(s))
        fn = MagicMock(side_effect=ValueError("bad"))
        with pytest.raises(ValueError, match="bad"):
            with_retry(fn)
        fn.assert_called_once()
        assert slept == []

    def test_exhausts_retries_and_raises(self, monkeypatch):
        slept = []
        monkeypatch.setattr("core.retry.time.sleep", lambda s: slept.append(s))
        fn = MagicMock(side_effect=ConnectionError("network down"))
        with pytest.raises(ConnectionError):
            with_retry(fn)
        assert fn.call_count == _MAX_RETRIES
        assert slept == [1, 2]  # 2^0, 2^1 — no sleep on last attempt

    def test_exponential_backoff_sequence(self, monkeypatch):
        slept = []
        monkeypatch.setattr("core.retry.time.sleep", lambda s: slept.append(s))
        fn = MagicMock(side_effect=TimeoutError("slow"))
        with pytest.raises(TimeoutError):
            with_retry(fn)
        assert slept == [1, 2]

    def test_args_and_kwargs_forwarded(self, monkeypatch):
        monkeypatch.setattr("core.retry.time.sleep", lambda s: None)
        fn = MagicMock(return_value=42)
        result = with_retry(fn, "a", "b", x=3)
        fn.assert_called_once_with("a", "b", x=3)
        assert result == 42

    def test_transient_then_non_transient_raises_non_transient(self, monkeypatch):
        slept = []
        monkeypatch.setattr("core.retry.time.sleep", lambda s: slept.append(s))
        calls = [0]

        def fn():
            calls[0] += 1
            if calls[0] == 1:
                raise TimeoutError("transient first")
            raise ValueError("non-transient second")

        with pytest.raises(ValueError, match="non-transient second"):
            with_retry(fn)
        assert slept == [1]

    def test_returns_value_not_none(self, monkeypatch):
        monkeypatch.setattr("core.retry.time.sleep", lambda s: None)
        fn = MagicMock(return_value={"data": [1, 2, 3]})
        result = with_retry(fn)
        assert result == {"data": [1, 2, 3]}
