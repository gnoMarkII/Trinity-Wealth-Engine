import logging
import pytest
from unittest.mock import patch, MagicMock

from core.logger import _DailyMarkdownHandler, setup_logging, get_logger

def test_daily_markdown_handler_emit():
    handler = _DailyMarkdownHandler()
    record = logging.LogRecord(
        name="test_logger",
        level=logging.WARNING,
        pathname="",
        lineno=0,
        msg="test message",
        args=(),
        exc_info=None,
    )
    
    with patch("core.logger._today_path") as mock_path_func:
        mock_path = MagicMock()
        mock_path_func.return_value = mock_path
        
        mock_file = MagicMock()
        mock_path.open.return_value.__enter__.return_value = mock_file
        
        with patch("core.logger._ensure_file") as mock_ensure:
            handler.emit(record)
            mock_ensure.assert_called_once_with(mock_path)
            
            # The file write should be called
            mock_file.write.assert_called_once()
            args, _ = mock_file.write.call_args
            assert "WARNING [test_logger]" in args[0]
            assert "test message" in args[0]

def test_daily_markdown_handler_emit_error():
    handler = _DailyMarkdownHandler()
    record = logging.LogRecord(
        name="test_logger",
        level=logging.WARNING,
        pathname="",
        lineno=0,
        msg="test message",
        args=(),
        exc_info=None,
    )
    
    with patch("core.logger._today_path", side_effect=Exception("File Error")):
        with patch.object(handler, "handleError") as mock_handle:
            handler.emit(record)
            mock_handle.assert_called_once_with(record)

def test_setup_logging(monkeypatch):
    monkeypatch.setenv("LOG_LEVEL", "DEBUG")
    
    with patch("core.logger.logging.basicConfig") as mock_basic:
        setup_logging()
        
        mock_basic.assert_called_once()
        assert mock_basic.call_args[1]["level"] == logging.DEBUG
        
        # Verify noisy loggers are silenced
        assert logging.getLogger("httpx").level == logging.WARNING
        assert logging.getLogger("yfinance").level == logging.CRITICAL
        
        # Check that file handler was added
        root_logger = logging.getLogger()
        has_daily_handler = any(isinstance(h, _DailyMarkdownHandler) for h in root_logger.handlers)
        assert has_daily_handler
        
        # clean up handlers for subsequent tests
        for h in root_logger.handlers[:]:
            if isinstance(h, _DailyMarkdownHandler):
                root_logger.removeHandler(h)

def test_setup_logging_fallback():
    with patch("core.logger.logging.basicConfig") as mock_basic:
        setup_logging("WARNING")
        mock_basic.assert_called_once()
        assert mock_basic.call_args[1]["level"] == logging.WARNING
        
        root_logger = logging.getLogger()
        for h in root_logger.handlers[:]:
            if isinstance(h, _DailyMarkdownHandler):
                root_logger.removeHandler(h)

def test_get_logger():
    logger = get_logger("my_logger")
    assert logger.name == "my_logger"
