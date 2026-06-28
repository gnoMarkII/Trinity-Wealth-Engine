"""Tests for core/logger.py idempotency"""
import pytest
import logging
from core.logger import setup_logging
import core.logger

def test_setup_logging_idempotent(tmp_path, monkeypatch):
    class MockPath:
        def __init__(self, *args):
            pass
        def mkdir(self, *args, **kwargs):
            pass
        def resolve(self):
            return self
        @property
        def parents(self):
            return [self, self, self]
        def __truediv__(self, other):
            return tmp_path / other

    monkeypatch.setattr(core.logger, "Path", MockPath)
    
    # reset state
    core.logger._SETUP_DONE = False
    
    # Count root handlers before
    initial_count = len(logging.getLogger().handlers)
    
    setup_logging()
    count_after_first = len(logging.getLogger().handlers)
    assert count_after_first == initial_count + 1
    
    setup_logging()
    count_after_second = len(logging.getLogger().handlers)
    assert count_after_second == count_after_first  # Should not add another handler
