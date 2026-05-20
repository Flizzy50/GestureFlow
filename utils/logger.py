"""Logging setup. Use this instead of print().

Structured logs become valuable in Phase 4+ when we're debugging missed
gestures or false positives in real time — we need timestamps and levels,
not stdout spew.
"""
from __future__ import annotations

import logging
import sys


_CONFIGURED = False


def configure_logging(level: str = "INFO") -> None:
    """Configure root logger. Idempotent — safe to call multiple times."""
    global _CONFIGURED
    if _CONFIGURED:
        logging.getLogger().setLevel(level.upper())
        return

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        logging.Formatter(
            fmt="%(asctime)s.%(msecs)03d [%(levelname)s] %(name)s: %(message)s",
            datefmt="%H:%M:%S",
        )
    )
    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(level.upper())
    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
