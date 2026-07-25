"""Typed domain exceptions for the market data layer."""

from __future__ import annotations


class MarketDataError(Exception):
    """Base error for market observation, bar, and surface failures."""


class FeatureTimeframeError(MarketDataError):
    """Raised when a feature/bar API receives an unexpected timeframe."""


# Alias retained for callers that use the Task 1 name FeatureFrameError.
FeatureFrameError = FeatureTimeframeError


class SnapshotError(MarketDataError):
    """Raised when snapshot persistence or retrieval fails."""


class OptionSurfaceError(MarketDataError):
    """Raised when option-surface construction or persistence fails."""
