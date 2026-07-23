"""Shared Webull market-data error and auth result types."""

from __future__ import annotations

from dataclasses import dataclass


class WebullApiError(Exception):
    """Webull API failure with safe classification metadata."""

    def __init__(
        self,
        message: str,
        *,
        status_code: int | None = None,
        error_code: str | None = None,
        subscription_related: bool = False,
        rate_limited: bool = False,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.error_code = error_code
        self.subscription_related = subscription_related
        self.rate_limited = rate_limited


class OptionEndpointUnverified(Exception):
    """Endpoint not verified against Webull OpenAPI docs."""


@dataclass
class WebullAuthResult:
    success: bool
    token_present: bool = False
    message: str = ""
