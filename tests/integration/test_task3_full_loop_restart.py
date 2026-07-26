"""Crash/restart coverage for the Task 3 automatic evolution loop."""

from __future__ import annotations

# Re-export the restart proof from the automatic-loop module so the required
# path exists and CI discovers it under the mandated filename.
from tests.integration.test_task3_full_automatic_evolution_loop import (  # noqa: F401
    test_task3_full_loop_restart,
)
