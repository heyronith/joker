"""Replay clock with multiple speed modes."""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from joker.schemas.replay import ReplayClock, ReplaySpeedMode


@dataclass
class ReplayClockController:
    """Controls pacing between replay events."""

    clock: ReplayClock
    speed_multiplier: float = 1.0
    _step_index: int = 0

    def wait_until_next(self, next_timestamp) -> None:
        if self.clock.mode is ReplaySpeedMode.DETERMINISTIC:
            return
        if self.clock.mode is ReplaySpeedMode.STEP:
            self._step_index += 1
            return
        if self.clock.mode is ReplaySpeedMode.ACCELERATED:
            delta = (next_timestamp - self.clock.current_time).total_seconds()
            if delta > 0 and self.speed_multiplier > 0:
                time.sleep(delta / self.speed_multiplier)
            return
        if self.clock.mode is ReplaySpeedMode.REALTIME:
            delta = (next_timestamp - self.clock.current_time).total_seconds()
            if delta > 0:
                time.sleep(delta)

    @classmethod
    def deterministic(cls, clock: ReplayClock) -> ReplayClockController:
        clock.mode = ReplaySpeedMode.DETERMINISTIC
        return cls(clock=clock, speed_multiplier=0)

    @classmethod
    def accelerated(cls, clock: ReplayClock, speed: float) -> ReplayClockController:
        clock.mode = ReplaySpeedMode.ACCELERATED
        return cls(clock=clock, speed_multiplier=speed)
