import math

from cython_extensions.geometry import cy_distance_to

from sc2.position import Point2


class ProgressWatch:
    def __init__(
        self,
        still_seconds: float = 12.0,
        no_gain_seconds: float = 45.0,
        moved: float = 4.0,
        gain: float = 5.0,
        arrival: float = 15.0,
    ):
        self.still_seconds = still_seconds
        self.no_gain_seconds = no_gain_seconds
        self.moved = moved
        self.gain = gain
        self.arrival = arrival
        self._target: Point2 | None = None
        self._ref_position: Point2 | None = None
        self._ref_time = 0.0
        self._best = math.inf
        self._best_time = 0.0

    def reset(self) -> None:
        self._target = None

    def _start(self, now: float, position: Point2, target: Point2) -> None:
        self._target = target
        self._ref_position, self._ref_time = position, now
        self._best, self._best_time = cy_distance_to(position, target), now

    def stuck(self, now: float, position: Point2, target: Point2, fighting: bool) -> bool:
        if self._target is None or cy_distance_to(self._target, target) > 5.0:
            self._start(now, position, target)
            return False

        distance = cy_distance_to(position, target)

        if fighting or distance <= self.arrival:
            self._start(now, position, target)
            return False

        if cy_distance_to(position, self._ref_position) > self.moved:
            self._ref_position, self._ref_time = position, now

        if distance < self._best - self.gain:
            self._best, self._best_time = distance, now

        return now - self._ref_time >= self.still_seconds or now - self._best_time >= self.no_gain_seconds