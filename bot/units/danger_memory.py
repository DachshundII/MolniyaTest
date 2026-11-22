import itertools
from typing import Iterable, List, Tuple

import numpy as np

from cython_extensions.geometry import cy_distance_to_squared

from sc2.ids.unit_typeid import UnitTypeId as U
from sc2.position import Point2

from bot.tactics.consts import ENEMY_WORKER_TYPES

BUNKER_RANGE = 7.0
DEFENDED_MARGIN = 3.0
REMEMBER_SECONDS = 45.0
AVOID_EXTRA_COST = 500.0
REMEMBER_RADIUS_SQ = 1.0


def ground_defenders(ai) -> List[Tuple[Point2, float]]:
    defenders: List[Tuple[Point2, float]] = []

    for e in itertools.chain(ai.enemy_units, ai.enemy_structures):
        if e.type_id in ENEMY_WORKER_TYPES or e.is_hallucination:
            continue

        reach = BUNKER_RANGE if e.type_id == U.BUNKER else (e.ground_range if e.can_attack_ground else 0.0)
        if reach > 0:
            defenders.append((e.position, reach + e.radius + DEFENDED_MARGIN))

    return defenders


class DangerMemory:
    def __init__(self, ai):
        self.ai = ai
        self._spots: List[Tuple[Point2, float, float]] = []

    def refresh(self, danger_now: Iterable[Tuple[Point2, float]]) -> None:
        now = self.ai.time
        danger_now = list(danger_now)
        still_there = [position for position, _ in danger_now]

        kept = [
            (position, reach, until) for position, reach, until in self._spots
            if not any(cy_distance_to_squared(position, p) <= REMEMBER_RADIUS_SQ for p in still_there)
            and now < until and not self.ai.is_visible(position)
        ]

        kept.extend((position, reach, now + REMEMBER_SECONDS) for position, reach in danger_now)
        self._spots = kept

    def spots(self) -> List[Tuple[Point2, float]]:
        return [(position, reach) for position, reach, _ in self._spots]

    def avoiding_grid(self, grid: np.ndarray) -> np.ndarray:
        if not self._spots:
            return grid

        grid = grid.copy()

        for position, reach, _ in self._spots:
            x0, x1 = max(0, int(position.x - reach)), min(grid.shape[0], int(position.x + reach) + 1)
            y0, y1 = max(0, int(position.y - reach)), min(grid.shape[1], int(position.y + reach) + 1)

            if x1 > x0 and y1 > y0:
                grid[x0:x1, y0:y1] += AVOID_EXTRA_COST

        return grid