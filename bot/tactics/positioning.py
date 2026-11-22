from typing import List, Optional, Tuple

import numpy as np

from cython_extensions.units_utils import cy_closer_than, cy_closest_to, cy_sorted_by_distance_to
from cython_extensions.geometry import cy_distance_to, cy_distance_to_squared, cy_towards

from sc2.ids.unit_typeid import UnitTypeId as U
from sc2.position import Point2
from sc2.units import Units

from bot.tactics.consts import (
    MAXED_SUPPLY_CAP,
    MAXED_SUPPLY_LEFT,
    STAGING_MIN_TANKS,
    STAGING_STANDOFF,
    STAGING_TRIGGER_RANGE,
)
from bot.utils.macro_utils import closest_ramp_point, get_rally_point
from bot.utils.geometry import plain_point

TANK_SPACING = 3.0
TANK_BACK_OFFSET = 3.0
BIO_FRONT_OFFSET = 2.0
APPROACH_DISTANCE = 18.0
APPROACH_DISTANCE_SQ = APPROACH_DISTANCE * APPROACH_DISTANCE
APPROACH_REFRESH = 20.0
APPROACH_MATCH_DISTANCE_SQ = 16.0
DIVERSION_TARGET_SPACING = 15.0
DIVERSION_TARGET_SPACING_SQ = DIVERSION_TARGET_SPACING * DIVERSION_TARGET_SPACING
GIVE_UP_RADIUS = 12.0
GIVE_UP_RADIUS_SQ = GIVE_UP_RADIUS * GIVE_UP_RADIUS
STAGING_MAX_PULL_BACK = 8.0
HOLD_SAFE_SEARCH_RADIUS = 12
SLOT_SAFE_SEARCH_RADIUS = 6

STAGING_STRUCTURE_TYPES = frozenset({U.PHOTONCANNON, U.SPINECRAWLER, U.PLANETARYFORTRESS, U.BUNKER})

TOWNHALL_TYPES = frozenset({
    U.COMMANDCENTER, U.COMMANDCENTERFLYING, U.ORBITALCOMMAND, U.ORBITALCOMMANDFLYING, U.PLANETARYFORTRESS,
    U.NEXUS, U.HATCHERY, U.LAIR, U.HIVE,
})


class Positioning:
    def __init__(self, ai):
        self.ai = ai
        self._approach: Optional[Point2] = None
        self._approach_for: Optional[Point2] = None
        self._approach_time: float = -1e9
        self._given_up: List[Tuple[Point2, float]] = []

    def _ground_safe_grid(self) -> np.ndarray:
        return np.maximum(
            self.ai.mediator.get_ground_grid,
            self.ai.mediator.get_priority_ground_avoidance_grid,
        )

    def hold_point(self) -> Point2:
        rally = get_rally_point(self.ai)
        grid = self._ground_safe_grid()
        if self.ai.mediator.is_position_safe(grid=grid, position=rally):
            return rally
        return self.ai.mediator.find_closest_safe_spot(
            from_pos=rally, grid=grid, radius=HOLD_SAFE_SEARCH_RADIUS,
        )

    def approach_point(self, hold_point: Point2) -> Point2:
        now = self.ai.time
        if (
            self._approach is not None
            and self._approach_for is not None
            and cy_distance_to_squared(self._approach_for, hold_point) < APPROACH_MATCH_DISTANCE_SQ
            and now - self._approach_time < APPROACH_REFRESH
        ):
            return self._approach

        approach: Optional[Point2] = None
        try:
            path = self.ai.mediator.find_raw_path(
                start=self.ai.enemy_start_locations[0],
                target=hold_point,
                grid=self.ai.mediator.get_cached_ground_grid,
                sensitivity=6,
            )
            if path:
                for point in reversed(path):
                    if cy_distance_to_squared(point, hold_point) >= APPROACH_DISTANCE_SQ:
                        approach = plain_point(point)
                        break
        except Exception:
            approach = None

        if approach is None:
            approach = Point2(cy_towards(hold_point, self.ai.enemy_start_locations[0], APPROACH_DISTANCE))

        self._approach, self._approach_for, self._approach_time = approach, hold_point, now
        return approach

    def front_vector(self, hold_point: Point2) -> Point2:
        toward = self.approach_point(hold_point) - hold_point
        return toward.normalized if toward.length > 0.01 else Point2((1.0, 0.0))

    def bio_position(self, hold_point: Point2) -> Point2:
        forward = hold_point + self.front_vector(hold_point) * BIO_FRONT_OFFSET
        if self.ai.get_terrain_height(forward) < self.ai.get_terrain_height(hold_point):
            forward = hold_point
        forward = self._pathable(forward, hold_point)
        grid = self._ground_safe_grid()
        if self.ai.mediator.is_position_safe(grid=grid, position=forward):
            return forward
        return self.ai.mediator.find_closest_safe_spot(
            from_pos=forward, grid=grid, radius=SLOT_SAFE_SEARCH_RADIUS,
        )

    def hold_at_ramp(self, threat_position: Point2) -> Point2:
        townhalls = self.ai.townhalls
        if not townhalls:
            return threat_position

        base_position = cy_closest_to(threat_position, townhalls).position
        if self.ai.get_terrain_height(threat_position) >= self.ai.get_terrain_height(base_position):
            return threat_position

        ramp_point = closest_ramp_point(self.ai, base_position)
        target = threat_position if ramp_point is None else ramp_point

        grid = self._ground_safe_grid()
        if self.ai.mediator.is_position_safe(grid=grid, position=target):
            return target
        return self.ai.mediator.find_closest_safe_spot(
            from_pos=target, grid=grid, radius=HOLD_SAFE_SEARCH_RADIUS,
        )

    def tank_slots(self, hold_point: Point2, count: int, front: Optional[Point2] = None) -> List[Point2]:
        if front is None:
            front = self.front_vector(hold_point)

        lateral = Point2((-front.y, front.x))
        base = hold_point - front * TANK_BACK_OFFSET
        grid = self._ground_safe_grid()
        slots: List[Point2] = []

        for i in range(count):
            k = ((i + 1) // 2) * (1 if i % 2 == 1 else -1)
            pos = self._pathable(base + lateral * (k * TANK_SPACING), hold_point)
            if self.ai.mediator.is_position_safe(grid=grid, position=pos):
                slots.append(pos)
            else:
                slots.append(self.ai.mediator.find_closest_safe_spot(
                    from_pos=pos, grid=grid, radius=SLOT_SAFE_SEARCH_RADIUS,
                ))

        return slots

    def _pathable(self, position: Point2, fallback: Point2) -> Point2:
        if self.ai.in_map_bounds(position) and self.ai.in_pathing_grid(position):
            return position
        return fallback

    def give_up(self, position: Point2, seconds: float) -> None:
        now = self.ai.time
        self._given_up = [(p, until) for p, until in self._given_up if until > now]
        self._given_up.append((position, now + seconds))

    def is_given_up(self, position: Point2) -> bool:
        return self._given_up_until(position) > self.ai.time

    def _given_up_until(self, position: Point2) -> float:
        return max(
            (until for p, until in self._given_up if cy_distance_to_squared(p, position) <= GIVE_UP_RADIUS_SQ),
            default=0.0,
        )

    def attack_target(self) -> Point2:
        structures: Units = self.ai.enemy_structures
        origin = self.ai.start_location
        candidates: List[Point2] = []

        if structures:
            ground = [s for s in structures if not s.is_flying] or list(structures)
            townhalls = cy_sorted_by_distance_to(
                [s for s in ground if s.type_id in TOWNHALL_TYPES], origin,
            )
            townhall_tags = {s.tag for s in townhalls}
            others = cy_sorted_by_distance_to(
                [s for s in ground if s.tag not in townhall_tags], origin,
            )
            candidates = [s.position for s in townhalls] + [s.position for s in others]

        candidates.append(self.ai.enemy_start_locations[0])

        for candidate in candidates:
            if not self.is_given_up(candidate):
                return candidate

        return min(candidates, key=self._given_up_until)

    def diversion_target(self, main_target: Point2) -> Optional[Point2]:
        enemy_home = self.ai.enemy_start_locations[0]
        structures: Units = self.ai.enemy_structures

        if (
            cy_distance_to_squared(enemy_home, main_target) > DIVERSION_TARGET_SPACING_SQ
            and not self.is_given_up(enemy_home)
        ):
            return enemy_home

        candidates: List[Point2] = []
        for loc, _ in self.ai.mediator.get_enemy_expansions:
            if (
                cy_distance_to_squared(loc, main_target) <= DIVERSION_TARGET_SPACING_SQ
                or self.is_given_up(loc)
            ):
                continue
            if cy_distance_to_squared(loc, self.ai.start_location) <= DIVERSION_TARGET_SPACING_SQ:
                continue
            if structures and cy_closer_than(structures, 12, loc):
                candidates.append(loc)

        if not candidates:
            return None

        return min(candidates, key=lambda loc: cy_distance_to_squared(loc, enemy_home))

    def staging_point(self, anchor: Point2, tank_count: int) -> Optional[Point2]:
        if tank_count < STAGING_MIN_TANKS:
            return None

        ai = self.ai
        candidates = [
            s for s in ai.enemy_structures
            if s.type_id in STAGING_STRUCTURE_TYPES and s.build_progress >= 1
        ] + [
            e for e in ai.enemy_units
            if e.type_id == U.SIEGETANKSIEGED and not e.is_memory
        ]

        if not candidates:
            return None

        nearest = cy_closest_to(anchor, candidates)
        distance = cy_distance_to(nearest.position, anchor)

        if distance > STAGING_TRIGGER_RANGE or distance <= STAGING_STANDOFF + 2.0:
            return None

        grid = self._ground_safe_grid()
        pull_back = 0.0
        while pull_back <= STAGING_MAX_PULL_BACK:
            standoff = min(STAGING_STANDOFF + pull_back, distance - 1.0)
            point = Point2(cy_towards(nearest.position, anchor, standoff))
            if self._walkable(point) and self.ai.mediator.is_position_safe(grid=grid, position=point):
                return point
            pull_back += 1.0

        return None

    def _walkable(self, position: Point2) -> bool:
        return bool(self.ai.in_map_bounds(position) and self.ai.in_pathing_grid(position))

    def is_maxed(self) -> bool:
        return self.ai.supply_cap >= MAXED_SUPPLY_CAP and self.ai.supply_left <= MAXED_SUPPLY_LEFT