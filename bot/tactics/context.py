from functools import cached_property
from typing import Dict, List, Optional, Tuple

import numpy as np
from cython_extensions.geometry import cy_distance_to_squared

from ares.consts import UnitTreeQueryType
from sc2.ids.upgrade_id import UpgradeId
from sc2.position import Point2
from sc2.unit import Unit
from sc2.units import Units

from bot.tactics.consts import (
    ATTACK_TARGET_IGNORE,
    BANELING_KITE_MARGIN,
    BANELING_TYPES,
    ENEMY_WORKER_TYPES,
    LOCAL_FIGHT_RADIUS_SQ,
    MELEE_KITE_MARGIN,
    MELEE_KITE_MAX_SPEED_RATIO,
    MELEE_RANGE_THRESHOLD,
    NEAR_ENEMY_RADIUS,
)
from bot.tactics.local_fight import FightMap

_GROUND_TARGET, _AIR_TARGET, _BANELING, _MELEE = 1, 2, 4, 8


class ArmyContext:
    def __init__(self, ai, positioning):
        self.ai = ai
        self.mediator = ai.mediator
        self.time: float = ai.time
        self.positioning = positioning

        self.hold_point: Point2 = positioning.hold_point()
        self.front: Point2 = positioning.front_vector(self.hold_point)
        self.bio_position: Point2 = positioning.bio_position(self.hold_point)
        self.enemy_start: Point2 = ai.enemy_start_locations[0]

        self.ground_grid: np.ndarray = self.mediator.get_ground_grid
        self.air_grid: np.ndarray = self.mediator.get_air_grid
        self.climber_grid: np.ndarray = self.mediator.get_climber_grid

        self.stim_researched: bool = ai.already_pending_upgrade(UpgradeId.STIMPACK) == 1
        self.cloak_researched: bool = ai.already_pending_upgrade(UpgradeId.BANSHEECLOAK) == 1

        self.fights: FightMap = FightMap()

        self._near: Dict[int, Units] = {}
        self._near_visible: Dict[int, List[Unit]] = {}
        self._banelings: Dict[int, List[Unit]] = {}
        self._wide: Dict[Tuple[int, float], Units] = {}
        self._kinds: Dict[int, Tuple[Unit, int]] = {}
        self._melee: Dict[int, List[Unit]] = {}
        self.tank_closest: Dict[int, Optional[float]] = {}

    @cached_property
    def _baneling_present(self) -> bool:
        return any(e.type_id in BANELING_TYPES for e in self.ai.enemy_units)

    @cached_property
    def _melee_present(self) -> bool:
        return any(self._kind(e) & _MELEE for e in self.ai.enemy_units)

    def prefetch_near(self, units, radius: float = NEAR_ENEMY_RADIUS) -> None:
        unit_list = [u for u in units if u.tag not in self._near]
        if not unit_list:
            return
        result = self.mediator.get_units_in_range(
            start_points=unit_list,
            distances=radius,
            query_tree=UnitTreeQueryType.AllEnemy,
            return_as_dict=True,
        )
        for unit in unit_list:
            self._near[unit.tag] = result.get(unit.tag, Units([], self.ai))

    def enemies_near(self, unit: Unit) -> Units:
        cached = self._near.get(unit.tag)
        if cached is None:
            self.prefetch_near([unit])
            cached = self._near.get(unit.tag, Units([], self.ai))
        return cached

    def enemies_within(self, unit: Unit, radius: float) -> Units:
        key = (unit.tag, radius)
        cached = self._wide.get(key)
        if cached is None:
            result = self.mediator.get_units_in_range(
                start_points=[unit],
                distances=radius,
                query_tree=UnitTreeQueryType.AllEnemy,
                return_as_dict=True,
            )
            cached = result.get(unit.tag, Units([], self.ai))
            self._wide[key] = cached
        return cached

    def _kind(self, enemy: Unit) -> int:
        entry = self._kinds.get(id(enemy))
        if entry is None:
            kind = 0
            if not enemy.is_memory and not enemy.is_hallucination:
                type_id = enemy.type_id
                if type_id in BANELING_TYPES:
                    kind |= _BANELING
                if type_id not in ATTACK_TARGET_IGNORE:
                    kind |= _AIR_TARGET if enemy.is_flying else _GROUND_TARGET
                    if (
                        not enemy.is_flying
                        and not enemy.is_structure
                        and type_id not in BANELING_TYPES
                        and type_id not in ENEMY_WORKER_TYPES
                        and 0 < enemy.ground_range <= MELEE_RANGE_THRESHOLD
                    ):
                        kind |= _MELEE
            entry = self._kinds[id(enemy)] = (enemy, kind)
        return entry[1]

    def targets_near(self, unit: Unit) -> List[Unit]:
        cached = self._near_visible.get(unit.tag)
        if cached is not None:
            return cached
        wanted = (_AIR_TARGET if unit.can_attack_air else 0) | (_GROUND_TARGET if unit.can_attack_ground else 0)
        targets = [e for e in self.enemies_near(unit) if self._kind(e) & wanted]
        self._near_visible[unit.tag] = targets
        return targets

    def banelings_near(self, unit: Unit) -> List[Unit]:
        cached = self._banelings.get(unit.tag)
        if cached is None:
            if not self._baneling_present:
                cached = []
            else:
                cached = [
                    e for e in self.enemies_near(unit)
                    if self._kind(e) & _BANELING
                    and cy_distance_to_squared(e.position, unit.position) <= LOCAL_FIGHT_RADIUS_SQ
                ]
            self._banelings[unit.tag] = cached
        return cached

    def _melee_enemies(self, unit: Unit, radius: float) -> List[Unit]:
        if not self._melee_present:
            return []
        radius_sq = radius * radius
        return [
            e for e in self.enemies_near(unit)
            if self._kind(e) & _MELEE
            and cy_distance_to_squared(e.position, unit.position) <= radius_sq
        ]

    def close_melee(self, unit: Unit) -> List[Unit]:
        cached = self._melee.get(unit.tag)
        if cached is None:
            speed = max(unit.real_speed, 0.1) * MELEE_KITE_MAX_SPEED_RATIO
            cached = [
                e for e in self._melee_enemies(unit, unit.ground_range + MELEE_KITE_MARGIN)
                if e.real_speed <= speed
            ]
            self._melee[unit.tag] = cached
        return cached

    def melee_within(self, unit: Unit, radius: float) -> List[Unit]:
        return self._melee_enemies(unit, radius)

    def close_banelings(self, unit: Unit) -> List[Unit]:
        reach = unit.ground_range + BANELING_KITE_MARGIN
        reach_sq = reach * reach
        return [
            b for b in self.banelings_near(unit)
            if cy_distance_to_squared(b.position, unit.position) <= reach_sq
        ]

    def grid_for(self, unit: Unit) -> np.ndarray:
        return self.air_grid if unit.is_flying else self.ground_grid

    def is_safe(self, unit: Unit, grid: Optional[np.ndarray] = None) -> bool:
        return self.mediator.is_position_safe(
            grid=self.grid_for(unit) if grid is None else grid,
            position=unit.position,
        )