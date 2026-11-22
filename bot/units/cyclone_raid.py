import math
from types import SimpleNamespace
from typing import Dict, List, Optional, Tuple

import numpy as np

from cython_extensions.units_utils import cy_closest_to
from cython_extensions.geometry import cy_distance_to_squared, cy_towards

from sc2.ids.ability_id import AbilityId
from sc2.ids.unit_typeid import UnitTypeId as U
from sc2.position import Point2
from sc2.unit import Unit
from sc2.units import Units

from bot.tactics.consts import ATTACK_TARGET_IGNORE, ATTACK_TARGET_IGNORE_WITH_WORKERS, ENEMY_WORKER_TYPES, FIGHT_GHOST_MAX_AGE, MELEE_RANGE_THRESHOLD
from bot.tactics.context import ArmyContext
from bot.tactics.orders import GroupOrders
from bot.units.common import move_to, path_move, step_back_from
from bot.units.cyclones import LOCK_ON_ABILITIES, CycloneController
from bot.units.danger_memory import DangerMemory, ground_defenders

RAID_SIGHT = 28.0
RAID_ENGAGE_RANGE = 20.0
RAID_MARGIN = 1.5
RAID_MELEE_EXTRA = 2.0
RAID_CAST_SLACK = 0.6
RAID_IDLE_MARGIN = 0.5
MEMORY_SECONDS = 30.0

RAID_STRUCTURES = {U.SHIELDBATTERY: 1, U.PHOTONCANNON: 2}
WORKER_RANK = 3
RAID_IGNORE = frozenset({U.INTERCEPTOR, U.ADEPTPHASESHIFT, U.DISRUPTORPHASED})


class CycloneRaid:
    def __init__(self, ai, cyclones: CycloneController):
        self.ai = ai
        self.cyclones = cyclones
        self._memo: Dict[int, Tuple[Unit, Optional[int], float]] = {}
        self.danger = DangerMemory(ai)

    def control(self, units: Units, orders: GroupOrders, ctx: ArmyContext) -> None:
        self._memo = {}
        self.cyclones.begin_step(units, ctx)
        self.danger.refresh(ground_defenders(self.ai))
        for unit in units:
            self._control_unit(unit, orders, ctx)

    def _control_unit(self, unit: Unit, orders: GroupOrders, ctx: ArmyContext) -> None:
        if self.cyclones.handle_running_lock(unit, orders, ctx):
            return

        if self.cyclones.repair_trip(unit, orders, ctx):
            return

        pool: List[Unit] = list(ctx.enemies_within(unit, RAID_SIGHT))
        ranks: List[Optional[int]] = []
        threats: List[Unit] = []

        for enemy in pool:
            rank, enemy_reach = self._info(enemy, unit)
            ranks.append(rank)
            if enemy_reach > 0:
                threats.append(enemy)

        xy = np.array([e.position_tuple for e in threats], dtype=float).reshape(-1, 2)
        reach = np.array([self._info(e, unit)[1] for e in threats], dtype=float)
        tags = np.array([e.tag for e in threats], dtype=np.int64)

        covering = [threats[k] for k in np.nonzero(np.hypot(xy[:, 0] - unit.position.x, xy[:, 1] - unit.position.y) <= reach)[0]]
        covering = covering + [
            SimpleNamespace(position=p)
            for p, r in self.danger.spots()
            if cy_distance_to_squared(unit.position, p) <= r * r
        ]

        if any(ability in unit.abilities for ability in LOCK_ON_ABILITIES):
            picked = self._pick_target(unit, pool, ranks, xy, reach, tags)
            if picked is not None:
                target, cast_from, in_range = picked
                if in_range:
                    if self.cyclones.cast_lock_on(unit, target):
                        return
                else:
                    move_to(unit, cast_from)
                    return

            if covering:
                step_back_from(self.ai, ctx, unit, covering, orders)
            else:
                threat_circles = [(Point2((float(x), float(y))), float(r)) for (x, y), r in zip(xy, reach)]
                remembered = [
                    (p, r) for p, r in self.danger.spots()
                    if not any(cy_distance_to_squared(p, t) <= 1.0 for t, _ in threat_circles)
                ]
                cast_standoff = self.cyclones.lockon_range - RAID_CAST_SLACK
                standoff = [(e.position, cast_standoff) for e, rank in zip(pool, ranks) if rank is not None]
                self._search(unit, pool, threat_circles + remembered + standoff, orders, ctx)
            return

        if covering:
            step_back_from(self.ai, ctx, unit, covering, orders)
        else:
            self._keep_out_of_own_range(unit, pool, orders, ctx)

    def _info(self, enemy: Unit, unit: Unit) -> Tuple[Optional[int], float]:
        entry = self._memo.get(id(enemy))
        if entry is None:
            entry = self._memo[id(enemy)] = (enemy, self._rank(enemy), self._reach(enemy, unit) if self._threatens(enemy, unit) else 0.0)
        return entry[1], entry[2]

    @staticmethod
    def _threatens(enemy: Unit, unit: Unit) -> bool:
        if enemy.is_hallucination or enemy.type_id in RAID_IGNORE or enemy.type_id in ENEMY_WORKER_TYPES:
            return False
        if enemy.is_memory and enemy.age > FIGHT_GHOST_MAX_AGE:
            return False
        if not (enemy.can_attack_air if unit.is_flying else enemy.can_attack_ground):
            return False
        return not (enemy.is_structure and (not enemy.is_ready or not enemy.is_powered))

    @staticmethod
    def _reach(enemy: Unit, unit: Unit) -> float:
        weapon = enemy.air_range if unit.is_flying else enemy.ground_range
        reach = weapon + enemy.radius + unit.radius + RAID_MARGIN
        if enemy.ground_range <= MELEE_RANGE_THRESHOLD:
            reach += RAID_MELEE_EXTRA
        return reach

    @staticmethod
    def _rank(enemy: Unit) -> Optional[int]:
        if enemy.is_memory or enemy.is_hallucination or not enemy.is_visible or (enemy.is_cloaked and not enemy.is_revealed):
            return None
        if enemy.type_id in RAID_IGNORE:
            return None
        if enemy.is_structure:
            return RAID_STRUCTURES.get(enemy.type_id)
        if enemy.type_id in ENEMY_WORKER_TYPES:
            return WORKER_RANK
        return None if enemy.type_id in ATTACK_TARGET_IGNORE_WITH_WORKERS else 0

    def _pick_target(
        self, unit: Unit, pool: List[Unit], ranks: List[Optional[int]], xy: np.ndarray, reach: np.ndarray, tags: np.ndarray
    ) -> Optional[Tuple[Unit, Point2, bool]]:
        cast_range = self.cyclones.lockon_range
        unit_radius = unit.radius
        candidates = []

        for enemy, rank in zip(pool, ranks):
            if rank is None or enemy.tag in self.cyclones.lock_ons:
                continue

            ability = AbilityId.LOCKONAIR_LOCKONAIR if enemy.is_flying else AbilityId.LOCKON_LOCKON
            if ability not in unit.abilities:
                continue

            enemy_radius = enemy.radius
            dist_sq = cy_distance_to_squared(unit.position, enemy.position)
            if dist_sq > (RAID_ENGAGE_RANGE + unit_radius + enemy_radius) ** 2:
                continue

            in_range = dist_sq <= (cast_range + unit_radius + enemy_radius) ** 2
            cast_from = (
                unit.position
                if in_range
                else Point2(cy_towards(enemy.position, unit.position, enemy_radius + unit_radius + cast_range - RAID_CAST_SLACK))
            )
            candidates.append(((rank, cy_distance_to_squared(unit.position, cast_from), enemy.tag), enemy, cast_from, in_range))

        candidates.sort(key=lambda c: c[0])

        for _, enemy, cast_from, in_range in candidates:
            others = tags != enemy.tag
            if not others.any():
                return enemy, cast_from, in_range

            if _any_within(xy[others], reach[others], cast_from):
                continue

            if not in_range and _any_within_reach_of_segment(xy[others], reach[others], unit.position, cast_from):
                continue

            return enemy, cast_from, in_range

        return None

    def _search(self, unit: Unit, pool: List[Unit], circles: List[Tuple[Point2, float]], orders: GroupOrders, ctx: ArmyContext) -> None:
        destination = self._destination(unit, ctx)
        point = clip_before(unit.position, destination, circles)

        if cy_distance_to_squared(point, unit.position) > 1.0:
            path_move(self.ai, ctx, unit, point)
        else:
            self._keep_out_of_own_range(unit, pool, orders, ctx)

    def _keep_out_of_own_range(self, unit: Unit, pool: List[Unit], orders: GroupOrders, ctx: ArmyContext) -> None:
        reach_with_radius = unit.ground_range + RAID_IDLE_MARGIN + unit.radius
        close = [
            e for e in pool
            if not e.is_memory and e.is_visible and not e.is_hallucination and e.type_id not in RAID_IGNORE
            and (unit.can_attack_air if e.is_flying else unit.can_attack_ground)
            and cy_distance_to_squared(unit.position, e.position) <= (reach_with_radius + e.radius) ** 2
        ]

        if close:
            step_back_from(self.ai, ctx, unit, close, orders)

    def _destination(self, unit: Unit, ctx: ArmyContext) -> Point2:
        ai = self.ai
        points: List[Point2] = [s.position for s in ai.enemy_structures if s.type_id in RAID_STRUCTURES or s.type_id == U.NEXUS]
        points += [
            e.position for e in ai.enemy_units
            if not e.is_structure and not e.is_hallucination and e.type_id not in ATTACK_TARGET_IGNORE and e.type_id not in RAID_IGNORE
            and (not e.is_memory or e.age <= MEMORY_SECONDS)
        ]

        if not points:
            return ctx.mediator.get_enemy_nat

        return cy_closest_to(unit.position, points)


def _any_within(xy: np.ndarray, reach: np.ndarray, p: Point2) -> bool:
    return bool((np.hypot(xy[:, 0] - p.x, xy[:, 1] - p.y) <= reach).any())


def _any_within_reach_of_segment(xy: np.ndarray, reach: np.ndarray, a: Point2, b: Point2) -> bool:
    dx, dy = b.x - a.x, b.y - a.y
    length2 = dx * dx + dy * dy
    t = 0.0 if length2 < 1e-9 else np.clip(((xy[:, 0] - a.x) * dx + (xy[:, 1] - a.y) * dy) / length2, 0.0, 1.0)
    return bool((np.hypot(a.x + dx * t - xy[:, 0], a.y + dy * t - xy[:, 1]) < reach).any())


def clip_before(a: Point2, b: Point2, circles: List[Tuple[Point2, float]]) -> Point2:
    dx, dy = b.x - a.x, b.y - a.y
    length = math.hypot(dx, dy)

    if length < 1e-9:
        return a

    ux, uy = dx / length, dy / length
    reach = length

    for centre, radius in circles:
        fx, fy = a.x - centre.x, a.y - centre.y
        c = fx * fx + fy * fy - radius * radius
        if c <= 0:
            return a

        half = fx * ux + fy * uy
        discriminant = half * half - c
        if discriminant < 0:
            continue

        t = -half - math.sqrt(discriminant)
        if 0 <= t < reach:
            reach = t

    return Point2((a.x + ux * reach, a.y + uy * reach))