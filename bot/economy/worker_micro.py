import math
from typing import Callable, Optional, Sequence, Set

import numpy as np
from scipy.spatial.distance import cdist

from ares.consts import UnitRole
from sc2.bot_ai import BotAI
from sc2.ids.unit_typeid import UnitTypeId
from sc2.position import Point2
from sc2.unit import Unit
from sc2.units import Units

from cython_extensions.units_utils import cy_closer_than, cy_closest_to
from bot.consts import HARMLESS_TO_WORKERS
from bot.utils.geometry import is_already_moving_to, segment_walkable
from bot.economy.repair import structure_repairers

WORKER_FLEE_RANGE = 7.0
WORKER_FLEE_STEP = 6.0
WORKER_FLEE_RELEASE_MARGIN = 3.0
BASE_DANGER_RANGE = 10.0

_ENEMY_WORKER_TYPES = frozenset({UnitTypeId.SCV, UnitTypeId.PROBE, UnitTypeId.DRONE, UnitTypeId.MULE})

ORACLE_AVOID_RANGE = 5.0
ORACLE_ALERT_MARGIN = 2.5
ORACLE_RELEASE_MARGIN = 5.0
ORACLE_FLEE_STEP = 6.0
FLEE_HOME_RADIUS = 22.0

_FLEE_TURNS = (0.0, 30.0, -30.0, 60.0, -60.0, 90.0, -90.0)
_ANY_TURN = _FLEE_TURNS + (120.0, -120.0, 150.0, -150.0, 180.0)


def base_is_threatened(self: BotAI, position: Point2, radius: float = BASE_DANGER_RANGE) -> bool:
    threats = self.visible_enemy_units.filter(
        lambda u: u.can_attack_ground and u.type_id not in _ENEMY_WORKER_TYPES and u.type_id not in HARMLESS_TO_WORKERS
    )
    if not threats:
        return False
    return len(cy_closer_than(threats, radius, position)) > 0


def prevent_PF_rush(self: BotAI):
    enemy_flying_structures: Units = self.enemy_structures.of_type({UnitTypeId.COMMANDCENTERFLYING})
    if enemy_flying_structures.amount == 0 or self.workers.gathering.amount == 0:
        return

    keys = [i for i in self.worker_assigned_to_follow.keys()]
    for i in keys:
        worker_tag = self.worker_assigned_to_follow[i]
        if enemy_flying_structures.find_by_tag(i) is None or (worker_tag != -1 and self.workers.find_by_tag(worker_tag) is None):
            if worker_tag != -1 and self.workers.find_by_tag(worker_tag) is not None:
                self.mediator.assign_role(tag=worker_tag, role=UnitRole.GATHERING)
            self.worker_assigned_to_follow.pop(i, None)

    for i in enemy_flying_structures:
        if not i.tag in self.worker_assigned_to_follow.keys():
            self.worker_assigned_to_follow[i.tag] = -1

    for i in self.structures:
        closest_enemy_struct = enemy_flying_structures.closest_to(i)
        if closest_enemy_struct.distance_to(i) > 14 or self.workers.gathering.amount == 0:
            continue
        if self.worker_assigned_to_follow[closest_enemy_struct.tag] != -1:
            self.workers.find_by_tag(self.worker_assigned_to_follow[closest_enemy_struct.tag]).move(closest_enemy_struct.position)
            continue
        closest_worker: Unit = cy_closest_to(closest_enemy_struct.position, self.workers.gathering)
        closest_worker.move(closest_enemy_struct.position)
        self.mediator.assign_role(tag=closest_worker.tag, role=UnitRole.DEFENDING)
        self.worker_assigned_to_follow[closest_enemy_struct.tag] = closest_worker.tag


def defend_building_workers(self: BotAI):
    enemy_workers = self.visible_enemy_units.of_type({UnitTypeId.SCV, UnitTypeId.PROBE, UnitTypeId.DRONE})
    if enemy_workers.amount == 0 or self.workers.gathering.amount == 0:
        return

    for i in self.workers:
        if not i.is_constructing_scv or enemy_workers.closest_distance_to(i) > 10:
            continue
        if not i.tag in self.worker_assigned_to_defend.keys() or self.workers.find_by_tag(self.worker_assigned_to_defend[i.tag]) is None:
            self.worker_assigned_to_defend[i.tag] = -1

    keys = [i for i in self.worker_assigned_to_defend.keys()]
    for i in keys:
        if self.workers.find_by_tag(i) is None or len(cy_closer_than(enemy_workers, 10, self.workers.find_by_tag(i).position)) > 1:
            if self.worker_assigned_to_defend[i] != -1 and self.workers.find_by_tag(self.worker_assigned_to_defend[i]) is not None:
                defender = self.workers.find_by_tag(self.worker_assigned_to_defend[i])
                defender.move(self.townhalls.first)
                self.mediator.assign_role(tag=defender.tag, role=UnitRole.GATHERING)
            self.worker_assigned_to_defend.pop(i)

    for i in self.worker_assigned_to_defend.keys():
        if self.worker_assigned_to_defend[i] != -1:
            continue
        closest_worker = cy_closest_to(self.workers.find_by_tag(i).position, self.workers.gathering)
        closest_worker.attack(self.workers.find_by_tag(i).position)
        self.mediator.assign_role(tag=closest_worker.tag, role=UnitRole.DEFENDING)
        self.worker_assigned_to_defend[i] = closest_worker.tag


def point_away_from(origin: Point2, threats: Sequence[Point2], step: float, ok: Callable[[Point2], bool]) -> Optional[Point2]:
    vx = vy = 0.0
    for threat in threats:
        dx, dy = origin.x - threat.x, origin.y - threat.y
        gap = math.hypot(dx, dy)
        if gap > 0.05:
            vx += dx / (gap * gap)
            vy += dy / (gap * gap)

    length = math.hypot(vx, vy)
    turns = _FLEE_TURNS

    if length < 1e-9:
        turns = _ANY_TURN
        vx, vy, length = 1.0, 0.0, 1.0
        for threat in threats:
            dx, dy = origin.x - threat.x, origin.y - threat.y
            gap = math.hypot(dx, dy)
            if gap > 0.05:
                vx, vy, length = -dy, dx, gap
                break

    vx, vy = vx / length, vy / length
    for turn in turns:
        angle = math.radians(turn)
        dx = vx * math.cos(angle) - vy * math.sin(angle)
        dy = vx * math.sin(angle) + vy * math.cos(angle)
        candidate = Point2((origin.x + dx * step, origin.y + dy * step))
        if ok(candidate):
            return candidate
    return None


def _safe_grid(self: BotAI) -> np.ndarray:
    return np.maximum(self.mediator.get_ground_grid, self.mediator.get_priority_ground_avoidance_grid)


def _avoidance_grid(self: BotAI) -> np.ndarray:
    return self.mediator.get_ground_avoidance_grid


def _flee_spot_fallback(self: BotAI, worker: Unit) -> Point2:
    return self.mediator.find_closest_safe_spot(
        from_pos=worker.position,
        grid=_safe_grid(self),
        radius=int(FLEE_HOME_RADIUS),
    )


def avoid_oracles(self: BotAI) -> Set[int]:
    handled: Set[int] = set()
    fleeing: Set[int] = self.oracle_fleeing
    oracles = [] if self.worker_rushed else [o for o in self.visible_enemy_units.of_type({UnitTypeId.ORACLE}) if not o.is_hallucination]

    if not oracles:
        for tag in fleeing:
            if self.workers.find_by_tag(tag) is not None:
                self.mediator.assign_role(tag=tag, role=UnitRole.GATHERING)
        fleeing.clear()
        return handled

    fleeing &= {w.tag for w in self.workers}
    fearless = structure_repairers(self)
    workers = [w for w in self.workers if not (w.is_repairing or w.is_constructing_scv or w.tag in fearless)]

    if not workers:
        return handled

    avoidance = _avoidance_grid(self)
    positions = [o.position for o in oracles]
    gaps = cdist(np.array([w.position_tuple for w in workers]), np.array([o.position_tuple for o in oracles]))
    alert = ORACLE_AVOID_RANGE + ORACLE_ALERT_MARGIN
    release = ORACLE_AVOID_RANGE + ORACLE_RELEASE_MARGIN

    for worker, row in zip(workers, gaps):
        nearest = row.min()
        if nearest < alert:
            def ok(point: Point2, start: Point2 = worker.position) -> bool:
                return bool(
                    self.in_map_bounds(point) and self.in_pathing_grid(point)
                    and self.townhalls.closest_distance_to(point) <= FLEE_HOME_RADIUS
                    and segment_walkable(start, point, self.in_pathing_grid)
                    and self.mediator.is_position_safe(grid=avoidance, position=point)
                )

            target = point_away_from(worker.position, [p for p, gap in zip(positions, row) if gap < alert], ORACLE_FLEE_STEP, ok)
            if target is not None:
                fleeing.add(worker.tag)
                handled.add(worker.tag)
                self.mediator.assign_role(tag=worker.tag, role=UnitRole.DEFENDING)
                if not is_already_moving_to(worker, target, epsilon=1.0):
                    worker.move(target)
            else:
                fallback = _flee_spot_fallback(self, worker)
                fleeing.add(worker.tag)
                handled.add(worker.tag)
                self.mediator.assign_role(tag=worker.tag, role=UnitRole.DEFENDING)
                if not is_already_moving_to(worker, fallback, epsilon=1.0):
                    worker.move(fallback)
        elif worker.tag in fleeing:
            if nearest >= release:
                fleeing.discard(worker.tag)
                self.mediator.assign_role(tag=worker.tag, role=UnitRole.GATHERING)
            else:
                handled.add(worker.tag)
    return handled


def flee_worker_threats(self: BotAI, skip: Set[int] = frozenset()):
    if self.worker_rushed:
        for tag in self.threat_fleeing:
            if self.workers.find_by_tag(tag) is not None:
                self.mediator.assign_role(tag=tag, role=UnitRole.GATHERING)
        self.threat_fleeing.clear()
        return

    threats: Units = self.visible_enemy_units.filter(
        lambda u: u.can_attack_ground and u.type_id not in {UnitTypeId.PROBE, UnitTypeId.SCV, UnitTypeId.DRONE, UnitTypeId.ORACLE}
        and u.type_id not in HARMLESS_TO_WORKERS
    )
    fleeing: Set[int] = self.threat_fleeing

    if threats.amount == 0:
        for tag in fleeing:
            if self.workers.find_by_tag(tag) is not None:
                self.mediator.assign_role(tag=tag, role=UnitRole.GATHERING)
        fleeing.clear()
        return

    fleeing &= {w.tag for w in self.workers}
    fearless = structure_repairers(self)
    workers = [w for w in self.workers if not (w.is_repairing or w.is_constructing_scv or w.tag in skip or w.tag in fearless)]

    if not workers:
        return

    avoidance = _avoidance_grid(self)
    positions = [t.position for t in threats]
    gaps = cdist(np.array([w.position_tuple for w in workers]), np.array([t.position_tuple for t in threats]))
    release = WORKER_FLEE_RANGE + WORKER_FLEE_RELEASE_MARGIN

    for worker, row in zip(workers, gaps):
        nearest = row.min()
        if nearest < WORKER_FLEE_RANGE:
            def ok(point: Point2, start: Point2 = worker.position) -> bool:
                return bool(
                    self.in_map_bounds(point) and self.in_pathing_grid(point)
                    and self.townhalls.closest_distance_to(point) <= FLEE_HOME_RADIUS
                    and segment_walkable(start, point, self.in_pathing_grid)
                    and self.mediator.is_position_safe(grid=avoidance, position=point)
                )

            nearby = [p for p, gap in zip(positions, row) if gap < WORKER_FLEE_RANGE]
            safe_spot = point_away_from(worker.position, nearby, WORKER_FLEE_STEP, ok)
            if safe_spot is None:
                safe_spot = _flee_spot_fallback(self, worker)
            fleeing.add(worker.tag)
            self.mediator.assign_role(tag=worker.tag, role=UnitRole.DEFENDING)
            if not is_already_moving_to(worker, safe_spot):
                worker.move(safe_spot)
        elif worker.tag in fleeing and nearest >= release:
            fleeing.discard(worker.tag)
            self.mediator.assign_role(tag=worker.tag, role=UnitRole.GATHERING)


def worker_micro(self: BotAI):
    prevent_PF_rush(self)
    defend_building_workers(self)
    skip = avoid_oracles(self)
    if self.scout_worker_tag is not None:
        skip.add(self.scout_worker_tag)
    flee_worker_threats(self, skip=skip)