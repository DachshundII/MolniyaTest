import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

from cython_extensions.units_utils import cy_closest_to, cy_sorted_by_distance_to
from cython_extensions.geometry import cy_distance_to_squared

from sc2.bot_ai import BotAI
from sc2.ids.ability_id import AbilityId
from sc2.ids.unit_typeid import UnitTypeId
from sc2.unit import Unit
from sc2.units import Units

from ares.consts import UnitRole

from bot.utils.geometry import path_length, reachable_from_ground

MAX_REPAIRERS = 4
MAX_TRAVEL = 70.0
MAX_TRAVEL_SQ = MAX_TRAVEL * MAX_TRAVEL
HOME_RADIUS = 25.0
HOME_RADIUS_SQ = HOME_RADIUS * HOME_RADIUS
LEASH = 35.0
LEASH_SQ = LEASH * LEASH
STRUCTURE_BELOW = 0.9
UNIT_BELOW = 0.7
ORDER_GRACE = 1.0
WALK_CACHE_SECONDS = 3.0
NO_CREW_BACKOFF = 1.0
_NOT_REPAIRED = {UnitTypeId.SCV, UnitTypeId.MULE, UnitTypeId.AUTOTURRET}


@dataclass
class RepairJob:
    target: int
    is_structure: bool
    since: float
    last: Tuple[float, float]
    walked: float = 0.0


def _home_distance_sq(bases: Units, unit: Unit) -> float:
    return cy_distance_to_squared(cy_closest_to(unit.position, bases).position, unit.position)


def structure_repairers(self: BotAI) -> Set[int]:
    return {tag for tag, job in self.repair_jobs.items() if job.is_structure}


def _release(self: BotAI, worker: Unit, bases: Units) -> None:
    self.repair_jobs.pop(worker.tag, None)
    self.mediator.assign_role(tag=worker.tag, role=UnitRole.GATHERING)
    if not self.mineral_field.empty:
        closest_base = cy_closest_to(worker.position, bases)
        worker.gather(cy_closest_to(closest_base.position, self.mineral_field))


def ground_walk(self: BotAI, worker: Unit, target: Unit) -> Optional[float]:
    key = (worker.tag, target.tag)
    now = self.time
    hit = self.repair_walks.get(key)
    if hit is not None and now - hit[0] < WALK_CACHE_SECONDS:
        return hit[1]
    path = self.mediator.find_raw_path(
        start=worker.position, target=target.position, grid=self.mediator.get_cached_ground_grid, sensitivity=1
    )
    length = path_length(worker.position, path)
    self.repair_walks[key] = (now, length)
    return length


def _follow_jobs(self: BotAI, bases: Units, repairing: List[Unit]) -> Dict[int, List[Unit]]:
    now = self.time
    crews: Dict[int, List[Unit]] = {}
    for worker in repairing:
        target = worker.order_target
        if isinstance(target, int):
            crews.setdefault(target, []).append(worker)
    seen_repairing = {w.tag for w in repairing}
    for tag, job in list(self.repair_jobs.items()):
        worker = self.workers.find_by_tag(tag)
        if worker is None:
            del self.repair_jobs[tag]
        elif tag not in seen_repairing or worker.order_target != job.target:
            if now - job.since > ORDER_GRACE:
                del self.repair_jobs[tag]
            else:
                crews.setdefault(job.target, []).append(worker)
        else:
            x, y = worker.position_tuple
            job.walked += math.hypot(x - job.last[0], y - job.last[1])
            job.last = (x, y)

    for worker in repairing:
        job = self.repair_jobs.get(worker.tag)
        if _home_distance_sq(bases, worker) > LEASH_SQ or (job is not None and job.walked > MAX_TRAVEL):
            _release(self, worker, bases)
            for crew in crews.values():
                if worker in crew:
                    crew.remove(worker)

    for tag, crew in crews.items():
        if len(crew) > MAX_REPAIRERS:
            target = (self.units | self.structures).find_by_tag(tag)
            if target is not None:
                crew.sort(key=lambda w: cy_distance_to_squared(w.position, target.position), reverse=True)
            for worker in crew[:len(crew) - MAX_REPAIRERS]:
                _release(self, worker, bases)
            del crew[:len(crew) - MAX_REPAIRERS]
    return crews


def _wanted(target: Unit) -> int:
    if target.is_structure:
        wanted = 1 if target.is_flying else (4 if target.health_percentage < 0.5 else 2)
    else:
        wanted = 2 if target.health_percentage < 0.3 else 1
    return min(wanted, MAX_REPAIRERS)


def _targets(self: BotAI, bases: Units) -> List[Unit]:
    grid = self.mediator.get_cached_ground_grid
    targets: List[Unit] = [
        s for s in self.structures.ready
        if s.health_percentage <= STRUCTURE_BELOW and s.type_id not in _NOT_REPAIRED and _home_distance_sq(bases, s) <= HOME_RADIUS_SQ
        and (not s.is_flying or reachable_from_ground(grid, s.position, s.radius))
    ]
    targets.extend(
        u for u in self.units
        if u.is_mechanical and u.type_id not in _NOT_REPAIRED and u.health_percentage <= UNIT_BELOW
        and _home_distance_sq(bases, u) <= HOME_RADIUS_SQ and self.is_unit_position_safe(u)
        and (not u.is_flying or reachable_from_ground(grid, u.position, u.radius))
    )
    targets.sort(key=lambda t: t.health_percentage)
    return targets


def manage_repairs(self: BotAI) -> None:
    bases: Units = self.townhalls.not_flying
    if bases.empty:
        return
    repairing: List[Unit] = [w for w in self.workers if w.is_repairing]
    crews = _follow_jobs(self, bases, repairing)

    if self.worker_rushed and not self.army_advisor.is_wall_closed():
        return

    now = self.time
    if len(self.repair_walks) > 400:
        self.repair_walks = {k: v for k, v in self.repair_walks.items() if now - v[0] < WALK_CACHE_SECONDS}
    self.repair_backoff = {tag: until for tag, until in self.repair_backoff.items() if until > now}

    busy: Set[int] = {t for t in (self.build_order_critical_worker, self.scout_worker_tag) if t is not None}
    pool: List[Unit] = [
        w for w in self.workers
        if (w.is_gathering or w.is_idle) and w.tag not in busy and w.tag not in self.repair_jobs and _home_distance_sq(bases, w) <= LEASH_SQ
    ]

    for target in _targets(self, bases):
        missing = _wanted(target) - len(crews.get(target.tag, ()))
        if missing <= 0 or not pool or self.repair_backoff.get(target.tag, 0.0) > now:
            continue
        candidates = cy_sorted_by_distance_to(
            [w for w in pool if cy_distance_to_squared(w.position, target.position) <= MAX_TRAVEL_SQ],
            target.position,
        )
        sent = 0
        for worker in candidates[:missing + 3]:
            walk = ground_walk(self, worker, target)
            if walk is None or walk > MAX_TRAVEL:
                continue
            self.mediator.remove_worker_from_mineral(worker_tag=worker.tag)
            self.mediator.assign_role(tag=worker.tag, role=UnitRole.DEFENDING)
            worker(AbilityId.EFFECT_REPAIR_SCV, target)
            self.repair_jobs[worker.tag] = RepairJob(
                target=target.tag, is_structure=target.is_structure, since=now, last=worker.position_tuple
            )
            crews.setdefault(target.tag, []).append(worker)
            pool.remove(worker)
            sent += 1
            if sent >= missing:
                break
        if sent == 0:
            self.repair_backoff[target.tag] = now + NO_CREW_BACKOFF