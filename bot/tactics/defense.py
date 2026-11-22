from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

from cython_extensions.units_utils import cy_center, cy_sorted_by_distance_to
from cython_extensions.geometry import cy_distance_to_squared

from ares.consts import UnitRole, UnitTreeQueryType
from sc2.ids.unit_typeid import UnitTypeId as U
from sc2.position import Point2
from sc2.unit import Unit
from sc2.units import Units

from bot.tactics.consts import (
    BASE_THREAT_RADIUS,
    DETACHMENT_RESULT,
    ENEMY_NON_ARMY_TYPES,
    THREAT_CLUSTER_RADIUS,
)
from bot.tactics.context import ArmyContext
from bot.tactics.fight import FightEvaluator, Stance
from bot.tactics.orders import GroupOrders, Mode

DEFENDER_TYPES = frozenset({
    U.MARINE, U.MARAUDER, U.CYCLONE, U.HELLION, U.HELLIONTANK,
    U.THOR, U.THORAP, U.GHOST, U.VIKINGFIGHTER, U.BATTLECRUISER,
})

RESPONSE_RADIUS = 70.0
RESPONSE_RADIUS_SQ = RESPONSE_RADIUS * RESPONSE_RADIUS
ESCALATE_FRACTION = 0.6
RELEASE_DELAY = 5.0
MIN_DETACHMENT = 3
TASK_MATCH_DISTANCE = 20.0
TASK_MATCH_DISTANCE_SQ = TASK_MATCH_DISTANCE * TASK_MATCH_DISTANCE
HOLD_SAFE_SEARCH_RADIUS = 12


@dataclass
class Threat:
    units: List[Unit]
    center: Point2


@dataclass
class DefenseTask:
    center: Point2
    threat_tags: Set[int]
    defender_tags: Set[int] = field(default_factory=set)
    last_threat_time: float = 0.0
    escalated: bool = False
    has_ground: bool = True


def can_hit(unit: Unit, enemy: Unit) -> bool:
    return unit.can_attack_air if enemy.is_flying else unit.can_attack_ground


def cluster_units(units: List[Unit], radius: float) -> List[List[Unit]]:
    radius_sq = radius * radius
    clusters: List[List[Unit]] = []
    for u in units:
        placed = False
        for cluster in clusters:
            if any(cy_distance_to_squared(u.position, m.position) <= radius_sq for m in cluster):
                cluster.append(u)
                placed = True
                break
        if not placed:
            clusters.append([u])

    merged = True
    while merged:
        merged = False
        for i in range(len(clusters)):
            for j in range(i + 1, len(clusters)):
                if any(
                    cy_distance_to_squared(a.position, b.position) <= radius_sq
                    for a in clusters[i]
                    for b in clusters[j]
                ):
                    clusters[i].extend(clusters[j])
                    del clusters[j]
                    merged = True
                    break
            if merged:
                break
    return clusters


class BaseDefense:
    def __init__(self, ai, fight: FightEvaluator, positioning):
        self.ai = ai
        self.fight = fight
        self.positioning = positioning
        self.tasks: List[DefenseTask] = []

    def _hold_target(self, position: Point2, has_ground: bool) -> Point2:
        target = self.positioning.hold_at_ramp(position) if has_ground else position
        if has_ground:
            base = self.ai.mediator.get_ground_grid
            avoidance = self.ai.mediator.get_priority_ground_avoidance_grid
        else:
            base = self.ai.mediator.get_air_grid
            avoidance = self.ai.mediator.get_air_avoidance_grid
        safe_grid = np.maximum(base, avoidance)
        if self.ai.mediator.is_position_safe(grid=safe_grid, position=target):
            return target
        return self.ai.mediator.find_closest_safe_spot(
            from_pos=target, grid=safe_grid, radius=HOLD_SAFE_SEARCH_RADIUS,
        )

    def find_threats(self, ctx: ArmyContext) -> List[Threat]:
        structures = [s for s in self.ai.structures if s.type_id != U.AUTOTURRET]
        if not structures:
            return []

        near_lists = ctx.mediator.get_units_in_range(
            start_points=structures,
            distances=BASE_THREAT_RADIUS,
            query_tree=UnitTreeQueryType.AllEnemy,
        )

        seen: Dict[int, Unit] = {}
        for units in near_lists:
            for e in units:
                if e.tag in seen:
                    continue
                if e.is_memory or e.is_structure or e.is_hallucination or e.type_id in ENEMY_NON_ARMY_TYPES:
                    continue
                seen[e.tag] = e

        return [
            Threat(units=cluster, center=Point2(cy_center(cluster)))
            for cluster in cluster_units(list(seen.values()), THREAT_CLUSTER_RADIUS)
        ]

    def update(
        self, ctx: ArmyContext, main_units: Units, defender_units: Units
    ) -> Tuple[List[Tuple[Units, GroupOrders]], Optional[Point2]]:
        now = self.ai.time
        threats = self.find_threats(ctx)
        mobile_pool = [u for u in main_units if u.type_id in DEFENDER_TYPES]
        alive_defenders: Dict[int, Unit] = {u.tag: u for u in defender_units}

        active: List[Tuple[DefenseTask, Threat]] = []
        unmatched = list(self.tasks)

        for threat in threats:
            task = None
            for candidate in unmatched:
                if cy_distance_to_squared(candidate.center, threat.center) <= TASK_MATCH_DISTANCE_SQ:
                    task = candidate
                    break

            if task is None:
                task = DefenseTask(center=threat.center, threat_tags={u.tag for u in threat.units})
                self.tasks.append(task)
            else:
                unmatched.remove(task)

            task.center = threat.center
            task.threat_tags = {u.tag for u in threat.units}
            task.has_ground = any(not u.is_flying for u in threat.units)
            task.last_threat_time = now
            active.append((task, threat))

        escalate_to: Optional[Point2] = None
        groups: List[Tuple[Units, GroupOrders]] = []
        claimed: Set[int] = set()
        total_fighters = len(mobile_pool) + len(defender_units)

        for task, threat in active:
            current = [alive_defenders[t] for t in task.defender_tags if t in alive_defenders]
            candidates = cy_sorted_by_distance_to(
                [
                    u for u in mobile_pool
                    if u.tag not in claimed
                    and u.tag not in task.defender_tags
                    and cy_distance_to_squared(u.position, threat.center) <= RESPONSE_RADIUS_SQ
                    and any(can_hit(u, e) for e in threat.units)
                ],
                threat.center,
            )
            ordered: List[Unit] = current + candidates
            eligible_total = len(ordered)

            if eligible_total == 0:
                task.escalated = True
                if escalate_to is None:
                    escalate_to = self._hold_target(threat.center, task.has_ground)
                continue

            subset, result = self.fight.smallest_winning_subset(
                ordered,
                threat.units,
                target=DETACHMENT_RESULT,
                min_size=min(MIN_DETACHMENT, eligible_total),
                stance=Stance.MEETING,
            )

            too_big = subset is not None and len(subset) > ESCALATE_FRACTION * total_fighters

            if subset is None or too_big:
                task.escalated = True
                if escalate_to is None:
                    escalate_to = self._hold_target(threat.center, task.has_ground)
                chosen = current
            else:
                task.escalated = False
                chosen = list(subset) if len(subset) >= len(current) else current
                new_tags = {u.tag for u in chosen if u.tag not in task.defender_tags}
                if new_tags:
                    ctx.mediator.batch_assign_role(tags=new_tags, role=UnitRole.BASE_DEFENDER)
                    task.defender_tags.update(new_tags)

            claimed.update(u.tag for u in chosen)

            if chosen:
                units = Units(chosen, self.ai)
                point = self._hold_target(threat.center, task.has_ground)
                groups.append((units, GroupOrders(
                    label="defense", mode=Mode.DEFEND, target=point, hold_point=ctx.hold_point,
                    front=ctx.front, bio_position=ctx.bio_position, anchor=point,
                )))

        for task in list(self.tasks):
            if any(task is t for t, _ in active):
                continue
            if now - task.last_threat_time >= RELEASE_DELAY:
                if task.defender_tags:
                    ctx.mediator.batch_assign_role(tags=task.defender_tags, role=UnitRole.ATTACKING)
                self.tasks.remove(task)
            else:
                current = [alive_defenders[t] for t in task.defender_tags if t in alive_defenders]
                if current:
                    point = self._hold_target(task.center, task.has_ground)
                    groups.append((Units(current, self.ai), GroupOrders(
                        label="defense", mode=Mode.DEFEND, target=point, hold_point=ctx.hold_point,
                        front=ctx.front, bio_position=ctx.bio_position, anchor=point, local_result=None,
                    )))

        for task in list(self.tasks):
            task.defender_tags &= set(alive_defenders) | {u.tag for u in main_units}

        return groups, escalate_to