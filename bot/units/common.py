import math
from typing import Iterable, Optional, Sequence

import numpy as np

from cython_extensions.combat_utils import cy_attack_ready
from cython_extensions.geometry import cy_distance_to, cy_distance_to_squared, cy_towards

from ares.behaviors.combat.individual import KeepUnitSafe, StutterUnitForward
from sc2.position import Point2
from sc2.unit import Unit

from bot.tactics.consts import BANELING_RETREAT_DISTANCE, KITE_IN_STOP_GAP, MELEE_RANGE_THRESHOLD
from bot.tactics.context import ArmyContext
from bot.tactics.orders import GroupOrders
from bot.utils.geometry import is_already_attack_moving_to, is_already_attacking, is_already_moving_to, segment_walkable


def run(ai, behavior) -> bool:
    return bool(behavior.execute(ai, ai.config, ai.mediator))


def attack_unit(unit: Unit, target: Unit) -> None:
    if not is_already_attacking(unit, target):
        unit.attack(target)


def attack_move(unit: Unit, point: Point2) -> None:
    if not is_already_attack_moving_to(unit, point):
        unit.attack(point)


def move_to(unit: Unit, point: Point2) -> None:
    if not is_already_moving_to(unit, point):
        unit.move(point)


def stutter_forward(ai, unit: Unit, target: Unit) -> bool:
    return run(ai, StutterUnitForward(unit=unit, target=target))


def futile_to_kite(unit: Unit, threat: Optional[Unit]) -> bool:
    if threat is None:
        return False
    return threat.ground_range > unit.ground_range and (
        threat.real_speed >= unit.real_speed or threat.real_speed < 0.5
    )


def target_harmless(target: Optional[Unit]) -> bool:
    return target is not None and target.ground_range == 0 and target.air_range == 0


def all_melee(enemies: Iterable[Unit]) -> bool:
    seen = False
    for e in enemies:
        seen = True
        if e.ground_range > MELEE_RANGE_THRESHOLD:
            return False
    return seen


def _safe_grid(ai, unit: Unit, base_grid: np.ndarray) -> np.ndarray:
    if unit.is_flying:
        avoid = ai.mediator.get_air_avoidance_grid
    else:
        avoid = ai.mediator.get_priority_ground_avoidance_grid
    return np.maximum(base_grid, avoid)


def kite_away(ai, ctx: ArmyContext, unit: Unit, grid=None) -> bool:
    if grid is None:
        grid = _safe_grid(ai, unit, ctx.grid_for(unit))
    return run(ai, KeepUnitSafe(unit=unit, grid=grid))


def retreat_point(ai, unit: Unit, threats: Sequence[Unit], fallback: Point2) -> Point2:
    cx = sum(t.position.x for t in threats) / len(threats)
    cy = sum(t.position.y for t in threats) / len(threats)
    threat_center = Point2((cx, cy))
    distance = cy_distance_to(unit.position, threat_center)

    if distance < 0.5:
        return fallback

    for fraction in (1.0, 0.5):
        reach = BANELING_RETREAT_DISTANCE * fraction
        point = Point2(cy_towards(threat_center, unit.position, reach))
        if ai.in_map_bounds(point) and ai.in_pathing_grid(point):
            return point

    return fallback


def step_back_from(
    ai, ctx: ArmyContext, unit: Unit, threats: Sequence[Unit], orders: GroupOrders, grid=None
) -> None:
    if orders.retreating:
        path_move(ai, ctx, unit, orders.hold_point, grid=grid, avoid_danger=True)
        return
    path_move(ai, ctx, unit, retreat_point(ai, unit, threats, orders.hold_point), grid=grid, avoid_danger=True)


def kite_from_banelings(
    ai, ctx: ArmyContext, unit: Unit, banelings: Sequence[Unit], orders: GroupOrders, grid=None
) -> None:
    step_back_from(ai, ctx, unit, banelings, orders, grid=grid)


def path_move(
    ai,
    ctx: ArmyContext,
    unit: Unit,
    target: Point2,
    grid=None,
    sense_danger: bool = True,
    avoid_danger: bool = False,
) -> None:
    if avoid_danger:
        sense_danger = False
    if grid is None:
        base = ctx.grid_for(unit)
        grid = _safe_grid(ai, unit, base) if avoid_danger else base
    next_point = ai.mediator.find_path_next_point(
        start=unit.position,
        target=target,
        grid=grid,
        sensitivity=2,
        sense_danger=sense_danger,
    )
    move_to(unit, next_point)


def follow_point(orders: GroupOrders, lead: float = 4.0, walkable=None) -> Point2:
    if orders.aggressive:
        anchor = orders.anchor
        if anchor is None or cy_distance_to_squared(anchor, orders.target) <= lead * lead:
            return orders.target

        point = Point2(cy_towards(anchor, orders.target, lead))
        if walkable is not None and not segment_walkable(anchor, point, walkable):
            return orders.target

        return point

    return orders.bio_position if orders.bio_position is not None else orders.hold_point