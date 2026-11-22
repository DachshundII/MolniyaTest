import math
from typing import Dict, List, Optional, Set, Tuple

from cython_extensions.combat_utils import cy_attack_ready
from cython_extensions.units_utils import cy_closest_to, cy_in_attack_range
from cython_extensions.geometry import cy_distance_to, cy_distance_to_squared, cy_towards

from sc2.ids.ability_id import AbilityId
from sc2.ids.buff_id import BuffId
from sc2.position import Point2
from sc2.unit import Unit
from sc2.units import Units

from bot.tactics.consts import ATTACK_TARGET_IGNORE_WITH_WORKERS, KITE_IN_RESULT, LOCAL_FIGHT_RADIUS, SKYTOSS_TYPES
from bot.tactics.context import ArmyContext
from bot.tactics.orders import GroupOrders
from bot.units.common import (
    _safe_grid as safe_grid_for,
    all_melee,
    attack_move,
    attack_unit,
    follow_point,
    futile_to_kite,
    kite_away,
    kite_from_banelings,
    move_to,
    path_move,
    stutter_forward,
)
from bot.units.repair_retreat import RepairRetreat
from bot.utils.geometry import plain_point, segment_walkable, terrain_view

LOCAL_FIGHT_RADIUS_SQ = LOCAL_FIGHT_RADIUS * LOCAL_FIGHT_RADIUS

LOCK_ON_TIMEOUT = 15.0
LOCK_ON_ABILITIES = {AbilityId.LOCKON_LOCKON, AbilityId.LOCKONAIR_LOCKONAIR}
LOCK_ON_HOLD_RANGE = 15.0
LOCK_ON_MARGIN = 1.5
LOCK_ON_STEP = 4.0
LOCK_ON_MIN_STEP = 1.0
MIN_LOCK_STANDOFF = 6.0
LOCK_ON_CAST_SECONDS = 0.3
LOCK_ON_CONFIRM_SECONDS = 1.5
VIEW_RING_FRACTIONS = (0.3, 0.5, 0.7, 0.85, 1.0)
VIEW_DIRECTIONS = 12
LOCK_DETOUR = 3.0
CYCLONE_SIGHT = 11.0
OTHER_SIGHT = 9.0
SIGHT_MARGIN = 0.5
CLOSER_TOLERANCE = 0.5


class CycloneController:
    def __init__(self, ai):
        self.ai = ai
        self.lock_ons: Dict[int, float] = {}
        self.locks: Dict[int, Tuple[int, float]] = {}
        self.lockon_range: float = ai.game_data.abilities[AbilityId.LOCKON_LOCKON.value]._proto.cast_range
        self._watchers: Dict[int, Set[int]] = {}
        self.repair = RepairRetreat(ai, lambda unit, orders, ctx: orders.hold_point)

    def begin_step(self, units: Units, ctx: ArmyContext) -> None:
        now = self.ai.time
        for tag in [t for t, cast_at in self.lock_ons.items() if now - cast_at > LOCK_ON_TIMEOUT]:
            del self.lock_ons[tag]

        alive = {u.tag for u in self.ai.units}
        for tag in [t for t in self.locks if t not in alive]:
            del self.locks[tag]

        self.repair.forget(alive)
        self._watchers = {}
        ctx.prefetch_near(units)

    def control(self, units: Units, orders: GroupOrders, ctx: ArmyContext) -> None:
        self.begin_step(units, ctx)
        for unit in units:
            self._control_unit(unit, orders, ctx)

    def _control_unit(self, unit: Unit, orders: GroupOrders, ctx: ArmyContext) -> None:
        if self.handle_running_lock(unit, orders, ctx) or self.repair_trip(unit, orders, ctx):
            return

        close_banelings = ctx.close_banelings(unit)
        if close_banelings:
            kite_from_banelings(self.ai, ctx, unit, close_banelings, orders)
            return

        targets = [e for e in ctx.targets_near(unit) if e.can_be_attacked]
        if self._try_lock_on(unit, targets):
            return

        if not targets:
            self._no_fight(unit, orders, ctx)
            return

        in_range = cy_in_attack_range(unit, targets)
        if not orders.aggressive and in_range:
            attack_unit(unit, cy_closest_to(unit.position, in_range))
            return

        nearest: Unit = cy_closest_to(unit.position, targets)
        if cy_attack_ready(self.ai, unit, nearest):
            attack_unit(unit, nearest)
            return

        banelings = ctx.banelings_near(unit)
        advance = orders.advance_result(unit)
        winning = (
            advance is not None
            and advance >= KITE_IN_RESULT
            and not banelings
            and not all_melee([
                e for e in ctx.enemies_near(unit)
                if not e.is_memory
                and cy_distance_to_squared(e.position, unit.position) <= LOCAL_FIGHT_RADIUS_SQ
            ])
        )

        if not (futile_to_kite(unit, nearest) and not banelings) and not winning and not ctx.is_safe(unit):
            if orders.retreating:
                path_move(self.ai, ctx, unit, orders.hold_point, avoid_danger=True)
                return
            if kite_away(self.ai, ctx, unit):
                return

        if winning and stutter_forward(self.ai, unit, nearest):
            return
        attack_unit(unit, nearest)

    def repair_trip(self, unit: Unit, orders: GroupOrders, ctx: ArmyContext) -> bool:
        home = self.repair.update(unit, orders, ctx)
        if home is None:
            return False
        wait_radius = self.repair.wait_radius
        if cy_distance_to_squared(unit.position, home) > wait_radius * wait_radius:
            path_move(self.ai, ctx, unit, home, avoid_danger=True)
        return True

    def handle_running_lock(self, unit: Unit, orders: GroupOrders, ctx: ArmyContext) -> bool:
        locked = self._locked_target(unit)
        if locked is not None:
            if not any(unit.is_using_ability(a) for a in LOCK_ON_ABILITIES) and self.ai.time - self.locks[unit.tag][1] >= LOCK_ON_CAST_SECONDS:
                self._kite_locked(unit, locked, orders, ctx)
            return True
        return any(unit.is_using_ability(a) for a in LOCK_ON_ABILITIES)

    def _try_lock_on(self, unit: Unit, targets: List[Unit]) -> bool:
        if not unit.abilities:
            return False

        unit_radius = unit.radius
        unit_pos = unit.position
        lockon_range = self.lockon_range
        candidates = [
            e for e in targets
            if e.type_id not in ATTACK_TARGET_IGNORE_WITH_WORKERS
            and e.tag not in self.lock_ons
            and cy_distance_to(unit_pos, e.position) <= unit_radius + e.radius + lockon_range
        ]

        if not candidates:
            return False

        return self.cast_lock_on(unit, self.pick_lockon_target(candidates))

    def cast_lock_on(self, unit: Unit, target: Unit) -> bool:
        ability = AbilityId.LOCKONAIR_LOCKONAIR if target.is_flying else AbilityId.LOCKON_LOCKON
        if ability not in unit.abilities:
            return False

        unit(ability, target)
        self.lock_ons[target.tag] = self.ai.time
        self.locks[unit.tag] = (target.tag, self.ai.time)
        return True

    @staticmethod
    def _gap(unit: Unit, target: Unit) -> float:
        return cy_distance_to(unit.position, target.position) - unit.radius - target.radius

    def _lock_shows(self, unit: Unit, target: Unit) -> bool:
        ability = AbilityId.LOCKONAIR_LOCKONAIR if target.is_flying else AbilityId.LOCKON_LOCKON
        return any(unit.is_using_ability(a) for a in LOCK_ON_ABILITIES) or target.has_buff(BuffId.LOCKON) or ability not in unit.abilities

    def _locked_target(self, unit: Unit) -> Optional[Unit]:
        record = self.locks.get(unit.tag)
        if record is None:
            return None

        tag, since = record
        now = self.ai.time
        target = self.ai.enemy_units.find_by_tag(tag)

        if (
            now - since > LOCK_ON_TIMEOUT
            or target is None or target.is_memory
            or self._gap(unit, target) > LOCK_ON_HOLD_RANGE
            or (now - since > LOCK_ON_CONFIRM_SECONDS and not self._lock_shows(unit, target))
        ):
            del self.locks[unit.tag]
            if target is not None and now - since <= LOCK_ON_CONFIRM_SECONDS + 1.0:
                self.lock_ons.pop(tag, None)
            return None

        return target

    def _kite_locked(self, unit: Unit, target: Unit, orders: GroupOrders, ctx: ArmyContext) -> None:
        hold = LOCK_ON_HOLD_RANGE - LOCK_ON_MARGIN
        in_pathing_grid = self.ai.in_pathing_grid

        if ctx.is_safe(unit):
            grid = ctx.grid_for(unit)
            gap = self._gap(unit, target)
            if gap > hold:
                step = Point2(cy_towards(unit.position, target.position, min(gap - hold + 2.0, LOCK_ON_STEP)))
                if ctx.mediator.is_position_safe(grid=grid, position=step):
                    if segment_walkable(unit.position, step, in_pathing_grid):
                        move_to(unit, step)
                    else:
                        path_move(self.ai, ctx, unit, target.position, grid=grid)
            elif gap < MIN_LOCK_STANDOFF:
                step = Point2(cy_towards(target.position, unit.position, MIN_LOCK_STANDOFF + unit.radius + target.radius))
                if ctx.mediator.is_position_safe(grid=grid, position=step):
                    if segment_walkable(unit.position, step, in_pathing_grid):
                        move_to(unit, step)
                    else:
                        kite_away(self.ai, ctx, unit)
            return

        grid = safe_grid_for(self.ai, unit, ctx.grid_for(unit))
        limit = hold + unit.radius + target.radius

        if orders.retreating:
            want = plain_point(self.ai.mediator.find_path_next_point(
                start=unit.position, target=orders.hold_point, grid=grid, sensitivity=2, sense_danger=False
            ))
            step = farthest_within(unit.position, want, target.position, limit)
        else:
            want = plain_point(ctx.mediator.find_closest_safe_spot(from_pos=unit.position, grid=grid, radius=11))
            step = self._way_out_keeping_the_lock(unit, target, want, farthest_within(unit.position, want, target.position, limit), ctx, grid)

        if step is not None:
            move_to(unit, step)
        elif orders.retreating:
            path_move(self.ai, ctx, unit, orders.hold_point, avoid_danger=True)
        else:
            kite_away(self.ai, ctx, unit)

    def _way_out_keeping_the_lock(self, unit: Unit, target: Unit, want: Point2, cut: Optional[Point2], ctx: ArmyContext, grid) -> Optional[Point2]:
        safe_cut = cut is not None and (
            cy_distance_to_squared(cut, want) < 0.0001
            or ctx.mediator.is_position_safe(grid=grid, position=cut)
        )

        if safe_cut and self._keeps_view(unit, cut, target):
            return cut

        spot = self._spot_with_view(unit, target, ctx, grid, max_reach=cy_distance_to(unit.position, want) + LOCK_DETOUR)
        if spot is not None:
            return spot

        return cut if safe_cut else None

    def _sees(self, spot: Point2, target: Unit) -> bool:
        return target.is_flying or terrain_view(self.ai.game_info.terrain_height.data_numpy, spot, target.position)

    @staticmethod
    def _sight(unit: Unit) -> float:
        return (unit.sight_range or CYCLONE_SIGHT) - SIGHT_MARGIN

    def _keeps_view(self, unit: Unit, spot: Point2, target: Unit) -> bool:
        sight = self._sight(unit)
        if cy_distance_to_squared(spot, target.position) <= sight * sight and self._sees(spot, target):
            return True

        watchers = self._watchers.get(target.tag)
        if watchers is None:
            watchers = self._watchers[target.tag] = {
                other.tag for other in self.ai.all_own_units
                if cy_distance_to(other.position, target.position) <= (other.sight_range or OTHER_SIGHT)
                and (other.is_flying or self._sees(other.position, target))
            }

        return bool(watchers - {unit.tag})

    def _spot_with_view(self, unit: Unit, target: Unit, ctx: ArmyContext, grid, max_reach: float = math.inf) -> Optional[Point2]:
        ai = self.ai
        sight = self._sight(unit)
        limit = LOCK_ON_HOLD_RANGE - LOCK_ON_MARGIN + unit.radius + target.radius
        here = cy_distance_to(unit.position, target.position)
        target_pos = target.position
        unit_pos = unit.position
        is_position_safe = ctx.mediator.is_position_safe
        in_map_bounds = ai.in_map_bounds
        in_pathing_grid = ai.in_pathing_grid

        for reach in (sight * f for f in VIEW_RING_FRACTIONS):
            if reach > max_reach:
                break

            for k in range(VIEW_DIRECTIONS):
                angle = 2.0 * math.pi * k / VIEW_DIRECTIONS
                spot = Point2((unit_pos.x + reach * math.cos(angle), unit_pos.y + reach * math.sin(angle)))
                gap = cy_distance_to(spot, target_pos)

                if gap > min(limit, sight) or gap < here - CLOSER_TOLERANCE or not (in_map_bounds(spot) and in_pathing_grid(spot)):
                    continue

                if not is_position_safe(grid=grid, position=spot) or not self._sees(spot, target):
                    continue

                if not segment_walkable(unit_pos, spot, in_pathing_grid):
                    continue

                return spot

        return None

    def _no_fight(self, unit: Unit, orders: GroupOrders, ctx: ArmyContext) -> None:
        if not ctx.is_safe(unit):
            if orders.retreating:
                path_move(self.ai, ctx, unit, orders.hold_point, avoid_danger=True)
                return
            if kite_away(self.ai, ctx, unit):
                return

        point = follow_point(orders, walkable=self.ai.in_pathing_grid)
        if orders.aggressive:
            if segment_walkable(unit.position, point, self.ai.in_pathing_grid):
                attack_move(unit, point)
            else:
                path_move(self.ai, ctx, unit, point)
        elif cy_distance_to(unit.position, point) > orders.hold_radius:
            path_move(self.ai, ctx, unit, point, avoid_danger=True)

    @staticmethod
    def pick_lockon_target(enemies: List[Unit]) -> Unit:
        priority = [e for e in enemies if e.type_id in SKYTOSS_TYPES]
        pool = priority if priority else enemies
        return min(pool, key=lambda e: (e.health + e.shield, e.tag))


def farthest_within(start: Point2, want: Point2, center: Point2, limit: float) -> Optional[Point2]:
    dx, dy = want.x - start.x, want.y - start.y
    length = math.hypot(dx, dy)

    if length < 0.1:
        return None

    fx, fy = start.x - center.x, start.y - center.y

    if fx * fx + fy * fy > limit * limit:
        return want if math.hypot(want.x - center.x, want.y - center.y) < math.hypot(fx, fy) else None

    a, b, c = length * length, 2 * (fx * dx + fy * dy), fx * fx + fy * fy - limit * limit
    t = min(1.0, (-b + math.sqrt(max(0.0, b * b - 4 * a * c))) / (2 * a))

    if t * length < LOCK_ON_MIN_STEP:
        return None

    return Point2((start.x + dx * t, start.y + dy * t))