from typing import Dict, List, Set, Tuple

import numpy as np

from cython_extensions.combat_utils import cy_attack_ready, cy_pick_enemy_target
from cython_extensions.units_utils import cy_in_attack_range
from cython_extensions.units_utils import cy_center, cy_closer_than, cy_closest_to
from cython_extensions.geometry import cy_distance_to, cy_distance_to_squared, cy_towards

from ares.behaviors.combat.individual import StutterUnitBack
from sc2.ids.ability_id import AbilityId
from sc2.position import Point2
from sc2.unit import Unit
from sc2.units import Units

from bot.tactics.consts import ENEMY_WORKER_TYPES
from bot.tactics.context import ArmyContext
from bot.tactics.orders import GroupOrders
from bot.tactics.positioning import TOWNHALL_TYPES, Positioning
from bot.units.common import attack_unit, kite_away, path_move, run
from bot.units.repair_retreat import RepairRetreat
from bot.utils.geometry import open_ground_near, path_length

RETREAT_BELOW_HEALTH = 0.4
RESUME_ABOVE_HEALTH = 0.9
BASE_DANGER_RADIUS = 8
BASE_REEVALUATE_SECONDS = 15.0
BASE_TOO_DANGEROUS = 30.0
MAX_HARASS_BASES = 4
STANDOFF_MARGIN = 0.5
TARGET_PATIENCE = 6.0
WRITE_OFF_SECONDS = 25.0
WRITE_OFF_RADIUS = 8.0
WRITE_OFF_RADIUS_EXT = WRITE_OFF_RADIUS + 4.0
IDLE_PATIENCE = 6.0
ARRIVED_RADIUS = 10.0
ARRIVED_RADIUS_SQ = ARRIVED_RADIUS * ARRIVED_RADIUS
RECALL_SECONDS = 20.0
REPAIR_WAIT_RADIUS = 6.0
REPAIR_PATIENCE = 25.0
RETREAT_COOLDOWN = 60.0
REPAIR_LANE_DISTANCE = 4.5
REPAIR_SPOT_DISTANCE = 9.0
REPAIR_SPOT_CANDIDATES = 40
REPAIR_SPOT_REFRESH = 5.0
REPAIR_WALK_FACTOR = 2.5
REPAIR_WALK_SLACK = 4.0
MIN_CLOAKED_ENERGY = 3.0
CLOAK_WORTHWHILE_ENERGY = 50.0

CLOAK_ON = AbilityId.BEHAVIOR_CLOAKON_BANSHEE
CLOAK_OFF = AbilityId.BEHAVIOR_CLOAKOFF_BANSHEE


class BansheeHarass:
    def __init__(self, ai, positioning: Positioning):
        self.ai = ai
        self.positioning = positioning
        self.roam: Dict[int, Point2] = {}
        self._last_assignment: float = -1e9
        self._written_off: List[Tuple[Point2, float]] = []
        self._dull_bases: List[Tuple[Point2, float]] = []
        self._focus: Dict[int, Tuple[int, float]] = {}
        self._last_fired: Dict[int, float] = {}
        self._idle_since: Dict[int, float] = {}
        self._recalled_until: Dict[int, float] = {}
        self._repair_spots: Dict[int, Tuple[float, Point2]] = {}

        self.repair = RepairRetreat(
            self.ai, self._repair_spot, below=RETREAT_BELOW_HEALTH, resume=RESUME_ABOVE_HEALTH,
            patience=REPAIR_PATIENCE, cooldown=RETREAT_COOLDOWN, wait_radius=REPAIR_WAIT_RADIUS,
        )
        self.retreating: Set[int] = self.repair.retreating

    def control(self, units: Units, orders: GroupOrders, ctx: ArmyContext) -> None:
        alive = {u.tag for u in units}
        for table in (self.roam, self._focus, self._last_fired, self._idle_since, self._recalled_until):
            for tag in [t for t in list(table) if t not in alive]:
                del table[tag]

        self.repair.forget(alive)
        now = self.ai.time
        self._written_off = [(p, until) for p, until in self._written_off if until > now]
        self._dull_bases = [(p, until) for p, until in self._dull_bases if until > now]
        ctx.prefetch_near(units)
        self._assign_bases(units, ctx)

        for unit in units:
            self._control_unit(unit, orders, ctx)

    def _candidate_bases(self, ctx: ArmyContext) -> List[Point2]:
        bases = [loc for loc, _ in ctx.mediator.get_enemy_expansions][:MAX_HARASS_BASES]
        structures = self.ai.enemy_structures
        townhalls = structures.of_type(TOWNHALL_TYPES) if structures else structures

        def occupied(loc: Point2) -> bool:
            return bool(townhalls) and len(cy_closer_than(townhalls, 10, loc)) > 0

        order = {loc: i for i, loc in enumerate(bases)}
        return sorted(
            bases,
            key=lambda loc: (
                self._is_written_off(loc, WRITE_OFF_RADIUS_EXT) or self._is_dull(loc),
                not occupied(loc),
                self._danger_at(loc, ctx),
                order[loc],
            ),
        )

    def _is_written_off(self, position: Point2, radius: float = WRITE_OFF_RADIUS) -> bool:
        threshold_sq = radius * radius
        return any(cy_distance_to_squared(p, position) <= threshold_sq for p, _ in self._written_off)

    def _is_dull(self, base: Point2) -> bool:
        return any(cy_distance_to_squared(p, base) <= 1.0 for p, _ in self._dull_bases)

    @staticmethod
    def _danger_at(position: Point2, ctx: ArmyContext) -> float:
        grid = ctx.air_grid
        r = BASE_DANGER_RADIUS
        x, y = int(position.x), int(position.y)
        window = grid[max(0, x - r): x + r + 1, max(0, y - r): y + r + 1]
        finite = window[np.isfinite(window)]
        return float(finite.max()) if finite.size else 0.0

    def _assign_bases(self, units: Units, ctx: ArmyContext) -> None:
        now = self.ai.time
        unassigned = [u for u in units if u.tag not in self.roam]
        stale = now - self._last_assignment >= BASE_REEVALUATE_SECONDS

        if not unassigned and not stale:
            return

        candidates = self._candidate_bases(ctx)
        if not candidates:
            return

        self._last_assignment = now
        taken = {tag: loc for tag, loc in self.roam.items()}

        def worth_going(loc: Point2) -> bool:
            return (
                self._danger_at(loc, ctx) < BASE_TOO_DANGEROUS
                and not self._is_written_off(loc, WRITE_OFF_RADIUS_EXT)
                and not self._is_dull(loc)
            )

        for unit in units:
            current = self.roam.get(unit.tag)
            if current is not None and worth_going(current):
                continue

            used = {loc for tag, loc in taken.items() if tag != unit.tag}
            free = [loc for loc in candidates if loc not in used and worth_going(loc)]
            pick = free[0] if free else min(candidates, key=lambda loc: self._danger_at(loc, ctx))
            self.roam[unit.tag] = pick
            taken[unit.tag] = pick

    def _repair_spot(self, unit: Unit, orders: GroupOrders, ctx: ArmyContext) -> Point2:
        bases = self.ai.townhalls.not_flying
        if not bases:
            return orders.hold_point

        base = cy_closest_to(unit.position, bases)
        now = self.ai.time
        chosen = self._repair_spots.get(base.tag)

        if chosen is None or now - chosen[0] >= REPAIR_SPOT_REFRESH:
            chosen = self._repair_spots[base.tag] = (now, self._find_repair_spot(base, ctx))

        return chosen[1]

    def _find_repair_spot(self, base: Unit, ctx: ArmyContext) -> Point2:
        grid = ctx.mediator.get_cached_ground_grid
        centre = base.position
        minerals = cy_closer_than(self.ai.mineral_field, 10.0, centre)

        if minerals:
            lane = Point2(cy_center(minerals))
            centre = Point2(cy_towards(centre, lane, min(REPAIR_LANE_DISTANCE, cy_distance_to(centre, lane))))

        spots = open_ground_near(grid, centre, REPAIR_SPOT_DISTANCE, limit=REPAIR_SPOT_CANDIDATES)

        for spot in spots:
            path = ctx.mediator.find_raw_path(start=base.position, target=spot, grid=grid, sensitivity=1)
            walk = path_length(base.position, path)
            if walk is not None and walk <= REPAIR_WALK_FACTOR * cy_distance_to(base.position, spot) + REPAIR_WALK_SLACK:
                return spot

        return spots[0] if spots else base.position

    def _destination(self, unit: Unit, orders: GroupOrders, ctx: ArmyContext, now: float) -> Point2:
        tag = unit.tag
        if self._recalled_until.get(tag, 0.0) > now:
            return self._army_point(orders)

        destination = self.roam.get(tag)
        if destination is None:
            return orders.target

        if cy_distance_to_squared(unit.position, destination) > ARRIVED_RADIUS_SQ:
            self._idle_since.pop(tag, None)
            return destination

        if now - self._idle_since.setdefault(tag, now) < IDLE_PATIENCE:
            return destination

        self._idle_since.pop(tag, None)
        self._dull_bases.append((destination, now + WRITE_OFF_SECONDS))
        self.roam.pop(tag, None)
        self._assign_bases(Units([unit], self.ai), ctx)

        pick = self.roam.get(tag)
        if pick is None or self._is_dull(pick) or self._is_written_off(pick, WRITE_OFF_RADIUS_EXT):
            self.roam.pop(tag, None)
            self._recalled_until[tag] = now + RECALL_SECONDS
            return self._army_point(orders)

        return pick

    @staticmethod
    def _army_point(orders: GroupOrders) -> Point2:
        return orders.anchor if orders.anchor is not None else orders.hold_point

    def _control_unit(self, unit: Unit, orders: GroupOrders, ctx: ArmyContext) -> None:
        now = self.ai.time
        self.repair.update(unit, orders, ctx)

        danger = self._in_danger(unit, ctx)
        if danger:
            if not unit.is_cloaked and ctx.cloak_researched and CLOAK_ON in unit.abilities and self._should_cloak(unit):
                unit(CLOAK_ON)
            if kite_away(self.ai, ctx, unit):
                return

        if unit.tag in self.retreating:
            self._idle_since.pop(unit.tag, None)
            path_move(self.ai, ctx, unit, self._repair_spot(unit, orders, ctx), grid=ctx.air_grid)
            return

        if unit.is_cloaked and not danger and CLOAK_OFF in unit.abilities and not ctx.enemies_near(unit):
            unit(CLOAK_OFF)

        if unit.weapon_cooldown > 0:
            self._last_fired[unit.tag] = now

        targets = [
            t for t in ctx.targets_near(unit)
            if not t.is_structure and not self._is_written_off(t.position) and self._can_hit_safely(unit, t, ctx)
        ]

        if unit.tag in self.retreating or not targets:
            self._focus.pop(unit.tag, None)

        if targets:
            self._idle_since.pop(unit.tag, None)
            workers = [t for t in targets if t.type_id in ENEMY_WORKER_TYPES]
            pool = workers if workers else targets
            in_range = cy_in_attack_range(unit, pool)
            target: Unit = cy_pick_enemy_target(in_range) if in_range else cy_closest_to(unit.position, pool)

            self._check_patience(unit, target, now)

            if cy_attack_ready(self.ai, unit, target):
                attack_unit(unit, target)
                return

            if run(self.ai, StutterUnitBack(unit=unit, target=target, kite_via_pathing=True, grid=ctx.air_grid)):
                return

            attack_unit(unit, target)
            return

        path_move(self.ai, ctx, unit, self._destination(unit, orders, ctx, now), grid=ctx.air_grid)

    @staticmethod
    def _should_cloak(unit: Unit) -> bool:
        return unit.energy >= CLOAK_WORTHWHILE_ENERGY or unit.health_percentage <= RETREAT_BELOW_HEALTH

    def _can_hit_safely(self, unit: Unit, target: Unit, ctx: ArmyContext) -> bool:
        if unit.is_cloaked and unit.energy > MIN_CLOAKED_ENERGY:
            return True
        if ctx.cloak_researched and CLOAK_ON in unit.abilities and self._should_cloak(unit):
            return True

        reach = max(1.0, unit.ground_range - STANDOFF_MARGIN)
        reach_sq = reach * reach
        standoff = (
            Point2(cy_towards(target.position, unit.position, reach))
            if cy_distance_to_squared(unit.position, target.position) > reach_sq
            else unit.position
        )
        return ctx.mediator.is_position_safe(grid=ctx.air_grid, position=standoff)

    def _check_patience(self, unit: Unit, target: Unit, now: float) -> None:
        focus = self._focus.get(unit.tag)
        if focus is None or focus[0] != target.tag:
            self._focus[unit.tag] = (target.tag, now)
            return

        fired_since = self._last_fired.get(unit.tag, -1e9) >= focus[1]
        if not fired_since and now - focus[1] > TARGET_PATIENCE:
            self._written_off.append((target.position, now + WRITE_OFF_SECONDS))
            self._focus.pop(unit.tag, None)

    @staticmethod
    def _in_danger(unit: Unit, ctx: ArmyContext) -> bool:
        if ctx.is_safe(unit):
            return False
        if unit.is_cloaked and unit.energy > MIN_CLOAKED_ENERGY:
            return ctx.mediator.get_is_detected(unit=unit, by_enemy=True)
        return True