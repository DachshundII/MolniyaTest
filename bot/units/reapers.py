import itertools

from cython_extensions.combat_utils import cy_attack_ready, cy_pick_enemy_target
from cython_extensions.units_utils import cy_in_attack_range
from cython_extensions.units_utils import cy_closer_than, cy_closest_to
from cython_extensions.geometry import cy_distance_to_squared, cy_towards

from ares.behaviors.combat.individual import ReaperGrenade
from sc2.data import Race
from sc2.ids.upgrade_id import UpgradeId
from sc2.position import Point2
from sc2.unit import Unit
from sc2.units import Units

from bot.tactics.consts import ATTACK_TARGET_IGNORE
from bot.tactics.context import ArmyContext
from bot.tactics.orders import GroupOrders
from bot.tactics.positioning import TOWNHALL_TYPES
from bot.units.common import (
    attack_unit,
    futile_to_kite,
    kite_away,
    kite_from_banelings,
    path_move,
    run,
)
from bot.units.danger_memory import DangerMemory, ground_defenders

RETREAT_BELOW_HEALTH = 0.5
RESUME_ABOVE_HEALTH = 0.85
HIDDEN_BASE_SCOUT_INTERVAL = 30.0
MINERAL_LINE_RADIUS = 10.0
TOUR_ARRIVED = 3.5
TOUR_ARRIVED_SQ = TOUR_ARRIVED * TOUR_ARRIVED
TOUR_REFRESH = 3.0
TOUR_LANE = 2.0
DESTINATION_ARRIVAL_SQ = 25.0
GRENADE_HOLD_SECONDS = 2.0


class ReaperHarass:
    def __init__(self, ai):
        self.ai = ai
        self.retreating: set[int] = set()
        self._hunt_index: int = 0
        self._hunt_since: float = 0.0
        self._stop: dict[int, int] = {}
        self._tour_cache: tuple[float, list[Point2]] = (-1e9, [])
        self.danger = DangerMemory(ai)
        self._grenade_until: dict[int, float] = {}

    def control(self, units: Units, orders: GroupOrders, ctx: ArmyContext) -> None:
        alive = {u.tag for u in units}
        self.retreating &= alive
        self._stop = {tag: stop for tag, stop in self._stop.items() if tag in alive}
        self._grenade_until = {tag: until for tag, until in self._grenade_until.items() if tag in alive}

        ctx.prefetch_near(units)
        self.danger.refresh(ground_defenders(self.ai))
        grid = ctx.climber_grid

        for unit in units:
            self._control_unit(unit, orders, ctx, grid)

    def _control_unit(self, unit: Unit, orders: GroupOrders, ctx: ArmyContext, grid) -> None:
        health = unit.health_percentage

        if unit.tag in self.retreating:
            if health >= RESUME_ABOVE_HEALTH:
                self.retreating.discard(unit.tag)
        elif health < RETREAT_BELOW_HEALTH:
            self.retreating.add(unit.tag)

        zergling_speed = (
            self.ai.enemy_race == Race.Zerg
            and self.ai.already_pending_upgrade(UpgradeId.ZERGLINGMOVEMENTSPEED) == 1
        )

        if unit.tag in self.retreating or zergling_speed:
            path_move(self.ai, ctx, unit, orders.hold_point, grid=grid)
            return

        targets = [t for t in ctx.targets_near(unit) if not t.is_structure]

        if targets and run(self.ai, ReaperGrenade(unit=unit, enemy_units=targets, retreat_target=orders.hold_point,
                                                  grid=grid)):
            self._grenade_until[unit.tag] = self.ai.time + GRENADE_HOLD_SECONDS
            return

        banelings = ctx.banelings_near(unit)
        close_banelings = ctx.close_banelings(unit)

        if targets and self._grenade_until.get(unit.tag, 0.0) > self.ai.time:
            if close_banelings:
                kite_from_banelings(self.ai, ctx, unit, close_banelings, orders, grid=grid)
                return
            in_range = cy_in_attack_range(unit, targets)
            if in_range:
                held_target = cy_pick_enemy_target(in_range)
                if cy_attack_ready(self.ai, unit, held_target):
                    attack_unit(unit, held_target)
                    return
            kite_away(self.ai, ctx, unit, grid)
            return

        target: Unit | None = None
        if targets:
            in_range = cy_in_attack_range(unit, targets)
            target = cy_pick_enemy_target(in_range) if in_range else cy_closest_to(unit.position, targets)
            if cy_attack_ready(self.ai, unit, target):
                attack_unit(unit, target)
                return

        if close_banelings:
            kite_from_banelings(self.ai, ctx, unit, close_banelings, orders, grid=grid)
            return

        if not (futile_to_kite(unit, target) and not banelings) and not ctx.is_safe(unit, grid):
            if kite_away(self.ai, ctx, unit, grid):
                return

        destination = self._destination(unit, ctx, grid)
        if target is not None and cy_distance_to_squared(unit.position, destination) <= DESTINATION_ARRIVAL_SQ:
            attack_unit(unit, target)
        else:
            path_move(self.ai, ctx, unit, destination, grid=self.danger.avoiding_grid(grid), sense_danger=bool(targets))

    def _destination(self, unit: Unit, ctx: ArmyContext, grid) -> Point2:
        ai = self.ai
        enemies = [
            e for e in ai.enemy_units
            if not e.is_memory and not e.is_flying and not e.is_cloaked and not e.is_hallucination
            and e.type_id not in ATTACK_TARGET_IGNORE
        ]

        if enemies:
            return min(enemies, key=lambda e: cy_distance_to_squared(e.position, ai.start_location)).position

        stops = self._tour(ctx, grid)
        stop = self._stop.setdefault(unit.tag, len(self._stop)) % len(stops)

        if cy_distance_to_squared(unit.position, stops[stop]) <= TOUR_ARRIVED_SQ:
            stop = (stop + 1) % len(stops)

        self._stop[unit.tag] = stop
        return stops[stop]

    def _tour(self, ctx: ArmyContext, grid) -> list[Point2]:
        ai = self.ai
        now = ai.time

        if now - self._tour_cache[0] < TOUR_REFRESH:
            return self._tour_cache[1]

        stops: list[Point2] = []
        bases = sorted(
            (s for s in ai.enemy_structures if s.type_id in TOWNHALL_TYPES and not s.is_flying),
            key=lambda s: cy_distance_to_squared(s.position, ai.start_location),
        )
        defenders = self._defenders() + self.danger.spots()

        for base in bases:
            stops.extend(stop for stop in self._mineral_line(base.position) if not self._defended(stop, defenders))

        if not stops:
            if ai.enemy_structures:
                stops.append(cy_closest_to(ai.start_location, ai.enemy_structures).position)
            stops.append(self._search_point())

        if len(stops) == 1:
            stops.append(Point2(cy_towards(stops[0], ai.start_location, 6.0)))

        self._tour_cache = (now, stops)
        return stops

    def _defenders(self) -> list[tuple[Point2, float]]:
        return ground_defenders(self.ai)

    @staticmethod
    def _defended(stop: Point2, defenders: list[tuple[Point2, float]]) -> bool:
        return any(
            cy_distance_to_squared(stop, position) <= reach * reach
            for position, reach in defenders
        )

    def _mineral_line(self, base: Point2) -> list[Point2]:
        minerals = cy_closer_than(self.ai.mineral_field, MINERAL_LINE_RADIUS, base)

        if len(minerals) < 2:
            return [base, Point2(cy_towards(base, self.ai.game_info.map_center, 7.0))]

        end_a, end_b = max(
            itertools.combinations([m.position for m in minerals], 2),
            key=lambda pair: cy_distance_to_squared(pair[0], pair[1]),
        )
        return [Point2(cy_towards(end_a, base, TOUR_LANE)), Point2(cy_towards(end_b, base, TOUR_LANE))]

    def _search_point(self) -> Point2:
        ai = self.ai
        if not ai.enemy_base_scouted:
            candidate = self._next_hunt_candidate()
            if candidate is not None:
                return candidate
        return ai.enemy_start_locations[0]

    def _next_hunt_candidate(self) -> Point2 | None:
        expansions = self.ai.expansion_locations_list
        if not expansions:
            return None

        if self.ai.time - self._hunt_since > HIDDEN_BASE_SCOUT_INTERVAL:
            self._hunt_since = self.ai.time
            self._hunt_index = (self._hunt_index + 1) % len(expansions)

        return expansions[self._hunt_index]