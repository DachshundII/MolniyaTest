from cython_extensions.units_utils import cy_closer_than
from cython_extensions.geometry import cy_distance_to, cy_distance_to_squared

from sc2.ids.ability_id import AbilityId
from sc2.ids.unit_typeid import UnitTypeId
from sc2.position import Point2
from sc2.unit import Unit
from sc2.units import Units

from bot.tactics.consts import ATTACK_TARGET_IGNORE_WITH_WORKERS, STAGING_ARRIVAL_RANGE
from bot.tactics.context import ArmyContext
from bot.tactics.orders import GroupOrders
from bot.tactics.positioning import TANK_SPACING, Positioning
from bot.units.common import attack_move, kite_away, path_move
from bot.utils.geometry import spread_out_point

TANK_HOLD_ARRIVAL_RANGE = 4.0
TANK_HOLD_ARRIVAL_RANGE_SQ = TANK_HOLD_ARRIVAL_RANGE * TANK_HOLD_ARRIVAL_RANGE
MIN_SIEGE_DURATION = 3.0
SIEGE_ENGAGE_RANGE = 11.0
ENEMY_HOLD_RANGE = 14.0
TANK_SEARCH_RADIUS = 18.0
LIBERATOR_GUARD_RADIUS = 9.0
LIBERATOR_GUARD_MAX = 2
STAGING_ARRIVAL_RANGE_SQ = STAGING_ARRIVAL_RANGE * STAGING_ARRIVAL_RANGE


class TankController:
    def __init__(self, ai, positioning: Positioning):
        self.ai = ai
        self.positioning = positioning
        self.siege_since: dict[int, float] = {}
        self.slot_index: dict[tuple[str, int], int] = {}

    def control(self, units: Units, orders: GroupOrders, ctx: ArmyContext) -> None:
        if not units:
            return

        ctx.prefetch_near(units)

        alive = {u.tag for u in self.ai.units}
        for tag in [t for t in self.siege_since if t not in alive]:
            del self.siege_since[tag]

        guard_point = self._liberator_guard_point()
        slots: list[Point2] | None = None

        if not orders.aggressive:
            self._assign_slots(units, orders.label)
            slots = self.positioning.tank_slots(
                orders.hold_point, max(self.slot_index[(orders.label, u.tag)] for u in units) + 1, orders.front
            )

        for unit in units:
            self._track_siege_state(unit)
            if orders.aggressive:
                self._attack(unit, units, orders, ctx, guard_point)
            else:
                slot = slots[self.slot_index[(orders.label, unit.tag)]] if slots else orders.hold_point
                self._hold(unit, units, ctx, guard_point if guard_point is not None else slot)

    def _attack(
        self, unit: Unit, units: Units, orders: GroupOrders, ctx: ArmyContext, guard_point: Point2 | None
    ) -> None:
        closest = self._closest_target_distance(unit, ctx)

        if unit.type_id == UnitTypeId.SIEGETANK and closest is not None and closest <= SIEGE_ENGAGE_RANGE:
            self._siege(unit)
            return

        if (
            unit.type_id == UnitTypeId.SIEGETANK
            and orders.staging is not None
            and cy_distance_to_squared(unit.position, orders.staging) <= STAGING_ARRIVAL_RANGE_SQ
        ):
            self._siege(unit)
            return

        if unit.type_id == UnitTypeId.SIEGETANKSIEGED:
            if self._can_unsiege(unit, ctx):
                self._unsiege(unit)
            return

        if not ctx.is_safe(unit):
            kite_away(self.ai, ctx, unit)
            return

        destination = guard_point if guard_point is not None else orders.target
        point = spread_out_point(
            unit, units, destination, radius=TANK_SPACING, nudge=TANK_SPACING,
            walkable=self.ai.in_pathing_grid,
        )

        avoidance = self.ai.mediator.get_priority_ground_avoidance_grid
        if self.ai.mediator.is_position_safe(grid=avoidance, position=point):
            attack_move(unit, point)
        else:
            path_move(self.ai, ctx, unit, point, avoid_danger=True)

    def _hold(self, unit: Unit, units: Units, ctx: ArmyContext, effective_pos: Point2) -> None:
        closest = self._closest_target_distance(unit, ctx)
        safe = ctx.is_safe(unit)
        at_hold_point = cy_distance_to_squared(unit.position, effective_pos) <= TANK_HOLD_ARRIVAL_RANGE_SQ

        if unit.type_id == UnitTypeId.SIEGETANKSIEGED:
            if not at_hold_point and self._can_unsiege(unit, ctx):
                self._unsiege(unit)
            return

        if unit.type_id == UnitTypeId.SIEGETANK and safe:
            close_enough_to_fight = closest is not None and closest <= SIEGE_ENGAGE_RANGE
            if close_enough_to_fight or at_hold_point:
                self._siege(unit)
                return

        if not safe:
            kite_away(self.ai, ctx, unit)
            return

        point = spread_out_point(
            unit, units, effective_pos, radius=TANK_SPACING, nudge=TANK_SPACING,
            walkable=self.ai.in_pathing_grid,
        )
        path_move(self.ai, ctx, unit, point, avoid_danger=True)

    def _assign_slots(self, units: Units, label: str) -> None:
        alive = {u.tag for u in self.ai.units}
        here = {u.tag for u in units}

        for key in [k for k in self.slot_index if k[1] not in alive or (k[0] == label and k[1] not in here)]:
            del self.slot_index[key]

        used = {index for (lab, _), index in self.slot_index.items() if lab == label}

        for unit in sorted(units, key=lambda u: u.tag):
            if (label, unit.tag) in self.slot_index:
                continue
            index = 0
            while index in used:
                index += 1
            self.slot_index[(label, unit.tag)] = index
            used.add(index)

    def _track_siege_state(self, unit: Unit) -> None:
        if unit.type_id == UnitTypeId.SIEGETANKSIEGED:
            self.siege_since.setdefault(unit.tag, self.ai.time)
        else:
            self.siege_since.pop(unit.tag, None)

    def _can_unsiege(self, unit: Unit, ctx: ArmyContext) -> bool:
        closest = self._closest_target_distance(unit, ctx)
        if closest is not None and closest <= ENEMY_HOLD_RANGE:
            return False

        held_since = self.siege_since.get(unit.tag)
        return held_since is None or self.ai.time - held_since >= MIN_SIEGE_DURATION

    @classmethod
    def _closest_target_distance(cls, unit: Unit, ctx: ArmyContext) -> float | None:
        if unit.tag in ctx.tank_closest:
            return ctx.tank_closest[unit.tag]

        best: float | None = None

        for e in cls._ground_enemies(unit, ctx):
            distance = cy_distance_to(unit.position, e.position)
            if e.is_structure:
                distance -= unit.radius + e.radius
            if best is None or distance < best:
                best = distance

        ctx.tank_closest[unit.tag] = best
        return best

    @staticmethod
    def _ground_enemies(unit: Unit, ctx: ArmyContext) -> list[Unit]:
        return [
            e for e in ctx.enemies_within(unit, TANK_SEARCH_RADIUS)
            if not e.is_flying and not e.is_memory and e.type_id not in ATTACK_TARGET_IGNORE_WITH_WORKERS
            and (not e.is_structure or e.is_visible)
        ]

    def _liberator_guard_point(self) -> Point2 | None:
        liberators = self.ai.units(UnitTypeId.LIBERATORAG)
        if not liberators:
            return None

        tanks = self.ai.units.of_type({UnitTypeId.SIEGETANK, UnitTypeId.SIEGETANKSIEGED})
        avoidance = self.ai.mediator.get_priority_ground_avoidance_grid

        for liberator in liberators:
            if len(cy_closer_than(tanks, LIBERATOR_GUARD_RADIUS, liberator.position)) < LIBERATOR_GUARD_MAX:
                if self.ai.mediator.is_position_safe(grid=avoidance, position=liberator.position):
                    return liberator.position

        return None

    @staticmethod
    def _siege(unit: Unit) -> None:
        if not unit.is_using_ability(AbilityId.SIEGEMODE_SIEGEMODE):
            unit(AbilityId.SIEGEMODE_SIEGEMODE)

    @staticmethod
    def _unsiege(unit: Unit) -> None:
        if not unit.is_using_ability(AbilityId.UNSIEGE_UNSIEGE):
            unit(AbilityId.UNSIEGE_UNSIEGE)