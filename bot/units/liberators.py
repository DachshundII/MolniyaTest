from cython_extensions.geometry import cy_distance_to_squared, cy_towards

from sc2.ids.ability_id import AbilityId
from sc2.ids.unit_typeid import UnitTypeId
from sc2.position import Point2
from sc2.unit import Unit
from sc2.units import Units

from bot.tactics.consts import ATTACK_TARGET_IGNORE_WITH_WORKERS
from bot.tactics.context import ArmyContext
from bot.tactics.orders import GroupOrders
from bot.units.common import attack_move, follow_point, kite_away, path_move

MIN_AG_DURATION = 3.0
MIN_AG_SHOOTING = 3.0
AG_ZONE_RADIUS = 5.0
AG_ZONE_MARGIN = 1.5
AG_ZONE_REACH = AG_ZONE_RADIUS + AG_ZONE_MARGIN
AG_ZONE_REACH_SQ = AG_ZONE_REACH * AG_ZONE_REACH
AG_OFFSET = 4.0
AG_DEFAULT_CAST_RANGE = 10.0
SIEGE_MORPH_TIMEOUT = 6.0
SIEGE_RETRY_LIMIT = 3
SIEGE_APPROACH_PATIENCE = 15.0
SIEGE_GIVE_UP_SECONDS = 20.0
CAST_BUFFER = 0.25
MORPH_COMMAND_COOLDOWN = 0.5
SIEGE_SEEK_RANGE = 45.0
SIEGE_SEEK_RANGE_SQ = SIEGE_SEEK_RANGE * SIEGE_SEEK_RANGE

_SIEGE_TANKS = {UnitTypeId.SIEGETANK, UnitTypeId.SIEGETANKSIEGED}
_AG = AbilityId.MORPH_LIBERATORAGMODE
_AA = AbilityId.MORPH_LIBERATORAAMODE


class LiberatorController:
    def __init__(self, ai):
        self.ai = ai
        self.ag_cast_range: float = max(AG_DEFAULT_CAST_RANGE, ai.game_data.abilities[_AG.value]._proto.cast_range)
        self.cast_range: float = self.ag_cast_range - CAST_BUFFER
        self.cast_range_sq: float = self.cast_range * self.cast_range
        self.ag_cast_range_plus_reach_sq: float = (self.ag_cast_range + AG_ZONE_REACH) ** 2
        self.ag_locked_until: dict[int, float] = {}
        self.ag_zone: dict[int, Point2] = {}
        self.ag_first_seen: dict[int, float] = {}
        self.last_morph_command: dict[int, float] = {}
        self.siege_ordered_at: dict[int, float] = {}
        self.siege_orders: dict[int, int] = {}
        self.siege_started: dict[int, float] = {}
        self.no_siege_until: dict[int, float] = {}

    def control(self, units: Units, orders: GroupOrders, ctx: ArmyContext) -> None:
        ctx.prefetch_near(units)
        alive = {u.tag for u in self.ai.units}

        for table in (
            self.ag_locked_until,
            self.ag_zone,
            self.ag_first_seen,
            self.last_morph_command,
            self.siege_ordered_at,
            self.siege_orders,
            self.siege_started,
            self.no_siege_until,
        ):
            for tag in [t for t in table if t not in alive]:
                del table[tag]

        for unit in units:
            if unit.type_id == UnitTypeId.LIBERATORAG:
                self._end_attempt(unit)
                self._maybe_leave_ag(unit, ctx)
                continue

            self.ag_first_seen.pop(unit.tag, None)

            if self._try_siege(unit, ctx):
                continue

            if kite_away(self.ai, ctx, unit):
                continue

            point = follow_point(orders)
            if orders.aggressive:
                attack_move(unit, point)
            else:
                hold_radius = orders.hold_radius
                if cy_distance_to_squared(unit.position, point) > hold_radius * hold_radius:
                    path_move(self.ai, ctx, unit, point)

    def _siege_target(self, unit: Unit) -> Unit | None:
        unit_pos = unit.position
        visible = [
            e for e in self.ai.enemy_units
            if not e.is_memory and not e.is_flying
            and cy_distance_to_squared(unit_pos, e.position) <= SIEGE_SEEK_RANGE_SQ
        ]

        tanks = [e for e in visible if e.type_id in _SIEGE_TANKS]
        if tanks:
            return min(tanks, key=lambda e: cy_distance_to_squared(unit_pos, e.position))

        others = [e for e in visible if e.type_id not in ATTACK_TARGET_IGNORE_WITH_WORKERS]
        if others:
            return min(others, key=lambda e: cy_distance_to_squared(unit_pos, e.position))

        return None

    def _try_siege(self, unit: Unit, ctx: ArmyContext) -> bool:
        now = self.ai.time

        if not self._can_morph(unit):
            return True

        if now - self.siege_ordered_at.get(unit.tag, -1e9) < SIEGE_MORPH_TIMEOUT:
            return True

        if self.no_siege_until.get(unit.tag, 0.0) > now:
            return False

        target = self._siege_target(unit)
        if target is None:
            self._end_attempt(unit)
            return False

        if not ctx.is_safe(unit):
            return False

        if now - self.siege_started.setdefault(unit.tag, now) > SIEGE_APPROACH_PATIENCE:
            return self._give_up(unit, now)

        siege_pos: Point2 = Point2(cy_towards(target.position, unit.position, AG_OFFSET))
        cast_range = self.cast_range

        if cy_distance_to_squared(unit.position, siege_pos) <= self.cast_range_sq:
            if self.siege_orders.get(unit.tag, 0) >= SIEGE_RETRY_LIMIT:
                return self._give_up(unit, now)

            unit(_AG, siege_pos)
            self.siege_orders[unit.tag] = self.siege_orders.get(unit.tag, 0) + 1
            self.siege_ordered_at[unit.tag] = now
            self.last_morph_command[unit.tag] = now
            self.ag_locked_until[unit.tag] = now + MIN_AG_DURATION
            self.ag_zone[unit.tag] = siege_pos
            return True

        approach = Point2(cy_towards(siege_pos, unit.position, cast_range))
        path_move(self.ai, ctx, unit, approach)
        return True

    def _end_attempt(self, unit: Unit) -> None:
        for table in (self.siege_ordered_at, self.siege_orders, self.siege_started):
            table.pop(unit.tag, None)

    def _give_up(self, unit: Unit, now: float) -> bool:
        self._end_attempt(unit)
        self.ag_zone.pop(unit.tag, None)
        self.ag_locked_until.pop(unit.tag, None)
        self.no_siege_until[unit.tag] = now + SIEGE_GIVE_UP_SECONDS
        return False

    def _maybe_leave_ag(self, unit: Unit, ctx: ArmyContext) -> None:
        now = self.ai.time
        first_seen = self.ag_first_seen.setdefault(unit.tag, now)
        locked_until = self.ag_locked_until.get(unit.tag)

        if locked_until is not None and now < locked_until:
            return

        if now - first_seen < MIN_AG_SHOOTING:
            return

        if self._something_to_shoot(unit):
            return

        if not self._can_morph(unit):
            return

        unit(_AA)
        self.last_morph_command[unit.tag] = now
        self.ag_locked_until.pop(unit.tag, None)
        self.ag_zone.pop(unit.tag, None)

    def _something_to_shoot(self, unit: Unit) -> bool:
        zone = self.ag_zone.get(unit.tag)
        unit_pos = unit.position
        plus_reach_sq = self.ag_cast_range_plus_reach_sq

        for e in self.ai.enemy_units:
            if e.is_memory or e.is_flying or e.type_id in ATTACK_TARGET_IGNORE_WITH_WORKERS:
                continue

            if zone is not None:
                if cy_distance_to_squared(e.position, zone) <= AG_ZONE_REACH_SQ:
                    return True
            elif cy_distance_to_squared(e.position, unit_pos) <= plus_reach_sq:
                return True

        return False

    def _can_morph(self, unit: Unit) -> bool:
        last = self.last_morph_command.get(unit.tag)
        return last is None or self.ai.time - last >= MORPH_COMMAND_COOLDOWN