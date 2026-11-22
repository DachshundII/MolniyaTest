from cython_extensions.units_utils import cy_closest_to
from cython_extensions.geometry import cy_distance_to, cy_distance_to_squared, cy_towards

from sc2.data import Race
from sc2.ids.ability_id import AbilityId
from sc2.ids.unit_typeid import UnitTypeId
from sc2.position import Point2
from sc2.unit import Unit
from sc2.units import Units

from bot.tactics.consts import ATTACK_TARGET_IGNORE
from bot.tactics.context import ArmyContext
from bot.tactics.orders import GroupOrders
from bot.units.common import attack_move, follow_point, kite_away, path_move

MATRIX = AbilityId.EFFECT_INTERFERENCEMATRIX
TURRET = AbilityId.BUILDAUTOTURRET_AUTOTURRET
FOLLOW_LEAD = 4.0
MATRIX_FREE_AFTER = 10.0
ENERGY_TO_SPEND = 125
TURRET_STANDOFF = 4.0
TURRET_MIN_FORWARD = 2.0
TURRET_MIN_FORWARD_SQ = (TURRET_MIN_FORWARD - 0.5) * (TURRET_MIN_FORWARD - 0.5)
TURRET_MAX_ADVANCE = 6.0
TURRET_SPOT_SEARCH = 2
TURRET_CAST_MARGIN = 0.25
TURRET_TRIGGER_RANGE = 10.0
TURRET_PLACEMENT_CANDIDATES = 4

MATRIX_TARGETS_BY_RACE = {
    Race.Terran: [UnitTypeId.SIEGETANKSIEGED, UnitTypeId.THOR, UnitTypeId.BATTLECRUISER],
    Race.Protoss: [UnitTypeId.COLOSSUS, UnitTypeId.CARRIER, UnitTypeId.ARCHON, UnitTypeId.IMMORTAL, UnitTypeId.WARPPRISM],
}


class RavenController:
    def __init__(self, ai):
        self.ai = ai
        self.matrices: dict[int, float] = {}
        self.matrix_range: float = ai.game_data.abilities[MATRIX.value]._proto.cast_range
        self.turret_range: float = ai.game_data.abilities[TURRET.value]._proto.cast_range

    async def control(self, units: Units, orders: GroupOrders, ctx: ArmyContext, lead: float = FOLLOW_LEAD) -> None:
        now = self.ai.time

        for tag in [t for t, cast_at in self.matrices.items() if now - cast_at > MATRIX_FREE_AFTER]:
            del self.matrices[tag]

        ctx.prefetch_near(units)

        for unit in units:
            if await self._try_cast(unit, ctx):
                continue

            if kite_away(self.ai, ctx, unit):
                continue

            point = follow_point(orders, lead)

            if orders.aggressive:
                attack_move(unit, point)
            elif cy_distance_to(unit.position, point) > orders.hold_radius:
                path_move(self.ai, ctx, unit, point)

    async def _try_cast(self, unit: Unit, ctx: ArmyContext) -> bool:
        if isinstance(unit.order_target, int):
            return True

        if unit.is_using_ability(TURRET):
            return True

        enemies = [e for e in ctx.enemies_near(unit) if not e.is_memory and e.type_id not in ATTACK_TARGET_IGNORE]
        if not enemies:
            return False

        race = self.ai.enemy_race
        matrix_types = MATRIX_TARGETS_BY_RACE.get(race)

        if matrix_types is not None:
            if MATRIX in unit.abilities and unit.energy < ENERGY_TO_SPEND:
                return self._matrix(unit, enemies, matrix_types)
            return await self._turret(unit, enemies, ctx)

        return await self._turret(unit, enemies, ctx)

    def _matrix(self, unit: Unit, enemies: list[Unit], priority_types: list[UnitTypeId]) -> bool:
        for type_id in priority_types:
            candidates = [
                e for e in enemies
                if e.type_id == type_id and e.tag not in self.matrices
            ]
            candidates.sort(key=lambda e: cy_distance_to_squared(unit.position, e.position))
            for target in candidates:
                if cy_distance_to_squared(unit.position, target.position) <= (unit.radius + target.radius + self.matrix_range) ** 2:
                    unit(MATRIX, target)
                    self.matrices[target.tag] = self.ai.time
                    return True
        return False

    async def _turret(self, unit: Unit, enemies: list[Unit], ctx: ArmyContext) -> bool:
        if TURRET not in unit.abilities:
            return False

        closest = cy_closest_to(unit.position, enemies)
        gap = cy_distance_to(unit.position, closest.position)

        if gap >= TURRET_TRIGGER_RANGE:
            return False

        spot = await self._find_turret_spot(unit, closest, gap)
        if spot is None:
            return False

        reach = unit.radius + self.turret_range

        if cy_distance_to_squared(unit.position, spot) <= reach * reach:
            unit(TURRET, spot)
            return True

        stand = Point2(cy_towards(spot, unit.position, reach - TURRET_CAST_MARGIN))
        if not ctx.mediator.is_position_safe(grid=ctx.air_grid, position=stand):
            return False

        path_move(self.ai, ctx, unit, stand, grid=ctx.air_grid)
        return True

    async def _find_turret_spot(self, unit: Unit, enemy: Unit, gap: float) -> Point2 | None:
        forward = min(TURRET_MAX_ADVANCE, max(TURRET_MIN_FORWARD, gap - TURRET_STANDOFF))
        ideal = Point2(cy_towards(unit.position, enemy.position, forward))
        r = TURRET_SPOT_SEARCH
        cells = []

        for x in range(int(ideal.x) - r, int(ideal.x) + r + 1):
            for y in range(int(ideal.y) - r, int(ideal.y) + r + 1):
                pos = Point2((x, y))
                if (
                    cy_distance_to_squared(unit.position, pos) >= TURRET_MIN_FORWARD_SQ
                    and cy_distance_to(pos, enemy.position) < gap
                    and self.ai.in_map_bounds(pos) and self.ai.in_placement_grid(pos)
                ):
                    cells.append(pos)

        cells.sort(key=lambda p: cy_distance_to_squared(p, ideal))

        for pos in cells[:TURRET_PLACEMENT_CANDIDATES]:
            if await self.ai.can_place_single(UnitTypeId.AUTOTURRET, pos):
                return pos

        return None