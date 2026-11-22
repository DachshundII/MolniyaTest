from typing import List, Optional

from cython_extensions.combat_utils import cy_attack_ready, cy_pick_enemy_target
from cython_extensions.units_utils import cy_in_attack_range
from cython_extensions.geometry import cy_distance_to, cy_distance_to_squared
from cython_extensions.units_utils import cy_closest_to

from sc2.ids.ability_id import AbilityId
from sc2.ids.buff_id import BuffId
from sc2.ids.unit_typeid import UnitTypeId
from sc2.position import Point2
from sc2.unit import Unit
from sc2.units import Units

from bot.tactics.consts import (
    FIGHT_GHOST_MAX_AGE,
    KITE_IN_MELEE_RADIUS,
    KITE_IN_RESULT,
    LOCAL_FIGHT_RADIUS_SQ,
)
from bot.tactics.context import ArmyContext
from bot.tactics.orders import GroupOrders
from bot.units.common import (
    all_melee,
    attack_move,
    attack_unit,
    futile_to_kite,
    kite_away,
    kite_from_banelings,
    path_move,
    step_back_from,
    stutter_forward,
    target_harmless,
)
from bot.utils.geometry import Crowd, spread_out_point

TANK_SPLIT_RANGE = 24.0
TANK_SPLIT_RANGE_SQ = TANK_SPLIT_RANGE * TANK_SPLIT_RANGE
TANK_SPLIT_RADIUS = 2.2
TANK_SPLIT_NUDGE = 2.5

_STIM = {
    UnitTypeId.MARINE: (AbilityId.EFFECT_STIM_MARINE, BuffId.STIMPACK),
    UnitTypeId.MARAUDER: (AbilityId.EFFECT_STIM_MARAUDER, BuffId.STIMPACKMARAUDER),
}


class BioController:
    def __init__(self, ai):
        self.ai = ai
        self._tanks: List[Unit] = []
        self._crowd: Optional[Crowd] = None

    def control(self, units: Units, orders: GroupOrders, ctx: ArmyContext) -> None:
        ctx.prefetch_near(units)

        self._tanks = (
            [
                e
                for e in self.ai.enemy_units
                if e.type_id == UnitTypeId.SIEGETANKSIEGED and (not e.is_memory or e.age <= FIGHT_GHOST_MAX_AGE)
            ]
            if orders.aggressive and not orders.retreating and orders.staging is None and not orders.pausing
            else []
        )

        self._crowd = Crowd(units) if orders.aggressive else None

        for unit in units:
            self._control_unit(unit, units, orders, ctx)

    def _split_point(self, unit: Unit, units: Units) -> Optional[Point2]:
        if not self._tanks:
            return None

        tank = cy_closest_to(unit.position, self._tanks)
        if cy_distance_to_squared(unit.position, tank.position) > TANK_SPLIT_RANGE_SQ:
            return None

        return spread_out_point(
            unit,
            units,
            tank.position,
            radius=TANK_SPLIT_RADIUS,
            nudge=TANK_SPLIT_NUDGE,
            walkable=self.ai.in_pathing_grid,
            crowd=self._crowd,
        )

    def _control_unit(self, unit: Unit, units: Units, orders: GroupOrders, ctx: ArmyContext) -> None:
        if orders.aggressive and unit.health < 10:
            safe_spot = ctx.mediator.find_closest_safe_spot(from_pos=unit.position, grid=ctx.grid_for(unit))
            path_move(self.ai, ctx, unit, safe_spot, sense_danger=False)
            return

        targets = ctx.targets_near(unit)
        if not targets:
            self._no_fight(unit, units, orders, ctx)
            return

        in_range = cy_in_attack_range(unit, targets)
        if not in_range and not ctx.banelings_near(unit):
            split = self._split_point(unit, units)
            if split is not None:
                attack_move(unit, split)
                return

        target: Unit = cy_pick_enemy_target(in_range) if in_range else cy_closest_to(unit.position, targets)

        banelings = ctx.banelings_near(unit)
        close_banelings = ctx.close_banelings(unit)

        if cy_attack_ready(self.ai, unit, target):
            if in_range:
                self._stim(unit, target, ctx)
            attack_unit(unit, target)
            return

        if close_banelings:
            kite_from_banelings(self.ai, ctx, unit, close_banelings, orders)
            return

        close_melee = ctx.close_melee(unit)
        if close_melee:
            step_back_from(self.ai, ctx, unit, close_melee, orders)
            return

        advance = orders.advance_result(unit)
        winning = (
            advance is not None
            and advance >= KITE_IN_RESULT
            and not banelings
            and not ctx.melee_within(unit, KITE_IN_MELEE_RADIUS)
            and not all_melee(
                [
                    e
                    for e in ctx.enemies_near(unit)
                    if not e.is_memory
                    and cy_distance_to_squared(e.position, unit.position) <= LOCAL_FIGHT_RADIUS_SQ
                ]
            )
        )
        futile = futile_to_kite(unit, target) and not banelings
        harmless = target_harmless(target) and not banelings

        if not futile and not harmless and not winning and not ctx.is_safe(unit):
            if orders.retreating:
                path_move(self.ai, ctx, unit, orders.hold_point)
                return
            if kite_away(self.ai, ctx, unit):
                return

        if winning or harmless:
            if stutter_forward(self.ai, unit, target):
                return
        attack_unit(unit, target)

    def _no_fight(self, unit: Unit, units: Units, orders: GroupOrders, ctx: ArmyContext) -> None:
        if orders.aggressive:
            point = self._split_point(unit, units)
            if point is None:
                point = orders.target
                if cy_distance_to(unit.position, point) > unit.ground_range:
                    point = spread_out_point(
                        unit, units, point, walkable=self.ai.in_pathing_grid, crowd=self._crowd
                    )
            attack_move(unit, point)
            return

        position = orders.bio_position if orders.bio_position is not None else orders.hold_point
        if cy_distance_to(unit.position, position) > orders.hold_radius:
            path_move(self.ai, ctx, unit, position)

    def _stim(self, unit: Unit, target: Unit, ctx: ArmyContext) -> None:
        entry = _STIM.get(unit.type_id)
        if entry is None or not ctx.stim_researched:
            return

        ability, buff = entry
        if unit.health_percentage < 1.0 or unit.has_buff(buff):
            return

        if cy_distance_to(target.position, unit.position) >= unit.ground_range:
            return

        unit(ability)