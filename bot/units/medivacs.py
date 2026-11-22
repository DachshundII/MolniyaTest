from cython_extensions.geometry import cy_distance_to

from ares.behaviors.combat.individual import MedivacHeal
from ares.consts import UnitTreeQueryType
from sc2.ids.ability_id import AbilityId
from sc2.unit import Unit
from sc2.units import Units

from bot.tactics.context import ArmyContext
from bot.tactics.orders import GroupOrders
from bot.units.common import attack_move, follow_point, path_move, run

HEAL_SEARCH_RADIUS = 10.0
BEHIND_OFFSET = 4.0


class MedivacController:
    def __init__(self, ai):
        self.ai = ai

    def control(self, units: Units, orders: GroupOrders, ctx: ArmyContext) -> None:
        if not units:
            return

        nearby_allies = ctx.mediator.get_units_in_range(
            start_points=list(units),
            distances=HEAL_SEARCH_RADIUS,
            query_tree=UnitTreeQueryType.AllOwn,
            return_as_dict=True,
        )

        for unit in units:
            self._control_unit(unit, orders, ctx, nearby_allies.get(unit.tag, []))

    def _control_unit(self, unit: Unit, orders: GroupOrders, ctx: ArmyContext, allies) -> None:
        if run(self.ai, MedivacHeal(unit=unit, close_allied=list(allies), grid=ctx.air_grid, keep_safe=True)):
            return

        point = follow_point(orders)

        if orders.aggressive:
            attack_move(unit, point)
            if AbilityId.EFFECT_MEDIVACIGNITEAFTERBURNERS in unit.abilities:
                unit(AbilityId.EFFECT_MEDIVACIGNITEAFTERBURNERS)
            return

        point = point - orders.front * BEHIND_OFFSET

        if cy_distance_to(unit.position, point) > orders.hold_radius:
            path_move(self.ai, ctx, unit, point)
            if AbilityId.EFFECT_MEDIVACIGNITEAFTERBURNERS in unit.abilities:
                unit(AbilityId.EFFECT_MEDIVACIGNITEAFTERBURNERS)