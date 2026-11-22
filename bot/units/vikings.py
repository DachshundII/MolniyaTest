from cython_extensions.combat_utils import cy_pick_enemy_target
from cython_extensions.units_utils import cy_in_attack_range
from cython_extensions.units_utils import cy_closest_to
from cython_extensions.geometry import cy_distance_to

from ares.behaviors.combat.individual import StutterUnitBack
from sc2.unit import Unit
from sc2.units import Units

from bot.tactics.context import ArmyContext
from bot.tactics.orders import GroupOrders
from bot.units.common import attack_move, attack_unit, follow_point, kite_away, path_move, run


class VikingController:
    def __init__(self, ai):
        self.ai = ai

    def control(self, units: Units, orders: GroupOrders, ctx: ArmyContext) -> None:
        ctx.prefetch_near(units)
        for unit in units:
            self._control_unit(unit, orders, ctx)

    def _control_unit(self, unit: Unit, orders: GroupOrders, ctx: ArmyContext) -> None:
        targets = ctx.targets_near(unit)

        if not targets:
            self._no_fight(unit, orders, ctx)
            return

        in_range = cy_in_attack_range(unit, targets)
        target: Unit = cy_pick_enemy_target(in_range) if in_range else cy_closest_to(unit.position, targets)

        if run(self.ai, StutterUnitBack(unit=unit, target=target, kite_via_pathing=True, grid=ctx.air_grid)):
            return

        attack_unit(unit, target)

    def _no_fight(self, unit: Unit, orders: GroupOrders, ctx: ArmyContext) -> None:
        if kite_away(self.ai, ctx, unit):
            return

        point = follow_point(orders)

        if orders.aggressive:
            attack_move(unit, point)
        elif cy_distance_to(unit.position, point) > orders.hold_radius:
            path_move(self.ai, ctx, unit, point)