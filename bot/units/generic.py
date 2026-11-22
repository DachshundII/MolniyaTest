from cython_extensions.units_utils import cy_closest_to
from cython_extensions.geometry import cy_distance_to_squared

from ares.behaviors.combat.individual import StutterUnitBack
from sc2.unit import Unit
from sc2.units import Units

from bot.tactics.context import ArmyContext
from bot.tactics.orders import GroupOrders
from bot.units.common import attack_move, attack_unit, follow_point, path_move, run


class GenericController:
    def __init__(self, ai):
        self.ai = ai

    def control(self, units: Units, orders: GroupOrders, ctx: ArmyContext) -> None:
        ctx.prefetch_near(units)

        for unit in units:
            targets = ctx.targets_near(unit)

            if not targets:
                point = follow_point(orders, walkable=None if unit.is_flying else self.ai.in_pathing_grid)
                if orders.aggressive:
                    attack_move(unit, point)
                else:
                    hold_radius = orders.hold_radius
                    if cy_distance_to_squared(unit.position, point) > hold_radius * hold_radius:
                        path_move(self.ai, ctx, unit, point)
                continue

            nearest: Unit = cy_closest_to(unit.position, targets)
            enemy_range = nearest.air_range if unit.is_flying else nearest.ground_range
            my_range = unit.air_range if nearest.is_flying else unit.ground_range

            if 0 < enemy_range < my_range:
                if run(self.ai, StutterUnitBack(unit=unit, target=nearest, kite_via_pathing=True,
                                                grid=ctx.grid_for(unit))):
                    continue

            attack_unit(unit, nearest)