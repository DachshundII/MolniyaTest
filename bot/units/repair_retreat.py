from collections.abc import Callable

from cython_extensions.geometry import cy_distance_to_squared

from sc2.position import Point2
from sc2.unit import Unit

from bot.tactics.context import ArmyContext
from bot.tactics.orders import GroupOrders


class RepairRetreat:
    def __init__(
        self,
        ai,
        spot: Callable[[Unit, GroupOrders, ArmyContext], Point2],
        below: float = 0.4,
        resume: float = 0.9,
        patience: float = 25.0,
        cooldown: float = 60.0,
        wait_radius: float = 6.0,
    ):
        self.ai = ai
        self.spot = spot
        self.below, self.resume, self.patience, self.cooldown, self.wait_radius = below, resume, patience, cooldown, wait_radius
        self.wait_radius_sq = wait_radius * wait_radius
        self.retreating: set[int] = set()
        self.wait_since: dict[int, float] = {}
        self.health: dict[int, float] = {}
        self.no_retreat_until: dict[int, float] = {}

    def forget(self, alive: set[int]) -> None:
        for table in (self.wait_since, self.health, self.no_retreat_until):
            for tag in [t for t in list(table) if t not in alive]:
                del table[tag]
        self.retreating &= alive

    def update(self, unit: Unit, orders: GroupOrders, ctx: ArmyContext):
        tag = unit.tag
        now = self.ai.time
        health = unit.health_percentage

        if tag in self.retreating:
            if health >= self.resume:
                self._done(tag)
                return None

            spot = self.spot(unit, orders, ctx)
            if cy_distance_to_squared(unit.position, spot) > self.wait_radius_sq:
                self.wait_since.pop(tag, None)
                self.health.pop(tag, None)
                return spot

            since = self.wait_since.setdefault(tag, now)

            if health > self.health.get(tag, health):
                since = self.wait_since[tag] = now

            self.health[tag] = health

            if now - since > self.patience:
                self._done(tag)
                self.no_retreat_until[tag] = now + self.cooldown
                return None

            return spot

        if health < self.below and now >= self.no_retreat_until.get(tag, 0.0):
            self.retreating.add(tag)
            return self.spot(unit, orders, ctx)

        return None

    def _done(self, tag: int) -> None:
        self.retreating.discard(tag)
        self.wait_since.pop(tag, None)
        self.health.pop(tag, None)