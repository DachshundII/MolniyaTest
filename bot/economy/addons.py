import math
from collections import Counter
from typing import Dict, List, Optional, Set, Tuple

from cython_extensions.geometry import cy_distance_to_squared

from loguru import logger

from sc2.data import Race
from sc2.ids.ability_id import AbilityId
from sc2.ids.unit_typeid import UnitTypeId
from sc2.position import Point2
from sc2.unit import Unit

ADDON_OFFSET = Point2((2.5, -0.5))
ORDER_PATIENCE = 3.0
MAX_MISSES = 3
MAX_LIFTS = 3
LANDING_PATIENCE = 8.0
LANDING_GAPS = (7.0, 5.5)
LANDING_BATCH = 12
LANDING_BATCHES = 3
NAKED_WARNING = 60.0

PRODUCTION = (
    (UnitTypeId.BARRACKS, UnitTypeId.BARRACKSFLYING, UnitTypeId.BARRACKSREACTOR, UnitTypeId.BARRACKSTECHLAB),
    (UnitTypeId.FACTORY, UnitTypeId.FACTORYFLYING, UnitTypeId.FACTORYREACTOR, UnitTypeId.FACTORYTECHLAB),
    (UnitTypeId.STARPORT, UnitTypeId.STARPORTFLYING, UnitTypeId.STARPORTREACTOR, UnitTypeId.STARPORTTECHLAB),
)
BUILDINGS = {building for building, _, _, _ in PRODUCTION}
GROUND_OF = {flying: building for building, flying, _, _ in PRODUCTION}
FLYING = set(GROUND_OF)
KINDS = {building: (reactor, techlab) for building, _, reactor, techlab in PRODUCTION}
FLYING_OF = {building: flying for building, flying, _, _ in PRODUCTION}

LANDING_OFFSETS = sorted((Point2((x, y)) for x in range(-22, 22) for y in range(-7, 7)), key=lambda point: point.x ** 2 + point.y ** 2)


def points_to_build_addon(position: Point2) -> List[Point2]:
    addon_position: Point2 = position + ADDON_OFFSET
    return [(addon_position + Point2((x - 0.5, y - 0.5))).rounded for x in range(0, 2) for y in range(0, 2)]


def footprint_cells(position: Point2) -> List[Point2]:
    return [(position + Point2((x, y))).rounded for x in range(-1, 2) for y in range(-1, 2)] + points_to_build_addon(position)


def cells_free(ai, cells: List[Point2]) -> bool:
    return all(ai.in_map_bounds(cell) and ai.in_placement_grid(cell) and ai.in_pathing_grid(cell) for cell in cells)


def has_room(ai, position: Point2) -> bool:
    return cells_free(ai, points_to_build_addon(position))


def game_allows_addon(ai, position: Point2) -> bool:
    return ai.mediator.can_place_structure(
        position=position + ADDON_OFFSET,
        structure_type=UnitTypeId.SUPPLYDEPOT,
    )


class AddonManager:
    def __init__(self, ai):
        self.ai = ai
        self.just_ordered: Set[int] = set()
        self.ordered: Dict[int, float] = {}
        self.misses: Dict[int, int] = {}
        self.lifts: Dict[int, int] = {}
        self.landing: Dict[int, Tuple[Point2, float]] = {}
        self.failed_landings: Dict[int, List[Point2]] = {}
        self.naked_since: Dict[int, float] = {}
        self.reported: Set[int] = set()
        self.why: Dict[int, str] = {}
        self._added: Counter = Counter()

    async def update(self) -> None:
        ai = self.ai
        self.just_ordered = set()
        self._added = Counter()
        alive = {s.tag for s in ai.structures}
        for book in (self.ordered, self.misses, self.lifts, self.landing, self.failed_landings, self.naked_since, self.why):
            for tag in [t for t in book if t not in alive]:
                del book[tag]
        self.reported &= alive
        await self.land_flying()
        if not ai.build_order_runner.build_completed:
            return
        for building in (UnitTypeId.BARRACKS, UnitTypeId.FACTORY, UnitTypeId.STARPORT):
            for s in ai.structures(building).ready:
                if s.has_add_on:
                    self.settled(s.tag)
                    continue
                if self.landing.pop(s.tag, None) is not None and ai.rally_point is not None:
                    s(AbilityId.SMART, ai.rally_point)
                self.watch(s)
                if s.is_idle:
                    await self.serve(s, building)

    def settled(self, tag: int) -> None:
        for book in (self.ordered, self.misses, self.lifts, self.landing, self.failed_landings, self.naked_since, self.why):
            book.pop(tag, None)
        self.reported.discard(tag)

    def watch(self, s: Unit) -> None:
        now = self.ai.time
        since = self.naked_since.setdefault(s.tag, now)
        if now - since > NAKED_WARNING and s.tag not in self.reported:
            self.reported.add(s.tag)
            logger.warning(f"[addons] {s.type_id.name} {s.tag} at ({s.position.x:.1f}, {s.position.y:.1f}) has had no add-on for "
                           f"{now - since:.0f}s: {self.why.get(s.tag, 'it is busy producing')}")

    async def serve(self, s: Unit, building: UnitTypeId) -> None:
        ai = self.ai
        now = ai.time
        if s.tag in self.ordered:
            if now - self.ordered[s.tag] < ORDER_PATIENCE:
                return
            del self.ordered[s.tag]
            self.misses[s.tag] = self.misses.get(s.tag, 0) + 1
        misses = self.misses.get(s.tag, 0)
        room = has_room(ai, s.position)
        if room and misses:
            room = misses < MAX_MISSES and game_allows_addon(ai, s.position)
        if room:
            self.order(s, building)
        else:
            self.lift(s, building)

    def choose(self, building: UnitTypeId) -> UnitTypeId:
        ai = self.ai
        reactor, techlab = KINDS[building]
        ground = ai.structures(building)
        total = ground.amount + ai.structures(FLYING_OF[building]).amount
        reactors = sum(1 for s in ground if s.has_reactor) + self._added[reactor]
        techlabs = sum(1 for s in ground if s.has_techlab) + self._added[techlab]
        if building == UnitTypeId.BARRACKS:
            return reactor if reactors < total / 2 else techlab
        if building == UnitTypeId.FACTORY:
            return techlab if techlabs < total * ai.army_advisor.factory_techlab_ratio else reactor
        if ai.build_starport_techlab_first:
            return techlab if techlabs < total / 2 else reactor
        return reactor if reactors < total / 2 else techlab

    def order(self, s: Unit, building: UnitTypeId) -> None:
        ai = self.ai
        if (ai.worker_rushed or ai.army_advisor.zergling_rushed) and not ai.army_advisor.is_wall_closed():
            self.why[s.tag] = "a rush is on and the wall is open"
            return
        add_on = self.choose(building)
        if not ai.can_afford(add_on):
            self.why[s.tag] = f"the {add_on.name} is not affordable yet"
            return
        s.build(add_on)
        self.ordered[s.tag] = ai.time
        self.just_ordered.add(s.tag)
        self._added[add_on] += 1
        self.why.pop(s.tag, None)

    def may_lift(self, s: Unit) -> bool:
        ai = self.ai
        if ai.enemy_race == Race.Terran:
            return True
        if ai.worker_rushed or ai.army_advisor.zergling_rushed:
            return False
        if s.position == ai.main_base_ramp.barracks_in_middle:
            return ai.army_advisor.total_enemy_supply() < ai.supply_army
        return True

    def lift(self, s: Unit, building: UnitTypeId) -> None:
        ai = self.ai
        if self.lifts.get(s.tag, 0) >= MAX_LIFTS:
            self.why[s.tag] = "no room for an add-on here, and it has been moved often enough"
        elif not self.may_lift(s):
            self.why[s.tag] = "no room for an add-on here, and it must not lift now (rush, or the wall)"
        elif not ai.can_afford(self.choose(building)):
            self.why[s.tag] = "no room for an add-on here; it lifts once the add-on can be paid"
        else:
            s(AbilityId.LIFT)
            self.lifts[s.tag] = self.lifts.get(s.tag, 0) + 1
            self.misses.pop(s.tag, None)
            self.ordered[s.tag] = ai.time
            self.just_ordered.add(s.tag)
            self.why[s.tag] = "lifted: no room for an add-on where it stood"

    async def land_flying(self) -> None:
        ai = self.ai
        now = ai.time
        for s in ai.structures.of_type(FLYING):
            self.ordered.pop(s.tag, None)
        for s in ai.structures.of_type(FLYING).idle:
            record = self.landing.get(s.tag)
            if record is not None:
                if now - record[1] < LANDING_PATIENCE:
                    continue
                self.failed_landings.setdefault(s.tag, []).append(record[0])
                del self.landing[s.tag]
            spot = await self.landing_spot(s)
            if spot is None:
                self.why[s.tag] = "in the air: nowhere to land with room for its add-on"
                continue
            s(AbilityId.LAND, spot)
            self.landing[s.tag] = (spot, now)

    async def landing_spot(self, s: Unit) -> Optional[Point2]:
        ai = self.ai
        ground = GROUND_OF[s.type_id]
        failed = self.failed_landings.get(s.tag, [])
        columns = [b.position.x for b in ai.structures.of_type(BUILDINGS)]
        origin = Point2((math.floor(s.position.x) + 0.5, math.floor(s.position.y) + 0.5))
        fallback: Optional[Point2] = None
        for gap in LANDING_GAPS:
            spots: List[Point2] = []
            for offset in LANDING_OFFSETS:
                spot = origin + offset
                if any(abs(x - spot.x) < gap for x in columns) or any(cy_distance_to_squared(spot, f) < 1.0 for f in failed):
                    continue
                if not cells_free(ai, footprint_cells(spot)):
                    continue
                spots.append(spot)
                if len(spots) == LANDING_BATCH * LANDING_BATCHES:
                    break
            if fallback is None and spots:
                fallback = spots[0]
            for i in range(0, len(spots), LANDING_BATCH):
                batch = spots[i:i + LANDING_BATCH]
                fits = await ai.can_place(ground, batch)
                batch = [spot for spot, fit in zip(batch, fits) if fit]
                if not batch:
                    continue
                room = await ai.can_place(AbilityId.TERRANBUILD_SUPPLYDEPOT, [spot + ADDON_OFFSET for spot in batch])
                for spot, fits_addon in zip(batch, room):
                    if fits_addon:
                        return spot
        return fallback