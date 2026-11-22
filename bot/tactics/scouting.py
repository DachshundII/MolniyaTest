from typing import List, Set

from ares.consts import UnitRole
from sc2.data import Race
from sc2.ids.unit_typeid import UnitTypeId as U
from sc2.position import Point2
from sc2.unit import Unit
from sc2.units import Units

from cython_extensions.units_utils import cy_closest_to
from cython_extensions.geometry import cy_distance_to_squared

from bot.tactics.context import ArmyContext

SWEEP_COOLDOWN = 90.0
SWEEP_HOLD = 30.0
MAX_SCOUTS = 6
MAX_SCOUT_FRACTION = 0.25
IDLE_AT_ENEMY_BASE_MIN_TIME = 300.0
BEHIND_BASE_OFFSET = 9.0

SCOUT_TYPES = (U.HELLION, U.HELLIONTANK, U.CYCLONE, U.MARINE, U.MARAUDER, U.THOR)
FALLBACK_SCOUT_TYPES = (U.BATTLECRUISER, U.LIBERATOR, U.BANSHEE, U.RAVEN, U.MEDIVAC)

_SPEED_ORDER = {t: i for i, t in enumerate(SCOUT_TYPES)}
_FALLBACK_ORDER = {t: i for i, t in enumerate(FALLBACK_SCOUT_TYPES)}


class HiddenBaseScouting:
    def __init__(self, ai):
        self.ai = ai
        self.scout_tags: Set[int] = set()
        self.started_at: float = -1e9
        self.last_sweep: float = -1e9

    def should_sweep(self) -> bool:
        ai = self.ai
        if ai.time - self.last_sweep < SWEEP_COOLDOWN or ai.enemy_structures:
            return False

        idle_at_enemy_base = (
            not ai.visible_enemy_units
            and ai.units
            and cy_distance_to_squared(
                cy_closest_to(ai.enemy_start_locations[0], ai.units).position,
                ai.enemy_start_locations[0],
            ) < 9.0
        )

        if idle_at_enemy_base and ai.time >= IDLE_AT_ENEMY_BASE_MIN_TIME and ai.enemy_race in {Race.Zerg, Race.Protoss}:
            return True

        return ai.supply_left <= 0

    def unscouted_bases(self, ctx: ArmyContext) -> List[Point2]:
        ai = self.ai
        ours = [t.position for t in ai.townhalls]
        result: List[Point2] = []

        for loc, _ in ctx.mediator.get_enemy_expansions:
            if ai.is_visible(loc) or any(cy_distance_to_squared(loc, o) < 64.0 for o in ours):
                continue
            result.append(loc)

        return result

    def _scout_pool(self, army_pool: Units) -> List[Unit]:
        preferred = sorted(
            (u for u in army_pool if u.type_id in _SPEED_ORDER and u.can_attack_ground),
            key=lambda u: (_SPEED_ORDER[u.type_id], u.tag),
        )

        if preferred:
            return preferred

        return sorted(
            (u for u in army_pool if u.type_id in FALLBACK_SCOUT_TYPES),
            key=lambda u: (_FALLBACK_ORDER[u.type_id], u.tag),
        )

    def start_sweep(self, ctx: ArmyContext, army_pool: Units) -> None:
        ai = self.ai
        bases = self.unscouted_bases(ctx)
        ground = self._scout_pool(army_pool)
        max_scouts = max(1, min(MAX_SCOUTS, int(len(ground) * MAX_SCOUT_FRACTION), len(bases) or MAX_SCOUTS))
        chosen = ground[:max_scouts]
        vikings = [u for u in army_pool if u.type_id == U.VIKINGFIGHTER][:2]
        corners = ai.map_corners

        if not (chosen and bases) and not (vikings and corners):
            return

        self.last_sweep = ai.time
        self.started_at = ai.time
        self.scout_tags = set()
        centre = ai.game_info.map_center

        for unit, base in zip(chosen, bases):
            ctx.mediator.assign_role(tag=unit.tag, role=UnitRole.SCOUTING)
            unit.attack(base)
            unit.attack(base.towards(centre, -BEHIND_BASE_OFFSET), queue=True)
            self.scout_tags.add(unit.tag)

        for i, viking in enumerate(vikings):
            ctx.mediator.assign_role(tag=viking.tag, role=UnitRole.SCOUTING)
            first = True
            for k in range(len(corners)):
                viking.attack(corners[(k + i * 2) % len(corners)], queue=not first)
                first = False
            self.scout_tags.add(viking.tag)

    def update(self, ctx: ArmyContext, army_pool: Units) -> None:
        ai = self.ai
        if self.scout_tags:
            alive = {u.tag for u in ai.units}
            self.scout_tags &= alive
            done = ai.time - self.started_at >= SWEEP_HOLD or bool(ai.enemy_structures) or not self.scout_tags

            if done:
                if self.scout_tags:
                    ctx.mediator.batch_assign_role(tags=self.scout_tags, role=UnitRole.ATTACKING)
                self.scout_tags = set()
            return

        if army_pool and self.should_sweep():
            self.start_sweep(ctx, army_pool)