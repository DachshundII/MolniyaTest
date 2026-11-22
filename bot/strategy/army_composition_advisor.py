from collections import Counter
from copy import copy
from typing import Dict, FrozenSet, List, Set

from loguru import logger

from cython_extensions.geometry import cy_distance_to_squared
from cython_extensions.units_utils import cy_closer_than

from sc2.bot_ai import BotAI
from sc2.data import Race, race_townhalls
from sc2.ids.unit_typeid import UnitTypeId
from sc2.position import Point2
from sc2.units import Units

from bot.consts import DANGEROUS_STRUCTURES, SKYTOSS_TYPES
from bot.strategy.reactions import KNOBS, REACTIONS, Scouted, react


DANGEROUS_STRUCTURE_THREAT = 5
SECOND_RAVEN_ARMY_SUPPLY = 30
_WORKER_TYPES = frozenset({UnitTypeId.PROBE, UnitTypeId.DRONE, UnitTypeId.SCV, UnitTypeId.MULE})


class ArmyCompositionAdvisor:

    def __init__(self, bot: BotAI):
        self.bot = bot
        self._army_units: Units = Units([], bot)
        self.marine_marauder_ratio = 0.5

        self.max_tanks = 10
        self.max_cyclones = 0
        self.max_hellions = 6
        self.max_ravens = 2
        self.max_medivacs = 4
        self.max_vikings = 4
        self.max_liberators = 0
        self.max_battlecruisers = 1
        self.max_banshees = 2

        self.prioritize_vikings = False

        self.raven_first = False
        self.priority_units: List[UnitTypeId] = []
        self.turrets_per_base = 0
        self.starport_now = False
        self.mech_focus = False
        self.reactions_fired: Set[str] = set()
        self.active_reactions: List[str] = []
        self._startup: Dict[str, object] = {}
        self._startup_race = None

        self.barracks_techlab_ratio = 0.5
        self.factory_techlab_ratio = 0.5
        self.starport_techlab_ratio = 0.5

        self.zergling_rushed = False
        self.enemy_likely_one_base = False

    def _refresh_army(self) -> None:
        self._army_units = self.bot.enemy_units.filter(lambda u: u.type_id not in _WORKER_TYPES)

    def is_wall_closed(self):
        ramp = self.bot.main_base_ramp
        if ramp is None:
            return False

        wall_positions = [ramp.barracks_in_middle, ramp.depot_in_middle]
        wall_positions.extend(ramp.corner_depots)
        wall_positions = [p for p in wall_positions if p is not None]

        if not wall_positions:
            return False

        close_structures = self.bot.structures.filter(
            lambda s: any(s.position.distance_to(p) < 2.0 for p in wall_positions)
        )

        return close_structures.amount >= 2

    def amount_of_enemies_of_type(self, type: UnitTypeId):
        return sum(1 for u in self._army_units if u.type_id == type)

    def remove_unit(self, tag: int) -> None:
        pass

    def supply_of(self, unit: UnitTypeId):
        if unit in {UnitTypeId.ZERGLING, UnitTypeId.BANELING}:
            return 0.5
        if unit in {UnitTypeId.RAVAGER, UnitTypeId.LURKER}:
            return 3
        if unit in {UnitTypeId.BROODLORD, UnitTypeId.ARCHON}:
            return 4
        return self.bot.calculate_supply_cost(unit)

    def total_enemy_supply(self):
        total_enemy_supply = sum(self.supply_of(u.type_id) for u in self._army_units)
        total_enemy_supply += (
            DANGEROUS_STRUCTURE_THREAT * self.bot.enemy_structures.of_type(DANGEROUS_STRUCTURES).amount
        )
        return total_enemy_supply

    def scouted(self) -> Scouted:
        return Scouted(
            race=self.bot.enemy_race,
            structures=Counter(s.type_id for s in self.bot.enemy_structures),
            units=Counter(u.type_id for u in self._army_units),
        )

    def provide_advices(self):
        if self._startup_race != self.bot.enemy_race:
            self.provide_advices_startup()

        for name, value in self._startup.items():
            setattr(self, name, copy(value))

        self._refresh_army()

        visible_enemies: Units = self.bot.visible_enemy_units

        if self.bot.enemy_race == Race.Zerg and self.bot.time < 180 and visible_enemies(UnitTypeId.ZERGLING).amount > 0:
            self.zergling_rushed = True
        elif self.zergling_rushed and self.bot.time > 400 and len(cy_closer_than(visible_enemies(UnitTypeId.ZERGLING), 20, self.bot.start_location)) == 0:
            self.zergling_rushed = False

        townhall_types = race_townhalls.get(self.bot.enemy_race)
        self.enemy_likely_one_base = bool(
            townhall_types and self.bot.time >= 300 and self.bot.enemy_structures.of_type(townhall_types).amount <= 1
        )

        if self.bot.enemy_race == Race.Zerg:
            if self.amount_of_enemies_of_type(UnitTypeId.ROACH) < len(self._army_units) / 2:
                self.marine_marauder_ratio = 0.8
            else:
                self.marine_marauder_ratio = 0.5

            enemy_mech_heavy = self.amount_of_enemies_of_type(UnitTypeId.ULTRALISK) + self.amount_of_enemies_of_type(UnitTypeId.BROODLORD) > 2
            self.max_tanks = 12 if enemy_mech_heavy else 8

            detected_anti_banshee = self.amount_of_enemies_of_type(UnitTypeId.OVERSEER) > 0 or self.bot.enemy_structures(UnitTypeId.SPORECRAWLER).amount > 0
            self.max_banshees = 0 if detected_anti_banshee else 2

            if self.bot.supply_army >= SECOND_RAVEN_ARMY_SUPPLY:
                self.max_ravens = max(self.max_ravens, 2)

        if self.bot.enemy_race == Race.Protoss:
            detected_skytoss = self.bot.enemy_structures(UnitTypeId.STARGATE).amount > 0 or any(
                self.amount_of_enemies_of_type(t) > 0 for t in SKYTOSS_TYPES
            )

            if detected_skytoss:
                self.marine_marauder_ratio = 0.9
                self.max_vikings = 10
                self.prioritize_vikings = True
            else:
                self.marine_marauder_ratio = 0.5
                self.max_vikings = 4
                self.prioritize_vikings = False

            enemy_mech_heavy = self.amount_of_enemies_of_type(UnitTypeId.COLOSSUS) + self.amount_of_enemies_of_type(UnitTypeId.IMMORTAL) > 3
            self.max_tanks = 6 if enemy_mech_heavy else 4

            hellion_countered = any(self.amount_of_enemies_of_type(t) > 0 for t in {UnitTypeId.IMMORTAL, UnitTypeId.COLOSSUS, UnitTypeId.ARCHON})
            self.max_hellions = 0 if hellion_countered else 2

            detected_anti_banshee = (
                self.amount_of_enemies_of_type(UnitTypeId.PHOENIX) > 0
                or self.amount_of_enemies_of_type(UnitTypeId.OBSERVER) > 0
                or self.bot.enemy_structures(UnitTypeId.PHOTONCANNON).amount > 0
            )
            self.max_banshees = 0 if detected_anti_banshee else 2

        if self.bot.enemy_race == Race.Terran:
            enemy_mech_heavy = (
                self.amount_of_enemies_of_type(UnitTypeId.SIEGETANK)
                + self.amount_of_enemies_of_type(UnitTypeId.CYCLONE)
                + self.amount_of_enemies_of_type(UnitTypeId.THOR)
                > 4
            )
            self.max_tanks = 14 if enemy_mech_heavy else 8

            detected_enemy_tanks = self.amount_of_enemies_of_type(UnitTypeId.SIEGETANK) + self.amount_of_enemies_of_type(UnitTypeId.SIEGETANKSIEGED) > 0
            self.max_liberators = 2 if detected_enemy_tanks else 0
            self.max_hellions = 2 if enemy_mech_heavy else 6

            detected_anti_banshee = self.amount_of_enemies_of_type(UnitTypeId.RAVEN) > 0 or self.bot.enemy_structures(UnitTypeId.MISSILETURRET).amount > 0
            self.max_banshees = 0 if detected_anti_banshee else 2

        self.active_reactions = react(self.scouted(), self)
        for name in self.active_reactions:
            if name not in self.reactions_fired:
                self.reactions_fired.add(name)
                reaction = next(r for r in REACTIONS if r.name == name)
                logger.info(f"[react] {name} (at {self.bot.time:.0f}s): {reaction.why}")

    def provide_advices_startup(self):
        self.mech_focus = False

        if self.bot.enemy_race == Race.Terran:
            self.max_medivacs = 2
            self.max_vikings = 10
            self.max_battlecruisers = 1
            self.max_ravens = 3
            self.max_tanks = 8
            self.max_cyclones = 0
            self.max_banshees = 2
            self.marine_marauder_ratio = 0.7
            self.barracks_techlab_ratio = 0.4
            self.factory_techlab_ratio = 0.5
            self.starport_techlab_ratio = 0.5

        if self.bot.enemy_race == Race.Protoss:
            self.max_medivacs = 4
            self.max_vikings = 4
            self.max_battlecruisers = 1
            self.max_ravens = 2
            self.max_tanks = 4
            self.max_cyclones = 12
            self.max_banshees = 2
            self.mech_focus = True
            self.marine_marauder_ratio = 0.5
            self.barracks_techlab_ratio = 0.5
            self.factory_techlab_ratio = 1.0
            self.starport_techlab_ratio = 0.5

        if self.bot.enemy_race == Race.Zerg:
            self.max_medivacs = 6
            self.max_vikings = 2
            self.max_battlecruisers = 1
            self.max_ravens = 1
            self.max_tanks = 8
            self.max_cyclones = 0
            self.max_banshees = 2
            self.marine_marauder_ratio = 0.5
            self.barracks_techlab_ratio = 0.4
            self.factory_techlab_ratio = 0.5
            self.starport_techlab_ratio = 0.5

        self._startup_race = self.bot.enemy_race
        self._startup = {name: copy(getattr(self, name)) for name in KNOBS}