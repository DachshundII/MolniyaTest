from itertools import chain
import math

from cython_extensions.units_utils import cy_closer_than, cy_closest_to
from cython_extensions.geometry import cy_distance_to_squared

from sc2.ids.unit_typeid import UnitTypeId
from sc2.ids.ability_id import AbilityId
from sc2.ids.upgrade_id import UpgradeId
from sc2.position import Point2
from sc2.unit import Unit
from sc2.units import Units

from ares import AresBot
from ares.behaviors.macro import (
    AutoSupply,
    BuildStructure,
    BuildWorkers,
    ExpansionController,
    GasBuildingController,
    MacroPlan,
    Mining,
    ProductionController,
    UpgradeCCs,
    UpgradeController,
)
from ares.consts import UnitRole

from bot.consts import HARMLESS_TO_WORKERS
from bot.economy.addons import AddonManager
from bot.economy.production import produce
from bot.economy.repair import manage_repairs
from bot.economy.worker_micro import worker_micro
from bot.economy.worker_rush_defense import worker_rush_defense
from bot.strategy.army_composition_advisor import ArmyCompositionAdvisor
from bot.strategy.worker_scouting import scout
from bot.tactics.manager import ArmyManager
from bot.utils.macro_utils import (
    handle_depot_status,
    handle_command_centers,
    research_cyclone_upgrade,
    get_rally_point,
    update_rally_points,
)


class Molniya(AresBot):

    army: ArmyManager

    def __init__(self):
        self.unit_command_uses_self_do = False
        self.distance_calculation_method = 2
        self.game_step: int = 2
        self.build_starport_techlab_first = True
        self.worker_rushed = False
        self.worker_rush_clear_since = None
        self.repair_jobs = {}
        self.repair_walks = {}
        self.repair_backoff = {}
        self.turret_backoff = {}
        self.worker_assigned_to_follow = {}
        self.cyclone_upgrade_next_look = 0.0
        self.oracle_fleeing = set()
        self.threat_fleeing = set()
        self.worker_assigned_to_defend = {}
        self.worker_assigned_to_resume_building = {}
        self.worker_assigned_to_expand = {}
        self.lifted_cc_pos = {}
        self.produce_from_starports = True
        self.produce_from_factories = True
        self.produce_from_barracks = True
        self.factory_prioritized = False
        self.made_banshee = False
        self.made_raven = False
        self.rally_point = None
        self.base_build_order = []
        self.addons = AddonManager(self)
        self.scout_worker_tag = None
        self.enemy_base_scouted = False
        self.scout_attempted = False
        self.build_order_critical_worker = None
        self.gas_bank_high = False

        self.army = None
        self._visible_enemies_loop = -1
        self._visible_enemies = None

        super().__init__()

    @property
    def visible_enemy_units(self) -> Units:
        loop = self.state.game_loop
        if self._visible_enemies_loop != loop or self._visible_enemies is None:
            self._visible_enemies = self.enemy_units.filter(lambda u: not u.is_memory)
            self._visible_enemies_loop = loop
        return self._visible_enemies

    def is_unit_position_safe(self, unit: Unit) -> bool:
        grid = self.mediator.get_air_grid if unit.is_flying else self.mediator.get_ground_grid
        return self.mediator.is_position_safe(grid=grid, position=unit.position)

    @property
    def army_composition(self) -> dict:
        advisor = self.army_advisor
        marines = advisor.marine_marauder_ratio
        marauders = 1.0 - marines

        weights = {
            UnitTypeId.MARINE: marines,
            UnitTypeId.MARAUDER: marauders,
            UnitTypeId.SIEGETANK: 0.10 if advisor.max_tanks > 0 else 0,
            UnitTypeId.CYCLONE: 0.05 if advisor.max_cyclones > 0 else 0,
            UnitTypeId.MEDIVAC: 0.05 if advisor.max_medivacs > 0 else 0,
            UnitTypeId.RAVEN: 0.02 if advisor.max_ravens > 0 else 0,
            UnitTypeId.VIKINGFIGHTER: 0.03 if advisor.max_vikings > 0 else 0,
        }

        total = sum(w for w in weights.values() if w > 0)
        if total == 0:
            return {UnitTypeId.MARINE: {"proportion": 1.0, "priority": 3}}

        priorities = {
            UnitTypeId.MARINE: 3,
            UnitTypeId.MARAUDER: 2,
            UnitTypeId.SIEGETANK: 0,
            UnitTypeId.CYCLONE: 1,
            UnitTypeId.MEDIVAC: 4,
            UnitTypeId.RAVEN: 1,
            UnitTypeId.VIKINGFIGHTER: 3,
        }

        return {
            unit: {"proportion": weight / total, "priority": priorities[unit]}
            for unit, weight in weights.items()
            if weight > 0
        }

    async def on_before_start(self) -> None:
        await super().on_before_start()
        self.client.game_step = self.game_step
        self.client.raw_affects_selection = True
        top_right = Point2((self.game_info.playable_area.right, self.game_info.playable_area.top))
        bottom_right = Point2((self.game_info.playable_area.right, self.game_info.playable_area.y))
        bottom_left = Point2((self.game_info.playable_area.x, self.game_info.playable_area.y))
        top_left = Point2((self.game_info.playable_area.x, self.game_info.playable_area.top))
        self.map_corners = [top_right, bottom_right, bottom_left, top_left]

    async def on_building_construction_started(self, unit: Unit):
        await super().on_building_construction_started(unit)
        if unit.type_id in {UnitTypeId.BARRACKS, UnitTypeId.FACTORY, UnitTypeId.STARPORT}:
            unit(AbilityId.SMART, get_rally_point(self))
        self.register_base(unit)

    async def on_building_construction_complete(self, unit: Unit):
        await super().on_building_construction_complete(unit)
        self.register_base(unit)

    def register_base(self, unit: Unit):
        if unit.type_id == UnitTypeId.COMMANDCENTER and unit.tag not in self.base_build_order:
            self.base_build_order.append(unit.tag)

    async def on_start(self) -> None:
        await super().on_start()
        self.client.game_step = self.game_step
        self.army_advisor = ArmyCompositionAdvisor(self)
        self.army_advisor.provide_advices_startup()
        self.army = ArmyManager(self)

    async def on_unit_created(self, unit: Unit):
        await super().on_unit_created(unit)
        if unit.type_id == UnitTypeId.SCV:
            self.mediator.assign_role(tag=unit.tag, role=UnitRole.GATHERING)

    async def on_unit_destroyed(self, unit_tag: int):
        await super().on_unit_destroyed(unit_tag)
        self.army_advisor.remove_unit(unit_tag)
        self.army.on_unit_destroyed(unit_tag)

    async def on_step(self, iteration: int):
        if self.townhalls.amount == 0 or self.supply_used == 0:
            await self.client.leave()
            return

        await super().on_step(iteration)

        self.army_advisor.provide_advices()
        self.resource_by_tag = {unit.tag: unit for unit in chain(self.mineral_field, self.gas_buildings)}

        macro_plan = MacroPlan()
        macro_plan.add(Mining(workers_per_gas=3 if self.supply_workers >= 13 else 0))
        macro_plan.add(AutoSupply(self.start_location))
        macro_plan.add(BuildWorkers(to_count=68))
        macro_plan.add(GasBuildingController(to_count=len(self.townhalls) * 2))
        macro_plan.add(UpgradeCCs(to=UnitTypeId.ORBITALCOMMAND, prioritize=True))
        macro_plan.add(ExpansionController(to_count=8, max_pending=2))
        macro_plan.add(ProductionController(self.army_composition, self.start_location))
        macro_plan.add(
            UpgradeController(
                upgrade_list=[
                    UpgradeId.STIMPACK,
                    UpgradeId.SHIELDWALL,
                    UpgradeId.PUNISHERGRENADES,
                    UpgradeId.TERRANINFANTRYWEAPONSLEVEL1,
                    UpgradeId.TERRANINFANTRYARMORSLEVEL1,
                    UpgradeId.TERRANINFANTRYWEAPONSLEVEL2,
                    UpgradeId.TERRANINFANTRYARMORSLEVEL2,
                    UpgradeId.TERRANINFANTRYWEAPONSLEVEL3,
                    UpgradeId.TERRANINFANTRYARMORSLEVEL3,
                    UpgradeId.BANSHEECLOAK,
                ],
                base_location=self.start_location,
            )
        )
        self.register_behavior(macro_plan)

        if not self.worker_rushed:
            produce(self)

        cancel_building(self)
        manage_repairs(self)
        await build_turrets(self)

        worker_rush_defense(self)
        worker_micro(self)

        await scout(self)

        handle_depot_status(self)
        await handle_command_centers(self)
        update_rally_points(self)
        if self.build_order_runner.build_completed:
            await research_cyclone_upgrade(self)
        await self.addons.update()

        await self.army.update()


def cancel_building(self: AresBot):
    for st in self.structures:
        if not st.is_ready and st.health_percentage < 0.1:
            st(AbilityId.CANCEL)


TURRET_BASE_RADIUS = 15.0


async def build_turrets(self: AresBot):
    per_base = self.army_advisor.turrets_per_base
    if per_base <= 0 or self.townhalls.amount == 0:
        return

    if not self.structures(UnitTypeId.ENGINEERINGBAY).ready:
        if self.structures(UnitTypeId.ENGINEERINGBAY).amount + self.already_pending(UnitTypeId.ENGINEERINGBAY) < 1:
            self.register_behavior(
                BuildStructure(
                    base_location=self.start_location,
                    structure_id=UnitTypeId.ENGINEERINGBAY,
                    closest_to=self.start_location,
                )
            )
        return

    if self.already_pending(UnitTypeId.MISSILETURRET) > 0 or not self.can_afford(UnitTypeId.MISSILETURRET):
        return

    for cc in self.townhalls.ready:
        if self.turret_backoff.get(cc.tag, 0.0) > self.time:
            continue
        if len(cy_closer_than(self.structures(UnitTypeId.MISSILETURRET), TURRET_BASE_RADIUS, cc.position)) < per_base:
            self.turret_backoff[cc.tag] = self.time + 4.0
            self.register_behavior(
                BuildStructure(
                    base_location=self.start_location,
                    structure_id=UnitTypeId.MISSILETURRET,
                    missile_turret=True,
                    to_count_per_base=per_base,
                    closest_to=cc.position,
                )
            )
            return