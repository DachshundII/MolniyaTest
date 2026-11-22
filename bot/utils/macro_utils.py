from typing import Optional

from sc2.bot_ai import BotAI
from sc2.ids.ability_id import AbilityId
from sc2.ids.unit_typeid import UnitTypeId
from sc2.position import Point2
from sc2.unit import Unit
from sc2.units import Units

from cython_extensions.units_utils import cy_closer_than, cy_closest_to
from cython_extensions.geometry import cy_distance_to_squared, cy_towards


def handle_depot_status(self: BotAI):
    visible_ground_enemies: Units = self.visible_enemy_units.not_flying.exclude_type(
        {UnitTypeId.REAPER, UnitTypeId.COLOSSUS}
    )
    if visible_ground_enemies.amount == 0:
        for depo in self.structures(UnitTypeId.SUPPLYDEPOT).ready:
            depo(AbilityId.MORPH_SUPPLYDEPOT_LOWER)
    else:
        for depo in self.structures(UnitTypeId.SUPPLYDEPOT).ready:
            closest_enemy = cy_closest_to(depo.position, visible_ground_enemies)
            if cy_distance_to_squared(closest_enemy.position, depo.position) >= 144:
                depo(AbilityId.MORPH_SUPPLYDEPOT_LOWER)
        for depo in self.structures(UnitTypeId.SUPPLYDEPOTLOWERED).ready:
            closest_enemy = cy_closest_to(depo.position, visible_ground_enemies)
            if cy_distance_to_squared(closest_enemy.position, depo.position) < 144:
                blocking: Units = self.units.filter(
                    lambda u: cy_distance_to_squared(u.position, depo.position) < 2.25
                )
                for u in blocking:
                    u.move(Point2(cy_towards(u.position, self.start_location, 2)))
                depo(AbilityId.MORPH_SUPPLYDEPOT_RAISE)


CYCLONE_UPGRADES = (
    (AbilityId.RESEARCH_CYCLONELOCKONDAMAGE, "CYCLONELOCKONDAMAGEUPGRADE"),
    (AbilityId.RESEARCH_CYCLONERAPIDFIRELAUNCHERS, "CYCLONERAPIDFIRELAUNCHERS"),
    (AbilityId.FACTORYTECHLABRESEARCH_CYCLONERESEARCHHURRICANETHRUSTERS, "HURRICANETHRUSTERS"),
    (AbilityId.FACTORYTECHLABRESEARCH_RESEARCHLOCKONRANGEUPGRADE, "CYCLONELOCKONRANGEUPGRADE"),
)
CYCLONE_UPGRADE_LOOK_SECONDS = 3.0


async def research_cyclone_upgrade(self: BotAI):
    from sc2.ids.upgrade_id import UpgradeId

    if self.time < self.cyclone_upgrade_next_look:
        return
    if self.units(UnitTypeId.CYCLONE).amount + self.already_pending(UnitTypeId.CYCLONE) == 0:
        return
    techlabs: Units = self.structures(UnitTypeId.FACTORYTECHLAB).ready.idle
    if techlabs.amount == 0:
        return
    self.cyclone_upgrade_next_look = self.time + CYCLONE_UPGRADE_LOOK_SECONDS
    offered = await self.get_available_abilities(techlabs, ignore_resource_requirements=True)
    for techlab, abilities in zip(techlabs, offered):
        for ability, upgrade_name in CYCLONE_UPGRADES:
            upgrade = getattr(UpgradeId, upgrade_name)
            if ability in abilities and self.already_pending_upgrade(upgrade) == 0 and self.can_afford(upgrade):
                techlab(ability)
                return


async def handle_command_centers(self: BotAI):
    if self.minerals < 800:
        for oc in self.townhalls(UnitTypeId.ORBITALCOMMAND).filter(lambda x: x.energy >= 50):
            mfs = cy_closer_than(self.mineral_field, 10, oc.position)
            if mfs:
                mf: Unit = max(mfs, key=lambda x: x.mineral_contents)
                oc(AbilityId.CALLDOWNMULE_CALLDOWNMULE, mf)

    bases = sorted(self.townhalls, key=lambda cc: cy_distance_to_squared(cc.position, self.start_location))
    first_three_tags = {cc.tag for cc in bases[:3]}

    for cc in self.townhalls(UnitTypeId.COMMANDCENTER).ready:
        if cc.is_using_ability(AbilityId.UPGRADETOORBITAL_ORBITALCOMMAND) or cc.is_using_ability(AbilityId.UPGRADETOPLANETARYFORTRESS_PLANETARYFORTRESS):
            continue

        if cc.tag in first_three_tags:
            if self.can_afford(UnitTypeId.ORBITALCOMMAND):
                cc(AbilityId.UPGRADETOORBITAL_ORBITALCOMMAND)
                break
        else:
            if self.can_afford(UnitTypeId.PLANETARYFORTRESS):
                cc(AbilityId.UPGRADETOPLANETARYFORTRESS_PLANETARYFORTRESS)
                break

    for cc in self.townhalls.ready:
        if (
            cc.type_id in {UnitTypeId.COMMANDCENTER, UnitTypeId.ORBITALCOMMAND, UnitTypeId.PLANETARYFORTRESS}
            and cc.health < 600
            and self.visible_enemy_units.amount > 0
            and cy_distance_to_squared(cy_closest_to(cc.position, self.visible_enemy_units).position, cc.position) < 64
        ):
            if cc.type_id == UnitTypeId.PLANETARYFORTRESS:
                continue

            if cc.is_using_ability(AbilityId.UPGRADETOORBITAL_ORBITALCOMMAND):
                cc(AbilityId.CANCEL_MORPHORBITAL)
            if cc.is_using_ability(AbilityId.COMMANDCENTERTRAIN_SCV):
                cc(AbilityId.CANCEL_LAST)
            self.lifted_cc_pos[cc.tag] = cc.position
            cc(AbilityId.LIFT)
            continue

        if cc.type_id in {UnitTypeId.COMMANDCENTERFLYING, UnitTypeId.ORBITALCOMMANDFLYING}:
            if cc.health < 600 or (
                self.visible_enemy_units.amount > 0
                and cy_distance_to_squared(cy_closest_to(cc.position, self.visible_enemy_units).position, cc.position) < 64
            ):
                cc.move(Point2(cy_towards(self.start_location, self.game_info.map_center, 6)))
            else:
                expo_pos = self.lifted_cc_pos[cc.tag]
                if cy_distance_to_squared(cc.position, expo_pos) < 1:
                    cc(AbilityId.LAND, expo_pos)
                else:
                    cc.move(expo_pos)


def closest_ramp_point(self: BotAI, position: Point2, max_distance: float = 15.0) -> Optional[Point2]:
    ramps = getattr(self.game_info, "map_ramps", None)
    if not ramps:
        return None
    closest = min(ramps, key=lambda r: cy_distance_to_squared(r.top_center, position))
    if cy_distance_to_squared(closest.top_center, position) > max_distance * max_distance:
        return None
    return closest.top_center


def get_defend_point(self: BotAI) -> Point2:
    townhalls: Units = self.townhalls
    if townhalls.amount >= 4:
        centroid = Point2(
            (
                sum(t.position.x for t in townhalls) / townhalls.amount,
                sum(t.position.y for t in townhalls) / townhalls.amount,
            )
        )
        base: Unit = cy_closest_to(centroid, townhalls)
    else:
        last_built_tag = next(
            (tag for tag in reversed(self.base_build_order) if townhalls.find_by_tag(tag) is not None),
            None,
        )
        base = (
            townhalls.find_by_tag(last_built_tag)
            if last_built_tag is not None
            else cy_closest_to(self.start_location, townhalls)
        )

    if base.position == self.start_location:
        return self.main_base_ramp.barracks_in_middle

    ramp_point = closest_ramp_point(self, base.position)
    if ramp_point is not None:
        return ramp_point

    enemy_base = self.enemy_start_locations[0]
    direction = (enemy_base - base.position).normalized
    distance_modifier = 10
    if townhalls.amount >= 4:
        distance_modifier = 20
    elif townhalls.amount >= 3:
        distance_modifier = 15
    return base.position + direction * distance_modifier


def get_rally_point(self: BotAI) -> Point2:
    if self.townhalls.amount <= 1 and not self.army_advisor.is_wall_closed():
        return self.start_location
    return get_defend_point(self)


def update_rally_points(self: BotAI):
    rally: Point2 = get_rally_point(self)
    if self.rally_point is not None and rally == self.rally_point:
        return
    self.rally_point = rally
    for structure in self.structures.of_type(
        {UnitTypeId.BARRACKS, UnitTypeId.FACTORY, UnitTypeId.STARPORT}
    ).ready:
        structure(AbilityId.SMART, rally)