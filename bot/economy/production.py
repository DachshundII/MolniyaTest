from typing import List, Optional, Tuple

from sc2.bot_ai import BotAI
from sc2.ids.unit_typeid import UnitTypeId

from cython_extensions.general_utils import cy_unit_pending

_PRIORITY_MAKERS = {
    UnitTypeId.SIEGETANK: (UnitTypeId.FACTORY, True),
    UnitTypeId.CYCLONE: (UnitTypeId.FACTORY, True),
    UnitTypeId.RAVEN: (UnitTypeId.STARPORT, True),
    UnitTypeId.VIKINGFIGHTER: (UnitTypeId.STARPORT, False),
}

CYCLONES_PER_TANK = 3


def idle_producers(self: BotAI, unit_type: UnitTypeId):
    return self.structures(unit_type).ready.idle.tags_not_in(self.addons.just_ordered)


def factory_order(self: BotAI) -> List[UnitTypeId]:
    advisor = self.army_advisor
    tanks = self.units.of_type({UnitTypeId.SIEGETANK, UnitTypeId.SIEGETANKSIEGED}).amount + cy_unit_pending(self, UnitTypeId.SIEGETANK)
    cyclones = self.units(UnitTypeId.CYCLONE).amount + cy_unit_pending(self, UnitTypeId.CYCLONE)
    tank_due = not advisor.mech_focus or cyclones >= CYCLONES_PER_TANK * (tanks + 1)
    order = [UnitTypeId.SIEGETANK, UnitTypeId.CYCLONE] if tank_due else [UnitTypeId.CYCLONE, UnitTypeId.SIEGETANK]
    caps = {UnitTypeId.SIEGETANK: (tanks, advisor.max_tanks), UnitTypeId.CYCLONE: (cyclones, advisor.max_cyclones)}
    return [unit for unit in order if caps[unit][0] < caps[unit][1]]


def priority_reserve(self: BotAI) -> Optional[Tuple[UnitTypeId, int, int]]:
    advisor = self.army_advisor
    candidates = list(advisor.priority_units)
    if advisor.mech_focus and self.produce_from_factories:
        candidates.extend(unit for unit in factory_order(self) if unit not in candidates)
    for unit in candidates:
        if unit not in _PRIORITY_MAKERS:
            continue
        maker, needs_techlab = _PRIORITY_MAKERS[unit]
        if unit == UnitTypeId.SIEGETANK:
            wanted = advisor.max_tanks
            have = self.units.of_type({UnitTypeId.SIEGETANK, UnitTypeId.SIEGETANKSIEGED}).amount
            allowed = self.produce_from_factories
        elif unit == UnitTypeId.CYCLONE:
            wanted, have, allowed = advisor.max_cyclones, self.units(UnitTypeId.CYCLONE).amount, self.produce_from_factories
        elif unit == UnitTypeId.RAVEN:
            wanted, have, allowed = 1, self.units(UnitTypeId.RAVEN).amount, self.produce_from_starports
        else:
            wanted, have, allowed = advisor.max_vikings, self.units(UnitTypeId.VIKINGFIGHTER).amount, self.produce_from_starports
        if not allowed or have + cy_unit_pending(self, unit) >= wanted:
            continue
        idle = [b for b in idle_producers(self, maker) if b.has_techlab or not needs_techlab]
        cost = self.calculate_cost(unit)
        if not idle or self.vespene < cost.vespene or self.supply_left < self.calculate_supply_cost(unit):
            continue
        return unit, cost.minerals, cost.vespene
    return None


def produce(self: BotAI):
    reserve = priority_reserve(self)

    def afford(unit_type) -> bool:
        if not self.can_afford(unit_type):
            return False
        if reserve is None or reserve[0] == unit_type:
            return True
        cost = self.calculate_cost(unit_type)
        return self.minerals - cost.minerals >= reserve[1] and self.vespene - cost.vespene >= reserve[2]

    if self.produce_from_starports:
        for st in idle_producers(self, UnitTypeId.STARPORT):
            spoken_for = False
            for unit_type in ((UnitTypeId.RAVEN, UnitTypeId.BANSHEE) if self.army_advisor.raven_first else (UnitTypeId.BANSHEE, UnitTypeId.RAVEN)):
                banshee = unit_type == UnitTypeId.BANSHEE
                cap = self.army_advisor.max_banshees if banshee else self.army_advisor.max_ravens
                if st.has_techlab and self.units(unit_type).amount == 0 and cap > 0 and not (self.made_banshee if banshee else self.made_raven):
                    if afford(unit_type):
                        st.build(unit_type)
                        if banshee:
                            self.made_banshee = True
                        else:
                            self.made_raven = True
                    spoken_for = True
                    break
            if spoken_for:
                continue

            if self.army_advisor.prioritize_vikings and afford(UnitTypeId.VIKINGFIGHTER) and self.units(UnitTypeId.VIKINGFIGHTER).amount < self.army_advisor.max_vikings:
                st.build(UnitTypeId.VIKINGFIGHTER)
                if st.has_reactor and afford(UnitTypeId.VIKINGFIGHTER) and self.units(UnitTypeId.VIKINGFIGHTER).amount + 1 < self.army_advisor.max_vikings:
                    st.build(UnitTypeId.VIKINGFIGHTER)
                continue

            total_liberators = self.units(UnitTypeId.LIBERATOR).amount + self.units(UnitTypeId.LIBERATORAG).amount
            if st.has_techlab and afford(UnitTypeId.RAVEN) and self.units(UnitTypeId.RAVEN).amount < self.army_advisor.max_ravens:
                st.build(UnitTypeId.RAVEN)
            elif st.has_techlab and self.structures(UnitTypeId.FUSIONCORE).amount > 0 and afford(UnitTypeId.BATTLECRUISER) and self.units(UnitTypeId.BATTLECRUISER).amount < self.army_advisor.max_battlecruisers:
                st.build(UnitTypeId.BATTLECRUISER)
            elif afford(UnitTypeId.LIBERATOR) and total_liberators < self.army_advisor.max_liberators:
                st.build(UnitTypeId.LIBERATOR)
                if st.has_reactor and afford(UnitTypeId.LIBERATOR) and total_liberators + 1 < self.army_advisor.max_liberators:
                    st.build(UnitTypeId.LIBERATOR)
            elif st.has_techlab and afford(UnitTypeId.MEDIVAC) and self.units(UnitTypeId.MEDIVAC).amount < self.army_advisor.max_medivacs:
                st.build(UnitTypeId.MEDIVAC)
            elif st.has_techlab and afford(UnitTypeId.VIKINGFIGHTER) and self.units(UnitTypeId.VIKINGFIGHTER).amount < self.army_advisor.max_vikings:
                st.build(UnitTypeId.VIKINGFIGHTER)
            elif st.has_techlab and afford(UnitTypeId.BANSHEE) and self.units(UnitTypeId.BANSHEE).amount < self.army_advisor.max_banshees:
                st.build(UnitTypeId.BANSHEE)
            elif st.has_reactor and afford(UnitTypeId.MEDIVAC) and self.units(UnitTypeId.MEDIVAC).amount < self.army_advisor.max_medivacs:
                st.build(UnitTypeId.MEDIVAC)
                if afford(UnitTypeId.MEDIVAC) and self.units(UnitTypeId.MEDIVAC).amount + 1 < self.army_advisor.max_medivacs:
                    st.build(UnitTypeId.MEDIVAC)
            elif st.has_reactor and afford(UnitTypeId.VIKINGFIGHTER) and self.units(UnitTypeId.VIKINGFIGHTER).amount < self.army_advisor.max_vikings:
                st.build(UnitTypeId.VIKINGFIGHTER)
                if afford(UnitTypeId.VIKINGFIGHTER) and self.units(UnitTypeId.VIKINGFIGHTER).amount + 1 < self.army_advisor.max_vikings:
                    st.build(UnitTypeId.VIKINGFIGHTER)
            elif afford(UnitTypeId.MEDIVAC) and self.units(UnitTypeId.MEDIVAC).amount < self.army_advisor.max_medivacs:
                st.build(UnitTypeId.MEDIVAC)
            elif afford(UnitTypeId.VIKINGFIGHTER) and self.units(UnitTypeId.VIKINGFIGHTER).amount < self.army_advisor.max_vikings:
                st.build(UnitTypeId.VIKINGFIGHTER)

        for st in self.structures(UnitTypeId.STARPORT).ready:
            if not st.has_reactor:
                continue
            if len(st.orders) != 1:
                continue
            if self.army_advisor.prioritize_vikings and afford(UnitTypeId.VIKINGFIGHTER) and self.units(UnitTypeId.VIKINGFIGHTER).amount < self.army_advisor.max_vikings:
                st.build(UnitTypeId.VIKINGFIGHTER)
            elif afford(UnitTypeId.MEDIVAC) and self.units(UnitTypeId.MEDIVAC).amount < self.army_advisor.max_medivacs:
                st.build(UnitTypeId.MEDIVAC)
            elif afford(UnitTypeId.VIKINGFIGHTER) and self.units(UnitTypeId.VIKINGFIGHTER).amount < self.army_advisor.max_vikings:
                st.build(UnitTypeId.VIKINGFIGHTER)

    if self.produce_from_factories:
        for fac in idle_producers(self, UnitTypeId.FACTORY):
            if fac.has_techlab:
                for unit_type in factory_order(self):
                    if afford(unit_type):
                        fac.build(unit_type)
                        break
            elif fac.has_reactor:
                if afford(UnitTypeId.CYCLONE) and self.units(UnitTypeId.CYCLONE).amount < self.army_advisor.max_cyclones:
                    fac.build(UnitTypeId.CYCLONE)
                    if afford(UnitTypeId.CYCLONE) and self.units(UnitTypeId.CYCLONE).amount + 1 < self.army_advisor.max_cyclones:
                        fac.build(UnitTypeId.CYCLONE)
                elif afford(UnitTypeId.HELLION) and self.units(UnitTypeId.HELLION).amount < self.army_advisor.max_hellions:
                    fac.build(UnitTypeId.HELLION)
                    if afford(UnitTypeId.HELLION) and self.units(UnitTypeId.HELLION).amount + 1 < self.army_advisor.max_hellions:
                        fac.build(UnitTypeId.HELLION)
            elif afford(UnitTypeId.CYCLONE) and self.units(UnitTypeId.CYCLONE).amount < self.army_advisor.max_cyclones:
                fac.build(UnitTypeId.CYCLONE)
            elif afford(UnitTypeId.HELLION) and self.units(UnitTypeId.HELLION).amount < self.army_advisor.max_hellions:
                fac.build(UnitTypeId.HELLION)

        for fac in self.structures(UnitTypeId.FACTORY).ready:
            if not fac.has_reactor:
                continue
            if len(fac.orders) == 1:
                if afford(UnitTypeId.CYCLONE) and self.units(UnitTypeId.CYCLONE).amount < self.army_advisor.max_cyclones:
                    fac.build(UnitTypeId.CYCLONE)
                elif afford(UnitTypeId.HELLION) and self.units(UnitTypeId.HELLION).amount < self.army_advisor.max_hellions:
                    fac.build(UnitTypeId.HELLION)

    if self.produce_from_barracks:
        total_marines = self.units.of_type({UnitTypeId.MARINE}).amount
        total_marauders = self.units.of_type({UnitTypeId.MARAUDER}).amount

        if total_marauders != 0 and total_marines / (total_marines + total_marauders) < self.army_advisor.marine_marauder_ratio:
            for bar in idle_producers(self, UnitTypeId.BARRACKS):
                if bar.has_techlab and afford(UnitTypeId.MARINE):
                    bar.build(UnitTypeId.MARINE)
                elif bar.has_reactor and afford(UnitTypeId.MARINE):
                    bar.build(UnitTypeId.MARINE)
                    if afford(UnitTypeId.MARINE):
                        bar.build(UnitTypeId.MARINE)
                elif self.supply_army == 0 and (afford(UnitTypeId.REAPER) or (self.minerals >= 50 and self.vespene >= 40)):
                    bar.build(UnitTypeId.REAPER)
                elif afford(UnitTypeId.MARINE):
                    bar.build(UnitTypeId.MARINE)

            for bar in self.structures(UnitTypeId.BARRACKS).ready:
                if not bar.has_reactor:
                    continue
                if len(bar.orders) == 1 and afford(UnitTypeId.MARINE):
                    bar.build(UnitTypeId.MARINE)

        else:
            for bar in idle_producers(self, UnitTypeId.BARRACKS):
                if bar.has_techlab and afford(UnitTypeId.MARAUDER):
                    bar.build(UnitTypeId.MARAUDER)
                elif bar.has_reactor and afford(UnitTypeId.MARINE):
                    bar.build(UnitTypeId.MARINE)
                    if afford(UnitTypeId.MARINE):
                        bar.build(UnitTypeId.MARINE)
                elif self.supply_army == 0 and (afford(UnitTypeId.REAPER) or (self.minerals >= 50 and self.vespene >= 40)):
                    bar.build(UnitTypeId.REAPER)
                elif afford(UnitTypeId.MARINE):
                    bar.build(UnitTypeId.MARINE)

            for bar in self.structures(UnitTypeId.BARRACKS).ready:
                if not bar.has_reactor:
                    continue
                if len(bar.orders) == 1 and afford(UnitTypeId.MARINE):
                    bar.build(UnitTypeId.MARINE)