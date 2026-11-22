from ares.dicts.unit_data import UNIT_DATA

from sc2.ids.unit_typeid import UnitTypeId
from sc2.units import Units

_DEFAULT_SUPPLY = 2.0


class EnemyTracker:
    def __init__(self, ai):
        self.ai = ai
        self.killed_supply: float = 0.0

    def remove(self, tag: int) -> None:
        unit = self.ai._enemy_units_previous_map.get(tag)
        if unit is not None:
            self.killed_supply += self.supply_of(unit.type_id)

    def total_supply(self) -> float:
        return self.ai.get_total_supply(self.known_army())

    def total_seen_supply(self) -> float:
        return self.total_supply() + self.killed_supply

    def known_army(self) -> Units:
        return self.ai.enemy_units

    def supply_of(self, type_id: UnitTypeId) -> float:
        entry = UNIT_DATA.get(type_id)
        if entry is not None:
            return float(entry["supply"])
        try:
            return float(self.ai.calculate_supply_cost(type_id))
        except Exception:
            return _DEFAULT_SUPPLY