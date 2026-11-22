from dataclasses import dataclass
from typing import Callable, List, Mapping, Optional

from sc2.data import Race
from sc2.ids.unit_typeid import UnitTypeId as U

from bot.consts import SKYTOSS_TYPES

KNOBS = (
    "marine_marauder_ratio", "max_tanks", "max_cyclones", "max_hellions", "max_ravens", "max_medivacs", "max_vikings",
    "max_liberators", "max_battlecruisers", "max_banshees", "prioritize_vikings", "raven_first", "priority_units",
    "turrets_per_base", "starport_now", "mech_focus",
)


@dataclass(frozen=True)
class Scouted:
    race: Race
    structures: Mapping[U, int]
    units: Mapping[U, int]

    def structure(self, *types: U) -> int:
        return sum(self.structures.get(t, 0) for t in types)

    def unit(self, *types: U) -> int:
        return sum(self.units.get(t, 0) for t in types)


def raise_to(advice, **floors) -> None:
    for name, value in floors.items():
        setattr(advice, name, max(getattr(advice, name), value))


def lower_to(advice, **ceilings) -> None:
    for name, value in ceilings.items():
        setattr(advice, name, min(getattr(advice, name), value))


def want_first(advice, unit: U) -> None:
    if unit not in advice.priority_units:
        advice.priority_units.append(unit)


@dataclass(frozen=True)
class Reaction:
    name: str
    race: Optional[Race]
    seen: Callable[[Scouted], bool]
    apply: Callable[[object], None]
    why: str


def _detection(advice) -> None:
    advice.raven_first = True
    advice.starport_now = True
    raise_to(advice, max_ravens=2, turrets_per_base=1)
    want_first(advice, U.RAVEN)


REACTIONS: List[Reaction] = [
    Reaction(
        "dark shrine", Race.Protoss, lambda s: s.structure(U.DARKSHRINE) > 0 or s.unit(U.DARKTEMPLAR) > 0, _detection,
        "Dark Templar are invisible: the Starport's first unit is a Raven (not a Banshee), the Starport comes early, a turret in every base",
    ),
    Reaction(
        "lurkers", Race.Zerg, lambda s: s.structure(U.LURKERDENMP) > 0 or s.unit(U.LURKERMP, U.LURKERMPBURROWED) > 0,
        lambda a: (_detection(a), raise_to(a, max_tanks=10, max_liberators=2)),
        "Burrowed Lurkers need a detector, and tanks and Liberators outrange them: Raven first, more tanks and Liberators",
    ),
    Reaction(
        "widow mines", Race.Terran, lambda s: s.unit(U.WIDOWMINE, U.WIDOWMINEBURROWED) > 0, _detection,
        "Burrowed mines are invisible until detected: Raven first",
    ),
    Reaction(
        "banshees", Race.Terran, lambda s: s.unit(U.BANSHEE) > 0,
        lambda a: raise_to(a, max_ravens=2, turrets_per_base=1, max_vikings=6),
        "An enemy Banshee may be cloaked: a turret in every base, a second Raven, Vikings",
    ),
    Reaction(
        "roach warren", Race.Zerg, lambda s: s.structure(U.ROACHWARREN) > 0 or s.unit(U.ROACH) >= 3,
        lambda a: (want_first(a, U.SIEGETANK), raise_to(a, max_tanks=10), lower_to(a, marine_marauder_ratio=0.6)),
        "Roaches are armored: Siege Tanks (money is held back for them, not spent on Marines) and more Marauders",
    ),
    Reaction(
        "hydralisk den", Race.Zerg, lambda s: s.structure(U.HYDRALISKDEN) > 0 or s.unit(U.HYDRALISK) >= 3,
        lambda a: (want_first(a, U.SIEGETANK), raise_to(a, max_tanks=8)),
        "Hydralisks die to splash at range: Siege Tanks",
    ),
    Reaction(
        "baneling nest", Race.Zerg, lambda s: s.structure(U.BANELINGNEST) > 0 or s.unit(U.BANELING) >= 3,
        lambda a: (raise_to(a, max_tanks=8), lower_to(a, marine_marauder_ratio=0.5)),
        "Banelings shred Marines (light) but not Marauders (armored): more Marauders, Siege Tanks",
    ),
    Reaction(
        "ultralisks", Race.Zerg, lambda s: s.structure(U.ULTRALISKCAVERN) > 0 or s.unit(U.ULTRALISK) >= 2,
        lambda a: (raise_to(a, max_tanks=10), lower_to(a, marine_marauder_ratio=0.4)),
        "Ultralisks are armored and armored-proof against Marines: Marauders and Siege Tanks",
    ),
    Reaction(
        "spire", Race.Zerg, lambda s: s.structure(U.SPIRE, U.GREATERSPIRE) > 0 or s.unit(U.MUTALISK) >= 3,
        lambda a: raise_to(a, max_liberators=2, max_cyclones=4, max_vikings=6, turrets_per_base=1, marine_marauder_ratio=0.85),
        "Mutalisks: Marines (they shoot up), a turret in every base, Cyclones, Liberators and Vikings",
    ),
    Reaction(
        "mutalisk flock", Race.Zerg, lambda s: s.unit(U.MUTALISK) >= 8, lambda a: raise_to(a, turrets_per_base=2),
        "A big Mutalisk flock: two turrets in every base",
    ),
    Reaction(
        "brood lords", Race.Zerg, lambda s: s.structure(U.GREATERSPIRE) > 0 or s.unit(U.BROODLORD) >= 2,
        lambda a: (raise_to(a, max_vikings=10), setattr(a, "prioritize_vikings", True)),
        "Brood Lords outrange everything on the ground: Vikings first",
    ),
    Reaction(
        "stargate", Race.Protoss, lambda s: s.structure(U.STARGATE) > 0,
        lambda a: raise_to(a, turrets_per_base=1),
        "A Stargate means Oracles: a turret in every mineral line",
    ),
    Reaction(
        "colossi", Race.Protoss, lambda s: s.structure(U.ROBOTICSBAY) > 0 or s.unit(U.COLOSSUS) > 0,
        lambda a: raise_to(a, max_vikings=6),
        "Colossi wipe out Marines and only Vikings outrange them",
    ),
    Reaction(
        "many colossi", Race.Protoss, lambda s: s.unit(U.COLOSSUS) >= 3,
        lambda a: (raise_to(a, max_vikings=10), setattr(a, "prioritize_vikings", True)),
        "Three Colossi or more: Vikings before anything else the Starport could make",
    ),
    Reaction(
        "battlecruisers", Race.Terran, lambda s: s.structure(U.FUSIONCORE) > 0 or s.unit(U.BATTLECRUISER) >= 2,
        lambda a: (raise_to(a, max_cyclones=4, max_vikings=10), setattr(a, "prioritize_vikings", True)),
        "Battlecruisers: Vikings first, Cyclones",
    ),
    Reaction(
        "skytoss", Race.Protoss, lambda s: s.structure(U.STARGATE) > 0 or s.unit(*SKYTOSS_TYPES) > 0,
        lambda a: lower_to(a, max_tanks=2, turrets_per_base=0),
        "Siege Tanks cannot shoot up: no more than 2 against skytoss. No turrets either.",
    ),
]


def react(scouted: Scouted, advice) -> List[str]:
    fired: List[str] = []
    for rule in REACTIONS:
        if (rule.race is None or rule.race == scouted.race) and rule.seen(scouted):
            rule.apply(advice)
            fired.append(rule.name)
    return fired