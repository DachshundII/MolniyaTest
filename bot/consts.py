from typing import Set

from ares.consts import CHANGELING_TYPES as ARES_CHANGELING_TYPES
from sc2.ids.unit_typeid import UnitTypeId


CHANGELING_TYPES: Set[UnitTypeId] = set(ARES_CHANGELING_TYPES)

ATTACK_TARGET_IGNORE: Set[UnitTypeId] = {
    UnitTypeId.LARVA,
    UnitTypeId.EGG,
}

HARMLESS_TO_WORKERS: Set[UnitTypeId] = ATTACK_TARGET_IGNORE | CHANGELING_TYPES | {
    UnitTypeId.OBSERVER,
    UnitTypeId.OBSERVERSIEGEMODE,
}

ATTACK_TARGET_IGNORE_WITH_WORKERS: Set[UnitTypeId] = ATTACK_TARGET_IGNORE | CHANGELING_TYPES | {
    UnitTypeId.SCV,
    UnitTypeId.PROBE,
    UnitTypeId.DRONE,
    UnitTypeId.MULE,
}

SKYTOSS_TYPES: Set[UnitTypeId] = {
    UnitTypeId.VOIDRAY,
    UnitTypeId.ORACLE,
    UnitTypeId.PHOENIX,
    UnitTypeId.CARRIER,
    UnitTypeId.TEMPEST,
    UnitTypeId.MOTHERSHIP,
}

DANGEROUS_STRUCTURES: Set[UnitTypeId] = {
    UnitTypeId.PHOTONCANNON,
    UnitTypeId.SHIELDBATTERY,
    UnitTypeId.SPINECRAWLER,
    UnitTypeId.SPORECRAWLER,
    UnitTypeId.MISSILETURRET,
    UnitTypeId.PLANETARYFORTRESS,
    UnitTypeId.BUNKER,
}