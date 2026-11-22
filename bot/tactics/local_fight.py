from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial.distance import cdist

from cython_extensions.units_utils import cy_center
from sc2.position import Point2
from sc2.unit import Unit

from bot.tactics.consts import FIGHT_CONTACT_SECONDS

FASTER_SPEED = 1.4


@dataclass
class Fight:
    own: List[Unit]
    enemy: List[Unit]
    engaged: bool
    center: Point2
    rank: int = 0


class FightMap:
    def __init__(self, fights: Sequence[Fight] = ()):
        self.fights: List[Fight] = sorted(fights, key=lambda f: -len(f.own))
        for rank, fight in enumerate(self.fights):
            fight.rank = rank
        self._by_tag: Dict[int, Fight] = {u.tag: f for f in self.fights for u in f.own}

    def fights_of(self, units: Iterable[Unit]) -> List[Fight]:
        found = {}
        for unit in units:
            fight = self._by_tag.get(unit.tag)
            if fight is not None:
                found[fight.rank] = fight
        return [found[rank] for rank in sorted(found)]


class _Side:
    def __init__(self, units: Sequence[Unit]):
        self.position = np.array([u.position_tuple for u in units], dtype=float).reshape(-1, 2)
        self.radius = np.array([u.radius for u in units], dtype=float)
        self.speed = np.array([u.movement_speed * FASTER_SPEED for u in units], dtype=float)
        self.ground_range = np.array([u.ground_range for u in units], dtype=float)
        self.air_range = np.array([u.air_range for u in units], dtype=float)
        self.can_ground = np.array([u.can_attack_ground for u in units], dtype=bool)
        self.can_air = np.array([u.can_attack_air for u in units], dtype=bool)
        self.flying = np.array([u.is_flying for u in units], dtype=bool)


def _seconds_to_weapons(attackers: _Side, targets: _Side) -> np.ndarray:
    gap = cdist(attackers.position, targets.position) - attackers.radius[:, None] - targets.radius[None, :]
    flying = targets.flying[None, :]
    reach = np.where(flying, attackers.air_range[:, None], attackers.ground_range[:, None])
    can_hit = np.where(flying, attackers.can_air[:, None], attackers.can_ground[:, None])
    walk = np.maximum(gap - reach, 0.0)

    with np.errstate(divide="ignore", invalid="ignore"):
        seconds = np.where(walk <= 0.0, 0.0, np.where(attackers.speed[:, None] > 0.0, walk / attackers.speed[:, None], np.inf))
    seconds[~can_hit] = np.inf
    return seconds


def find_fights(own: Sequence[Unit], enemy: Sequence[Unit], horizon: float = FIGHT_CONTACT_SECONDS) -> FightMap:
    if not own or not enemy:
        return FightMap()

    ours, theirs = _Side(own), _Side(enemy)
    n, m = len(own), len(enemy)
    own_to_enemy = _seconds_to_weapons(ours, theirs)
    enemy_to_own = _seconds_to_weapons(theirs, ours).T
    seconds = np.minimum(own_to_enemy, enemy_to_own)
    linked = seconds <= horizon
    rows, cols = np.nonzero(linked)

    if not len(rows):
        return FightMap()

    graph = coo_matrix((np.ones(len(rows)), (rows, n + cols)), shape=(n + m, n + m))
    _, labels = connected_components(graph, directed=False)
    own_labels, enemy_labels = labels[:n], labels[n:]
    fights: List[Fight] = []

    for label in np.unique(own_labels[linked.any(axis=1)]):
        mine = np.nonzero(own_labels == label)[0]
        opposing = np.nonzero(enemy_labels == label)[0]
        block = np.ix_(mine, opposing)
        engaged = bool((own_to_enemy[block] <= 0.0).any() and (enemy_to_own[block] <= 0.0).any())
        members = [own[i] for i in mine]
        center = Point2(cy_center(members))
        fights.append(Fight(own=members, enemy=[enemy[j] for j in opposing], engaged=engaged, center=center))

    return FightMap(fights)