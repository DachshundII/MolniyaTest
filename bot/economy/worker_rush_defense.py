from sc2.ids.unit_typeid import UnitTypeId
from sc2.ids.ability_id import AbilityId
from sc2.unit import Unit
from sc2.units import Units
from sc2.position import Point2
from typing import FrozenSet, Set
from sc2.bot_ai import BotAI

from ares.consts import UnitRole
from ares.behaviors.macro import BuildStructure
from cython_extensions.units_utils import cy_closest_to, cy_sorted_by_distance_to
from cython_extensions.geometry import cy_distance_to_squared

WORKER_RUSH_HOME_RADIUS = 30.0
WORKER_RUSH_DANGER_RADIUS = 10.0
WORKER_RUSH_DANGER_RADIUS_SQ = WORKER_RUSH_DANGER_RADIUS * WORKER_RUSH_DANGER_RADIUS
WORKER_ATTACK_LEASH = 45.0
WORKER_ATTACK_LEASH_SQ = WORKER_ATTACK_LEASH * WORKER_ATTACK_LEASH
WORKER_RUSH_CLEAR_DELAY = 30.0


def emergency_wall(self: BotAI):
    if self.army_advisor.is_wall_closed():
        return
    if not self.visible_enemy_units or not self.structures:
        return
    closest_enemy_dist_sq = min(
        cy_distance_to_squared(cy_closest_to(e.position, self.structures).position, e.position)
        for e in self.visible_enemy_units
    )
    if closest_enemy_dist_sq > 64.0:
        return
    self.register_behavior(
        BuildStructure(
            base_location=self.start_location,
            structure_id=UnitTypeId.SUPPLYDEPOT,
            wall=True,
            max_on_route=1,
        )
    )
    self.register_behavior(
        BuildStructure(
            base_location=self.start_location,
            structure_id=UnitTypeId.BARRACKS,
            wall=True,
            max_on_route=1,
        )
    )


def are_we_worker_rushed(self: BotAI):
    enemies: Units = self.visible_enemy_units.of_type({UnitTypeId.PROBE, UnitTypeId.SCV, UnitTypeId.DRONE})
    if enemies.empty:
        return 0, None, 0, None

    home: Units = self.structures.filter(lambda s: self.townhalls.closest_distance_to(s) <= WORKER_RUSH_HOME_RADIUS)
    if home.empty:
        return 0, None, 0, None

    dangerous_units = 0
    pathable_units = 0
    closest, closest_gap_sq = None, WORKER_RUSH_DANGER_RADIUS_SQ
    closest_inside, closest_inside_gap_sq = None, WORKER_RUSH_DANGER_RADIUS_SQ

    for e in enemies:
        closest_home = cy_closest_to(e.position, home)
        gap_sq = cy_distance_to_squared(e.position, closest_home.position)
        if gap_sq < WORKER_RUSH_DANGER_RADIUS_SQ:
            dangerous_units += 1
            if gap_sq < closest_gap_sq:
                closest, closest_gap_sq = e, gap_sq
            if e.position3d.z + 0.01 < self.townhalls.first.position3d.z:
                pathable_units += 1
            elif gap_sq < closest_inside_gap_sq:
                closest_inside, closest_inside_gap_sq = e, gap_sq

    return (
        dangerous_units,
        closest.position if closest is not None else None,
        dangerous_units - pathable_units,
        closest_inside.position if closest_inside is not None else None,
    )


def counter_worker_rush(self: BotAI, w: int, pos: Point2 | None) -> bool:
    if (w < 3 and not self.worker_rushed) or w == 0 or pos is None:
        return False

    mfs: Units = self.mineral_field.closer_than(12, self.start_location)
    self.worker_rushed = True
    counter = 0

    for worker in cy_sorted_by_distance_to(self.workers, self.start_location):
        if worker.is_constructing_scv:
            continue

        if worker.health <= 6:
            if worker.is_carrying_minerals:
                worker(AbilityId.SMART, self.townhalls.first)
            elif mfs.amount > 0:
                worker(AbilityId.SMART, cy_closest_to(worker.position, mfs))
            continue

        if worker.weapon_cooldown > 5:
            if mfs.amount > 0:
                worker(AbilityId.SMART, cy_closest_to(worker.position, mfs))
            continue

        counter += 1
        if counter > w + 1:
            break
        worker.attack(pos)
        self.mediator.assign_role(tag=worker.tag, role=UnitRole.DEFENDING)

    return True


def pull_back_workers(self: BotAI):
    if not self.townhalls:
        return
    mfs: Units = self.mineral_field.closer_than(12, self.townhalls.first)
    if mfs.amount == 0:
        return
    for worker in self.workers.idle:
        self.mediator.assign_role(tag=worker.tag, role=UnitRole.GATHERING)
        worker.gather(cy_closest_to(worker.position, mfs))


def recall_far_attackers(self: BotAI):
    if self.mineral_field.empty or not self.townhalls:
        return
    for worker in self.workers:
        if not worker.is_attacking:
            continue
        closest_th = cy_closest_to(worker.position, self.townhalls)
        if cy_distance_to_squared(closest_th.position, worker.position) > WORKER_ATTACK_LEASH_SQ:
            self.mediator.assign_role(tag=worker.tag, role=UnitRole.GATHERING)
            worker.gather(cy_closest_to(closest_th.position, self.mineral_field))


def worker_rush_defense(self: BotAI):
    w, pos, enemies_inside_wall, inside_pos = are_we_worker_rushed(self)

    if self.army_advisor.is_wall_closed():
        w, pos = enemies_inside_wall, inside_pos

    if counter_worker_rush(self, w, pos):
        self.build_order_runner.set_build_completed()

    recall_far_attackers(self)

    if self.worker_rushed:
        if w == 0:
            if self.worker_rush_clear_since is None:
                self.worker_rush_clear_since = self.time
            elif self.time - self.worker_rush_clear_since > WORKER_RUSH_CLEAR_DELAY:
                self.worker_rushed = False
                self.worker_rush_clear_since = None
        else:
            self.worker_rush_clear_since = None

    if self.worker_rushed:
        emergency_wall(self)
        if w == 0:
            pull_back_workers(self)