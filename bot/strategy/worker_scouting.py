from sc2.bot_ai import BotAI
from sc2.units import Units
from sc2.ids.unit_typeid import UnitTypeId

from ares.consts import UnitRole
from cython_extensions.units_utils import cy_closest_to
from cython_extensions.geometry import cy_distance_to_squared

from bot.consts import HARMLESS_TO_WORKERS
from bot.utils.geometry import is_already_moving_to

SCOUT_DANGER_RANGE = 10.0
SCOUT_DANGER_RANGE_SQ = SCOUT_DANGER_RANGE * SCOUT_DANGER_RANGE
SCOUT_GIVE_UP_TIME = 150.0


async def scout(self: BotAI) -> None:
    if self.enemy_base_scouted or (self.scout_attempted and self.scout_worker_tag is None):
        return

    if self.scout_worker_tag is None:
        barracks_placed = self.structures(UnitTypeId.BARRACKS).amount > 0 or self.already_pending(UnitTypeId.BARRACKS) > 0
        if self.time > SCOUT_GIVE_UP_TIME or not barracks_placed or self.worker_rushed:
            return

        candidates: Units = (self.workers.gathering | self.workers.idle).tags_not_in({self.build_order_critical_worker})
        if candidates.amount == 0:
            return

        empty_workers = candidates.filter(lambda w: not w.is_carrying_minerals and not w.is_carrying_vespene)

        if empty_workers.amount == 0:
            return

        self.scout_attempted = True
        self.scout_worker_tag = empty_workers.random.tag
        self.mediator.assign_role(tag=self.scout_worker_tag, role=UnitRole.SCOUTING)
        return

    worker = self.workers.find_by_tag(self.scout_worker_tag)
    if worker is None:
        self.scout_worker_tag = None
        return

    if self.is_visible(self.enemy_start_locations[0]):
        self.enemy_base_scouted = True
        self.mediator.assign_role(tag=self.scout_worker_tag, role=UnitRole.GATHERING)
        self.scout_worker_tag = None
        return

    danger: Units = self.visible_enemy_units.filter(
        lambda u: u.type_id not in {UnitTypeId.PROBE, UnitTypeId.SCV, UnitTypeId.DRONE}
        and u.type_id not in HARMLESS_TO_WORKERS
        and u.can_attack_ground
    )

    if danger.amount > 0:
        closest_danger = cy_closest_to(worker.position, danger)
        if cy_distance_to_squared(closest_danger.position, worker.position) < SCOUT_DANGER_RANGE_SQ:
            self.mediator.assign_role(tag=self.scout_worker_tag, role=UnitRole.GATHERING)
            self.scout_worker_tag = None
            worker.move(self.start_location)
            return

    if not is_already_moving_to(worker, self.enemy_start_locations[0]):
        worker.move(self.enemy_start_locations[0])