import math
from dataclasses import replace
from typing import Dict, Iterable, List, NamedTuple, Optional, Set, Tuple

from loguru import logger

from cython_extensions.units_utils import cy_center, cy_closer_than, cy_closest_to, cy_sorted_by_distance_to
from cython_extensions.geometry import cy_distance_to_squared

from ares.consts import EngagementResult, UnitRole, UnitTreeQueryType
from sc2.data import Race
from sc2.ids.unit_typeid import UnitTypeId
from sc2.position import Point2
from sc2.unit import Unit
from sc2.units import Units

from bot.tactics.consts import (
    BANSHEE_TYPES,
    BIO_TYPES,
    CONTINUE_ATTACK_RESULT,
    CYCLONE_TYPES,
    DANGEROUS_STRUCTURES,
    DEDICATED_TYPES,
    ENEMY_NON_ARMY_TYPES,
    FIGHT_GHOST_MAX_AGE,
    GROUPED_FRACTION,
    KITE_IN_CONFIRM_SECONDS,
    KITE_IN_RESULT,
    LIBERATOR_TYPES,
    LOCAL_RETREAT_RESULT,
    MAX_FIGHTS_JUDGED,
    MEDIVAC_TYPES,
    NON_ARMY_TYPES,
    RAVEN_TYPES,
    REAPER_TYPES,
    SQUAD_RADIUS,
    STAGING_COOLDOWN,
    STAGING_TIMEOUT,
    START_ATTACK_RESULT,
    TANK_TYPES,
    UNCONFIRMED_RESULT,
    VIKING_TYPES,
)
from bot.tactics.context import ArmyContext
from bot.tactics.defense import BaseDefense
from bot.tactics.enemy_tracker import EnemyTracker
from bot.tactics.fight import FightEvaluator, Stance
from bot.tactics.local_fight import Fight, find_fights
from bot.tactics.orders import GroupOrders, Mode
from bot.tactics.positioning import Positioning
from bot.tactics.progress import ProgressWatch
from bot.tactics.scouting import HiddenBaseScouting
from bot.units.banshees import BansheeHarass
from bot.units.bio import BioController
from bot.units.common import attack_move
from bot.units.cyclone_raid import CycloneRaid
from bot.units.cyclones import CycloneController
from bot.units.generic import GenericController
from bot.units.liberators import LiberatorController
from bot.units.medivacs import MedivacController
from bot.units.ravens import RavenController
from bot.units.reapers import ReaperHarass
from bot.units.tanks import TankController
from bot.units.vikings import VikingController
from bot.utils.geometry import plain_point

MIN_PUSH_SUPPLY = 40
INTEL_MIN_SUPPLY = 10.0
INTEL_FRACTION = 0.5
RETREAT_CONFIRM_SECONDS = 2.0
RETREAT_HOLD_SECONDS = 12.0
KEEP_ATTACK_SUPPLY_FRACTION = 0.5
MIN_ARMY_TO_KEEP_ATTACKING = 12
RECALL_RADIUS = 80.0
RECALL_RADIUS_SQ = RECALL_RADIUS * RECALL_RADIUS
GIVE_UP_SECONDS = 180.0
ZERGLING_RUSH_STAY_DEFENSIVE_UNTIL = 240.0
MIN_PUSH_SUPPLY_VS_PROTOSS = 60
START_ATTACK_RESULT_VS_PROTOSS = EngagementResult.VICTORY_OVERWHELMING
GROUPED_FRACTION_VS_PROTOSS = 0.9
REGROUP_TAIL_DISTANCE = 45.0
REGROUP_TAIL_DISTANCE_SQ = REGROUP_TAIL_DISTANCE * REGROUP_TAIL_DISTANCE
REGROUP_TAIL_FRACTION = 0.12
REGROUP_MAX_SECONDS = 12.0
REGROUP_COOLDOWN = 25.0
REINFORCE_HOME_RADIUS = 25.0
REINFORCE_HOME_RADIUS_SQ = REINFORCE_HOME_RADIUS * REINFORCE_HOME_RADIUS
REINFORCE_WAVE_FRACTION = 0.2
REINFORCE_WAVE_SUPPLY = (8.0, 20.0)
REINFORCE_RELEASE_SECONDS = 20.0
DIVERSION_SQUAD_SIZE = 3
DIVERSION_MIN_MAIN_ARMY = 10
DIVERSION_ARRIVAL_RANGE = 10.0
DIVERSION_ARRIVAL_RANGE_SQ = DIVERSION_ARRIVAL_RANGE * DIVERSION_ARRIVAL_RANGE
RAVEN_ESCORT_PARTS = ("defense", "diversion")
RAVEN_ESCORT_MIN_UNITS = 3
RAVEN_ESCORT_RANGE = 60.0
RAVEN_ESCORT_RANGE_SQ = RAVEN_ESCORT_RANGE * RAVEN_ESCORT_RANGE
RAVEN_ESCORT_LEAD = 0.0
DANGEROUS_STRUCTURE_RADIUS_SQ = 400.0

MANAGED_ROLES = (
    UnitRole.ATTACKING, UnitRole.BASE_DEFENDER, UnitRole.CONTROL_GROUP_ONE, UnitRole.CONTROL_GROUP_TWO, UnitRole.SCOUTING,
    UnitRole.HARASSING_BANSHEE, UnitRole.HARASSING_REAPER, UnitRole.HARASSING,
)


class MainPlan(NamedTuple):
    orders: GroupOrders
    stragglers: Units
    straggler_orders: GroupOrders
    waiting: Units
    waiting_orders: GroupOrders


class ArmyManager:
    def __init__(self, ai):
        self.ai = ai
        self.tracker = EnemyTracker(ai)
        self.fight = FightEvaluator(ai)
        self.positioning = Positioning(ai)
        self.defense = BaseDefense(ai, self.fight, self.positioning)
        self.scouting = HiddenBaseScouting(ai)

        self.bio = BioController(ai)
        self.tanks = TankController(ai, self.positioning)
        self.cyclones = CycloneController(ai)
        self.cyclone_raid = CycloneRaid(ai, self.cyclones)
        self.medivacs = MedivacController(ai)
        self.ravens = RavenController(ai)
        self.vikings = VikingController(ai)
        self.liberators = LiberatorController(ai)
        self.generic = GenericController(ai)
        self.banshees = BansheeHarass(ai, self.positioning)
        self.reapers = ReaperHarass(ai)

        self._confident_since: Dict[int, float] = {}

        self.attacking: bool = False
        self.attack_start_supply: float = 0.0
        self.losing_since: Optional[float] = None
        self.retreating_until: float = 0.0
        self.anchor: Optional[Point2] = None
        self.global_result: Optional[EngagementResult] = None

        self.staging_since: Optional[float] = None
        self.staging_cooldown_until: float = 0.0

        self.grouped: bool = True
        self.regroup_point: Optional[Point2] = None
        self.regroup_until: float = 0.0
        self.regroup_cooldown_until: float = 0.0
        self.regroup_timeouts: int = 0
        self.wave_until: float = 0.0

        self.attack_watch = ProgressWatch()
        self.diversion_watch = ProgressWatch()

        self.diversion_tags: Set[int] = set()
        self.diversion_target: Optional[Point2] = None

        self._errors: Dict[str, int] = {}

    def on_unit_destroyed(self, tag: int) -> None:
        self.tracker.remove(tag)
        self._confident_since.pop(tag, None)

    async def update(self) -> None:
        ai = self.ai
        self.fight.begin_step()
        ctx = self._guard("context", self._begin_step)
        if ctx is None:
            self._guard("fallback", self._emergency_fallback)
            return

        self._guard("roles", self._sweep_roles, ctx)
        army = self._army_units()
        self._guard("prefetch", ctx.prefetch_near, army)

        role = ctx.mediator.get_units_from_role
        self._guard("scouting", self.scouting.update, ctx, role(role=UnitRole.ATTACKING))

        defense_groups, escalate_to = self._guard(
            "defense", self.defense.update, ctx, role(role=UnitRole.ATTACKING), role(role=UnitRole.BASE_DEFENDER)
        ) or ([], None)

        main_units: Units = role(role=UnitRole.ATTACKING)
        defenders: Units = role(role=UnitRole.BASE_DEFENDER)
        self._guard("fights", self._find_fights, ctx)

        for units, orders in defense_groups:
            orders.local_result, orders.unit_results = self._guard(
                "defense fights", self._fight_view, ctx, units, Stance.DEFENDING
            ) or (None, {})

        plan = self._guard("orders", self._main_orders, ctx, main_units, defenders, escalate_to)
        if plan is None:
            plan = MainPlan(
                self._hold_orders(ctx), Units([], ai), self._hold_orders(ctx),
                Units([], ai), self._hold_orders(ctx),
            )
        main_orders, stragglers, straggler_orders = plan.orders, plan.stragglers, plan.straggler_orders

        diversion = self._guard("diversion", self._manage_diversion, ctx, main_units, main_orders)
        main_units = role(role=UnitRole.ATTACKING)
        stragglers = stragglers.tags_in(main_units.tags)
        waiting = plan.waiting.tags_in(main_units.tags)

        groups: List[Tuple[str, Units, GroupOrders]] = []
        if stragglers:
            groups.append(("regroup", stragglers, straggler_orders))
        if waiting:
            groups.append(("waiting", waiting, plan.waiting_orders))

        apart = stragglers.tags | waiting.tags
        groups.append(("main", main_units.tags_not_in(apart) if apart else main_units, main_orders))

        for units, orders in defense_groups:
            groups.append(("defense", units, orders))
        if diversion is not None:
            groups.append(("diversion", diversion[0], diversion[1]))
        if ai.enemy_race == Race.Zerg:
            groups = self._share_ravens(groups)

        for name, units, orders in groups:
            if name == "escort":
                await self._aguard(name, self.ravens.control, units, orders, ctx, RAVEN_ESCORT_LEAD)
            else:
                await self._aguard(name, self._dispatch, units, orders, ctx)

        handled = {tag for _, units, _ in groups for tag in units.tags}
        orphans = (role(role=UnitRole.BASE_DEFENDER) | role(role=UnitRole.CONTROL_GROUP_ONE)).tags_not_in(handled)
        if orphans:
            ctx.mediator.batch_assign_role(tags=orphans.tags, role=UnitRole.ATTACKING)
            self.diversion_tags -= orphans.tags
            await self._aguard("orphans", self._dispatch, orphans, main_orders, ctx)

        self._guard("banshees", self._harass, self.banshees, role(role=UnitRole.HARASSING_BANSHEE), main_orders, ctx)
        self._guard("reapers", self._harass, self.reapers, role(role=UnitRole.HARASSING_REAPER), main_orders, ctx)
        self._guard("cyclone raid", self._raid, role(role=UnitRole.HARASSING), main_orders, ctx)

    def _begin_step(self) -> ArmyContext:
        return ArmyContext(self.ai, self.positioning)

    def _hold_orders(self, ctx: ArmyContext) -> GroupOrders:
        return GroupOrders(
            label="main", mode=Mode.HOLD, target=ctx.hold_point, hold_point=ctx.hold_point,
            front=ctx.front, bio_position=ctx.bio_position, anchor=ctx.hold_point,
        )

    def _log_error(self, name: str, e: BaseException) -> None:
        count = self._errors.get(name, 0) + 1
        self._errors[name] = count
        if count <= 3 or count % 200 == 0:
            logger.opt(exception=e).error(f"[army] {name} failed (occurrence {count})")

    def _guard(self, name: str, func, *args):
        try:
            return func(*args)
        except Exception as e:
            self._log_error(name, e)
            return None

    async def _aguard(self, name: str, func, *args):
        try:
            return await func(*args)
        except Exception as e:
            self._log_error(name, e)
            return None

    def _emergency_fallback(self) -> None:
        ai = self.ai
        try:
            target = self.positioning.attack_target() if self.positioning.is_maxed() else self.positioning.hold_point()
        except Exception:
            target = ai.start_location
        for unit in self._army_units():
            if unit.type_id == UnitTypeId.SIEGETANKSIEGED:
                continue
            attack_move(unit, target)

    def _army_units(self) -> Units:
        return self.ai.units.exclude_type(NON_ARMY_TYPES)

    def _sweep_roles(self, ctx: ArmyContext) -> None:
        mediator = ctx.mediator
        role_sets = mediator.get_unit_role_dict
        assigned: Dict[int, str] = {}

        for role in MANAGED_ROLES:
            for tag in role_sets.get(role.name, ()):
                assigned[tag] = role.name

        raiding = self._raiding()
        to_assign: List[Tuple[int, UnitRole]] = []
        for unit in self._army_units():
            wanted: Optional[UnitRole] = None
            current = assigned.get(unit.tag)
            if unit.type_id in BANSHEE_TYPES:
                if current not in (UnitRole.HARASSING_BANSHEE.name, UnitRole.SCOUTING.name):
                    wanted = UnitRole.HARASSING_BANSHEE
            elif unit.type_id in REAPER_TYPES:
                if current not in (UnitRole.HARASSING_REAPER.name, UnitRole.SCOUTING.name):
                    wanted = UnitRole.HARASSING_REAPER
            elif unit.type_id in CYCLONE_TYPES and current in (None, UnitRole.ATTACKING.name, UnitRole.HARASSING.name):
                wanted = UnitRole.HARASSING if raiding else UnitRole.ATTACKING
                if current == wanted.name:
                    wanted = None
            elif current is None:
                wanted = UnitRole.ATTACKING

            if wanted is not None:
                to_assign.append((unit.tag, wanted))

        for tag, role in to_assign:
            mediator.assign_role(tag=tag, role=role)

    def _raiding(self) -> bool:
        return self.ai.enemy_race == Race.Protoss and not self.attacking and not self.defense.tasks

    def _main_orders(
        self, ctx: ArmyContext, main_units: Units, defenders: Units, escalate_to: Optional[Point2]
    ) -> MainPlan:
        ai = self.ai
        now = ai.time
        hold = ctx.hold_point
        core = main_units.of_type(BIO_TYPES | TANK_TYPES | CYCLONE_TYPES)

        stragglers = Units([], ai)
        squads = ctx.mediator.get_squads(role=UnitRole.ATTACKING, squad_radius=SQUAD_RADIUS) if main_units else []
        main_squad = self._pick_main_squad(squads)

        if main_squad is not None:
            self.anchor = plain_point(main_squad.squad_position)
            in_squad = main_squad.tags
            stragglers = core.tags_not_in(in_squad)
        elif not main_units:
            self.anchor = None

        grouped = True
        if core:
            needed = GROUPED_FRACTION_VS_PROTOSS if ai.enemy_race == Race.Protoss else GROUPED_FRACTION
            grouped = (core.amount - stragglers.amount) / core.amount >= needed
        self.grouped = grouped

        fighters = main_units | defenders
        self.global_result = self._global_result(fighters)
        self._update_push_state(now, fighters, grouped, escalate_to, self._push_result(ctx, fighters))

        defend_target = None
        if escalate_to is not None and (
            self.anchor is None or cy_distance_to_squared(self.anchor, escalate_to) <= RECALL_RADIUS_SQ
        ):
            local = self.fight.evaluate(fighters, self._enemy_units_near(ctx, escalate_to, 25.0), stance=Stance.DEFENDING)
            if local >= EngagementResult.TIE:
                defend_target = escalate_to

        anchor = self.anchor if self.anchor is not None else hold
        staging: Optional[Point2] = None
        pausing = False
        fighting = bool(ctx.fights.fights_of(main_units))

        if defend_target is not None:
            mode, target = Mode.DEFEND, defend_target
        elif self.attacking:
            mode, target = Mode.ATTACK, self.positioning.attack_target()
            staging = self._staging(now, anchor, main_units)
            if staging is not None:
                target = staging
                self.attack_watch.reset()
            else:
                pause = self._regroup_pause(now, fighting, core, stragglers, anchor)
                if pause is not None:
                    target, anchor, pausing = pause, pause, True
                    self.attack_watch.reset()
                else:
                    target = self._unstick(now, anchor, target, main_units)
        else:
            mode, target = Mode.HOLD, hold
            self.regroup_point = None
            self.staging_since = None
            self.attack_watch.reset()

        if mode == Mode.DEFEND:
            self.attack_watch.reset()

        waiting = Units([], ai)
        if mode == Mode.ATTACK:
            waiting = self._waiting_at_home(now, main_units, anchor, hold)
            if waiting:
                stragglers = stragglers.tags_not_in(waiting.tags)

        stance = {Mode.HOLD: Stance.DEFENDING, Mode.ATTACK: Stance.ATTACKING}.get(mode, Stance.MEETING)
        local_result, unit_results = self._fight_view(ctx, main_units, stance)

        orders = GroupOrders(
            label="main", mode=mode, target=target, hold_point=hold, front=ctx.front, bio_position=ctx.bio_position,
            anchor=anchor, local_result=local_result, unit_results=unit_results,
            retreating=(mode == Mode.HOLD and now < self.retreating_until),
            staging=staging, pausing=pausing,
            hold_radius=max(3.5, 1.1 * math.sqrt(max(1, main_units.amount))),
        )
        regroup_orders = GroupOrders(
            label="regroup", mode=Mode.ATTACK if fighting and mode != Mode.HOLD else Mode.HOLD,
            target=anchor, hold_point=anchor, front=ctx.front, bio_position=anchor, anchor=anchor,
            local_result=None, retreating=False,
        )

        if mode == Mode.HOLD:
            stragglers = Units([], ai)

        waiting_orders = replace(self._hold_orders(ctx), label="waiting")
        return MainPlan(orders, stragglers, regroup_orders, waiting, waiting_orders)

    def _regroup_pause(
        self, now: float, fighting: bool, core: Units, stragglers: Units, anchor: Point2
    ) -> Optional[Point2]:
        tail = cy_closer_than(stragglers, REGROUP_TAIL_DISTANCE, anchor) if stragglers else []

        if self.regroup_point is not None:
            if now < self.regroup_until and not fighting and len(tail) >= max(2, 0.04 * core.amount):
                return self.regroup_point
            if now >= self.regroup_until and not fighting:
                self.regroup_timeouts += 1
            elif not fighting:
                self.regroup_timeouts = 0
            self.regroup_point = None
            self.regroup_cooldown_until = now + REGROUP_COOLDOWN * 2 ** min(3, max(0, self.regroup_timeouts - 1))
            return None

        if now < self.regroup_cooldown_until or fighting or len(tail) < max(3, REGROUP_TAIL_FRACTION * core.amount):
            return None

        self.regroup_point = anchor
        self.regroup_until = now + REGROUP_MAX_SECONDS
        logger.info(
            f"[army] the push is strung out ({len(tail)} of {core.amount} ground units are behind): "
            f"holding at {anchor} for the tail"
        )
        return anchor

    def _waiting_at_home(self, now: float, main_units: Units, anchor: Point2, hold: Point2) -> Units:
        ai = self.ai
        away = main_units.filter(
            lambda u: cy_distance_to_squared(u.position, hold) <= REINFORCE_HOME_RADIUS_SQ
            and cy_distance_to_squared(u.position, anchor) > REGROUP_TAIL_DISTANCE_SQ
        )
        if not away or now < self.wave_until:
            return Units([], ai)

        supply = sum(ai.calculate_supply_cost(u.type_id) for u in away)
        field = sum(ai.calculate_supply_cost(u.type_id) for u in cy_closer_than(main_units, REGROUP_TAIL_DISTANCE, anchor))
        lowest, highest = REINFORCE_WAVE_SUPPLY

        if supply >= min(highest, max(lowest, REINFORCE_WAVE_FRACTION * field)):
            self.wave_until = now + REINFORCE_RELEASE_SECONDS
            logger.info(f"[army] a wave of {supply:.0f} supply leaves home for the army ({field:.0f} supply out there)")
            return Units([], ai)

        return away

    def _unstick(self, now: float, anchor: Point2, target: Point2, units: Units) -> Point2:
        fighting = any(u.weapon_cooldown > 0 for u in units)
        if not self.attack_watch.stuck(now, anchor, target, fighting):
            return target

        logger.warning(
            f"[army] stuck at {anchor} on the way to {target}: giving that target up for {GIVE_UP_SECONDS:.0f}s"
        )
        self.positioning.give_up(target, GIVE_UP_SECONDS)
        self.attack_watch.reset()
        return self.positioning.attack_target()

    def _staging(self, now: float, anchor: Point2, main_units: Units) -> Optional[Point2]:
        if now < self.staging_cooldown_until:
            return None

        tanks = main_units.of_type(TANK_TYPES)
        point = self.positioning.staging_point(anchor, tanks.amount)

        if point is None:
            self.staging_since = None
            return None

        if self.staging_since is None:
            self.staging_since = now

        waited = now - self.staging_since
        dug_in = len(cy_closer_than(tanks.of_type({UnitTypeId.SIEGETANKSIEGED}), 10.0, point))
        ready = tanks.amount > 0 and dug_in >= max(2, int(0.6 * tanks.amount))

        if (ready and waited >= 4.0) or waited > STAGING_TIMEOUT:
            self.staging_since = None
            self.staging_cooldown_until = now + STAGING_COOLDOWN
            return None

        return point

    @staticmethod
    def _pick_main_squad(squads):
        if not squads:
            return None
        core = BIO_TYPES | TANK_TYPES | CYCLONE_TYPES
        return max(squads, key=lambda s: (sum(1 for u in s.squad_units if u.type_id in core), len(s.squad_units)))

    def _global_result(self, fighters: Units) -> Optional[EngagementResult]:
        known = self.tracker.known_army()
        if not known or not fighters:
            return None

        enemy: List[Unit] = list(known)
        structures = self.ai.enemy_structures

        if structures:
            target = self.positioning.attack_target()
            enemy.extend(
                s for s in structures.of_type(DANGEROUS_STRUCTURES)
                if cy_distance_to_squared(s.position, target) <= DANGEROUS_STRUCTURE_RADIUS_SQ
            )

        return self.fight.evaluate(fighters, enemy, stance=Stance.ATTACKING, cautious=True, cache_seconds=1.0)

    def _push_result(self, ctx: ArmyContext, fighters: Units) -> Optional[EngagementResult]:
        fights = ctx.fights.fights_of(fighters)
        if fights:
            result = self._judge(fights[0], Stance.ATTACKING)
            if result is not None:
                return result
        return self.global_result

    def _update_push_state(
        self, now: float, fighters: Units, grouped: bool, escalate_to: Optional[Point2],
        push_result: Optional[EngagementResult],
    ) -> None:
        ai = self.ai
        maxed = self.positioning.is_maxed()
        army_supply = float(ai.supply_army)
        result = self.global_result
        protoss = ai.enemy_race == Race.Protoss

        if not self.attacking:
            start = False
            if maxed:
                start = True
            elif ai.army_advisor.zergling_rushed and now < ZERGLING_RUSH_STAY_DEFENSIVE_UNTIL:
                pass
            elif (
                escalate_to is None and grouped
                and army_supply >= (MIN_PUSH_SUPPLY_VS_PROTOSS if protoss else MIN_PUSH_SUPPLY)
                and result is not None and now >= self.retreating_until
            ):
                intel_ok = self.tracker.total_seen_supply() >= max(INTEL_MIN_SUPPLY, INTEL_FRACTION * army_supply)
                start = intel_ok and result >= (START_ATTACK_RESULT_VS_PROTOSS if protoss else START_ATTACK_RESULT)

            if start and fighters:
                self.attacking = True
                self.attack_start_supply = army_supply
                self.losing_since = None
            return

        result = push_result
        stop = False

        if result is not None and result < CONTINUE_ATTACK_RESULT:
            self.losing_since = self.losing_since if self.losing_since is not None else now
            if now - self.losing_since >= RETREAT_CONFIRM_SECONDS and not maxed:
                stop = True
        else:
            self.losing_since = None

        if (
            self.attack_start_supply > 0
            and army_supply < KEEP_ATTACK_SUPPLY_FRACTION * self.attack_start_supply
            and (result is None or result < EngagementResult.VICTORY_CLOSE)
        ):
            stop = True

        if not maxed and army_supply < MIN_ARMY_TO_KEEP_ATTACKING:
            stop = True
        if not fighters:
            stop = True

        if stop:
            self.attacking = False
            self.losing_since = None
            self.retreating_until = now + RETREAT_HOLD_SECONDS

    def _enemy_units_near(self, ctx: ArmyContext, position: Point2, radius: float) -> Units:
        near = ctx.mediator.get_units_in_range(
            start_points=[position], distances=radius, query_tree=UnitTreeQueryType.AllEnemy,
        )[0]
        return Units([e for e in near if not e.is_memory and e.type_id not in ENEMY_NON_ARMY_TYPES], self.ai)

    def _fight_enemies(self) -> List[Unit]:
        ai = self.ai
        enemies = [e for e in ai.enemy_units if not e.is_memory or e.age <= FIGHT_GHOST_MAX_AGE]
        enemies.extend(ai.enemy_structures)
        return enemies

    def _find_fights(self, ctx: ArmyContext) -> None:
        role = ctx.mediator.get_units_from_role
        fighters = (
            role(role=UnitRole.ATTACKING)
            | role(role=UnitRole.BASE_DEFENDER)
            | role(role=UnitRole.CONTROL_GROUP_ONE)
        )
        own = self.fight.clean_own(fighters)
        enemy = self.fight.clean_enemy(self._fight_enemies(), any(u.is_flying for u in own), True)
        ctx.fights = find_fights(own, enemy)

    def _judge(self, fight: Fight, stance: Stance, cautious: bool = False) -> Optional[EngagementResult]:
        if fight.rank >= MAX_FIGHTS_JUDGED:
            return None
        return self.fight.evaluate(fight.own, fight.enemy, stance=stance, engaged=fight.engaged, cautious=cautious)

    def _fight_view(
        self, ctx: ArmyContext, units: Units, stance: Stance
    ) -> Tuple[Optional[EngagementResult], Dict[int, EngagementResult]]:
        fights = ctx.fights.fights_of(units)
        group = self._judge(fights[0], stance) if fights else None
        tags = units.tags
        now = self.ai.time
        per_unit: Dict[int, EngagementResult] = {}

        for fight in fights:
            if not fight.engaged:
                continue
            advance = self._judge(fight, Stance.ATTACKING, cautious=True)
            if advance is None:
                continue
            for unit in fight.own:
                if unit.tag in tags:
                    per_unit[unit.tag] = advance

        for tag in tags:
            advance = per_unit.get(tag)
            if advance is None or advance < KITE_IN_RESULT:
                self._confident_since.pop(tag, None)
            elif now - self._confident_since.setdefault(tag, now) < KITE_IN_CONFIRM_SECONDS:
                per_unit[tag] = UNCONFIRMED_RESULT

        return group, per_unit

    def _manage_diversion(
        self, ctx: ArmyContext, main_units: Units, main_orders: GroupOrders
    ) -> Optional[Tuple[Units, GroupOrders]]:
        ai = self.ai
        role = ctx.mediator.get_units_from_role
        squad: Units = role(role=UnitRole.CONTROL_GROUP_ONE)
        self.diversion_tags &= squad.tags

        def dissolve() -> None:
            if squad.tags:
                ctx.mediator.batch_assign_role(tags=squad.tags, role=UnitRole.ATTACKING)
            self.diversion_tags = set()
            self.diversion_target = None

        if self.diversion_target is not None and (not squad or main_orders.mode != Mode.ATTACK or squad.amount < 2):
            dissolve()
            self.diversion_watch.reset()
            return None

        if not squad:
            bio = main_units.of_type(BIO_TYPES)
            if main_orders.mode == Mode.ATTACK and bio.amount >= DIVERSION_MIN_MAIN_ARMY and self.diversion_target is None:
                target = self.positioning.diversion_target(main_orders.target)
                if target is not None:
                    chosen = cy_sorted_by_distance_to(bio, target)[:DIVERSION_SQUAD_SIZE]
                    for unit in chosen:
                        ctx.mediator.assign_role(tag=unit.tag, role=UnitRole.CONTROL_GROUP_ONE)
                    self.diversion_tags = {u.tag for u in chosen}
                    self.diversion_target = target
                    squad = role(role=UnitRole.CONTROL_GROUP_ONE)
            if not squad:
                return None

        target = self.diversion_target
        if target is None:
            dissolve()
            return None

        center = Point2(cy_center(squad))

        if cy_distance_to_squared(center, target) <= DIVERSION_ARRIVAL_RANGE_SQ and not cy_closer_than(ai.enemy_structures, 15.0, target):
            dissolve()
            self.diversion_watch.reset()
            return None

        if self.diversion_watch.stuck(ai.time, center, target, any(u.weapon_cooldown > 0 for u in squad)):
            logger.warning(f"[army] diversion squad stuck at {center} on the way to {target}: giving that target up")
            self.positioning.give_up(target, GIVE_UP_SECONDS)
            dissolve()
            self.diversion_watch.reset()
            return None

        local, unit_results = self._fight_view(ctx, squad, Stance.ATTACKING)

        if local is not None and local <= LOCAL_RETREAT_RESULT:
            anchor = main_orders.anchor if main_orders.anchor is not None else ctx.hold_point
            orders = GroupOrders(
                label="diversion", mode=Mode.HOLD, target=anchor, hold_point=anchor, front=ctx.front,
                bio_position=anchor, anchor=anchor, local_result=local, unit_results=unit_results, retreating=True,
            )
        else:
            orders = GroupOrders(
                label="diversion", mode=Mode.ATTACK, target=target, hold_point=ctx.hold_point,
                front=ctx.front, bio_position=ctx.bio_position, anchor=center, local_result=local,
                unit_results=unit_results,
            )

        return squad, orders

    def _share_ravens(self, groups: List[Tuple[str, Units, GroupOrders]]) -> List[Tuple[str, Units, GroupOrders]]:
        ravens: List[Unit] = [raven for _, units, _ in groups for raven in units.of_type(RAVEN_TYPES)]
        parts = [
            (units, orders) for name, units, orders in groups
            if name in RAVEN_ESCORT_PARTS and units.amount >= RAVEN_ESCORT_MIN_UNITS
        ]

        if len(ravens) < 2 or not parts:
            return groups

        main = next(
            ((units, orders) for name, units, orders in groups if name == "main" and units.of_type(RAVEN_TYPES)),
            None,
        )
        if main is not None:
            centre = main[1].anchor if main[1].anchor is not None else Point2(cy_center(main[0]))
            closest = cy_closest_to(centre, main[0].of_type(RAVEN_TYPES))
            ravens.remove(closest)

        escorts: List[Tuple[str, Units, GroupOrders]] = []
        for units, orders in sorted(parts, key=lambda part: -part[0].amount):
            squads = self.ai.mediator.get_squads(role=UnitRole.ATTACKING, squad_radius=RAVEN_ESCORT_RANGE)
            centre = orders.anchor if orders.anchor is not None else Point2(cy_center(units))
            near = [raven for raven in ravens if cy_distance_to_squared(raven.position, centre) <= RAVEN_ESCORT_RANGE_SQ]
            if not near:
                continue
            raven = cy_closest_to(centre, near)
            ravens.remove(raven)
            escorts.append(("escort", Units([raven], self.ai), replace(orders, label="escort", anchor=centre)))

        if not escorts:
            return groups

        gone = {raven.tag for _, units, _ in escorts for raven in units}
        return [(name, units.tags_not_in(gone), orders) for name, units, orders in groups] + escorts

    async def _dispatch(self, units: Units, orders: GroupOrders, ctx: ArmyContext) -> None:
        if not units:
            return
        self.bio.control(units.of_type(BIO_TYPES), orders, ctx)
        self.tanks.control(units.of_type(TANK_TYPES), orders, ctx)
        self.cyclones.control(units.of_type(CYCLONE_TYPES), orders, ctx)
        self.vikings.control(units.of_type(VIKING_TYPES), orders, ctx)
        self.liberators.control(units.of_type(LIBERATOR_TYPES), orders, ctx)
        self.medivacs.control(units.of_type(MEDIVAC_TYPES), orders, ctx)
        await self.ravens.control(units.of_type(RAVEN_TYPES), orders, ctx)
        self.generic.control(units.exclude_type(DEDICATED_TYPES), orders, ctx)

    def _raid(self, units: Units, main_orders: GroupOrders, ctx: ArmyContext) -> None:
        raiders = units.of_type(CYCLONE_TYPES)
        if raiders:
            self.cyclone_raid.control(raiders, main_orders, ctx)

    def _harass(self, controller, units: Units, main_orders: GroupOrders, ctx: ArmyContext) -> None:
        if not units:
            return
        controller.control(units, main_orders, ctx)