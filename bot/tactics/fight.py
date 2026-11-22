from dataclasses import dataclass
from enum import Enum, auto
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from loguru import logger

from ares.consts import EngagementResult
from sc2.ids.unit_typeid import UnitTypeId
from sc2.unit import Unit
from sc2.units import Units

from bot.tactics.consts import (
    ANTI_AIR_ONLY_STRUCTURES,
    DANGEROUS_STRUCTURES,
    DETACHMENT_RESULT,
    ENEMY_NON_ARMY_TYPES,
    ENEMY_WORKER_TYPES,
    NON_ARMY_TYPES,
)

MAX_SIM_CALLS_PER_STEP = 24
_CACHE_MAX_AGE = 3.0
_VICTORY = EngagementResult.VICTORY_EMPHATIC
_LOSS = EngagementResult.LOSS_EMPHATIC


class Stance(Enum):
    ATTACKING = auto()
    DEFENDING = auto()
    MEETING = auto()


@dataclass(frozen=True)
class SimSetup:
    timing_adjust: bool = False
    defender_player: int = 0
    good_positioning: bool = False


BASELINE = SimSetup()


class FightEvaluator:
    def __init__(self, ai):
        self.ai = ai
        self.sim_calls_this_step: int = 0
        self.total_sim_calls: int = 0
        self.total_fallbacks: int = 0
        self._cache: Dict[tuple, Tuple[float, EngagementResult]] = {}
        self._configured = False

    def begin_step(self) -> None:
        self.sim_calls_this_step = 0
        now = self.ai.time
        if self._cache and len(self._cache) > 64:
            self._cache = {k: v for k, v in self._cache.items() if now - v[0] <= _CACHE_MAX_AGE}

    def _configure_simulator(self) -> None:
        if self._configured:
            return
        self._configured = True
        try:
            sim = self.ai.manager_hub.combat_sim_manager.combat_sim
            sim.bad_micro(False)
            sim.enable_splash(True)
            sim.enable_surround_limits(True)
            sim.enable_melee_blocking(True)
        except Exception as e:
            logger.warning(f"[fight] could not configure combat simulator ({e!r}); using its defaults")

    def clean_own(self, own: Iterable[Unit]) -> List[Unit]:
        return [u for u in own if u.type_id not in NON_ARMY_TYPES and not u.is_structure]

    def clean_enemy(self, enemy: Iterable[Unit], own_has_air: bool, workers_do_no_damage: bool) -> List[Unit]:
        result: List[Unit] = []
        for u in enemy:
            type_id = u.type_id
            if u.is_hallucination:
                continue
            if u.is_structure:
                if type_id not in DANGEROUS_STRUCTURES or u.build_progress < 1:
                    continue
                if type_id in ANTI_AIR_ONLY_STRUCTURES and not own_has_air:
                    continue
                result.append(u)
                continue
            if type_id in ENEMY_NON_ARMY_TYPES and type_id not in ENEMY_WORKER_TYPES:
                continue
            if type_id in ENEMY_WORKER_TYPES and workers_do_no_damage:
                continue
            result.append(u)
        return result

    @staticmethod
    def plan(
        stance: Stance, engaged: bool, cautious: bool, own: Sequence[Unit], enemy: Sequence[Unit]
    ) -> List[Tuple[SimSetup, Sequence[Unit]]]:
        baseline = (BASELINE, own)
        if engaged or stance is Stance.MEETING:
            return [baseline]

        if stance is Stance.ATTACKING:
            mobile = [u for u in own if u.movement_speed > 0]
            if not mobile:
                return [baseline]
            model = (SimSetup(timing_adjust=True, defender_player=2, good_positioning=False), mobile)
        else:
            if any(u.movement_speed <= 0 for u in enemy):
                return [baseline]
            model = (SimSetup(timing_adjust=True, defender_player=1, good_positioning=True), own)

        return [baseline, model] if cautious else [model]

    def evaluate(
        self,
        own: Iterable[Unit],
        enemy: Iterable[Unit],
        *,
        stance: Stance = Stance.MEETING,
        engaged: bool = False,
        cautious: bool = False,
        workers_do_no_damage: bool = True,
        cache_seconds: float = 0.5,
    ) -> EngagementResult:
        own_list = self.clean_own(own)
        own_has_air = any(u.is_flying for u in own_list)
        enemy_list = self.clean_enemy(enemy, own_has_air, workers_do_no_damage)

        if not enemy_list:
            return _VICTORY
        if not own_list:
            return _LOSS

        runs = self.plan(stance, engaged, cautious, own_list, enemy_list)
        return min(
            self._simulate(mine, enemy_list, setup, workers_do_no_damage, cache_seconds)
            for setup, mine in runs
        )

    def _simulate(
        self, own: Sequence[Unit], enemy: Sequence[Unit], setup: SimSetup, workers_do_no_damage: bool, cache_seconds: float
    ) -> EngagementResult:
        key = (
            tuple(sorted(u.tag for u in own)),
            tuple(sorted(u.tag for u in enemy)),
            setup, workers_do_no_damage,
        )
        now = self.ai.time
        cached = self._cache.get(key)

        if cached is not None and now - cached[0] <= cache_seconds:
            return cached[1]

        if self.sim_calls_this_step >= MAX_SIM_CALLS_PER_STEP:
            if cached is not None:
                return cached[1]
            return self._fallback(own, enemy)

        self._configure_simulator()
        try:
            self.sim_calls_this_step += 1
            self.total_sim_calls += 1
            result = self.ai.mediator.can_win_fight(
                own_units=Units(list(own), self.ai),
                enemy_units=Units(list(enemy), self.ai),
                timing_adjust=setup.timing_adjust,
                good_positioning=setup.good_positioning,
                workers_do_no_damage=workers_do_no_damage,
            )
        except Exception as e:
            logger.warning(f"[fight] mediator.can_win_fight failed ({e!r}); using power-ratio fallback")
            return self._fallback(own, enemy)

        self._cache[key] = (now, result)
        return result

    def smallest_winning_subset(
        self,
        candidates: Sequence[Unit],
        enemy: Iterable[Unit],
        *,
        target: EngagementResult = DETACHMENT_RESULT,
        min_size: int = 1,
        **kwargs,
    ) -> Tuple[Optional[List[Unit]], EngagementResult]:
        enemy_list = list(enemy)
        n = len(candidates)
        if n == 0:
            return None, _LOSS

        everyone = self.evaluate(candidates, enemy_list, **kwargs)
        if everyone < target:
            return None, everyone

        lo, hi = max(1, min_size), n
        best_result = everyone

        while lo < hi:
            mid = (lo + hi) // 2
            result = self.evaluate(candidates[:mid], enemy_list, **kwargs)
            if result >= target:
                hi, best_result = mid, result
            else:
                lo = mid + 1

        return list(candidates[:max(hi, min_size)]), best_result

    @staticmethod
    def _power(unit: Unit) -> float:
        return max(unit.ground_dps, unit.air_dps) * (unit.health + unit.shield)

    def _fallback(self, own: Sequence[Unit], enemy: Sequence[Unit]) -> EngagementResult:
        self.total_fallbacks += 1
        own_power = sum(self._power(u) for u in own)
        enemy_power = sum(self._power(u) for u in enemy)

        if enemy_power <= 0:
            return _VICTORY

        ratio = own_power / enemy_power
        if ratio >= 4.0:
            return EngagementResult.VICTORY_EMPHATIC
        if ratio >= 2.5:
            return EngagementResult.VICTORY_OVERWHELMING
        if ratio >= 1.6:
            return EngagementResult.VICTORY_DECISIVE
        if ratio >= 1.15:
            return EngagementResult.VICTORY_CLOSE
        if ratio >= 0.85:
            return EngagementResult.TIE
        if ratio >= 0.6:
            return EngagementResult.LOSS_CLOSE
        if ratio >= 0.35:
            return EngagementResult.LOSS_DECISIVE
        return EngagementResult.LOSS_OVERWHELMING