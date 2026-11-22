from dataclasses import dataclass
from enum import Enum, auto

from ares.consts import EngagementResult
from sc2.position import Point2
from sc2.unit import Unit


class Mode(Enum):
    ATTACK = auto()
    DEFEND = auto()
    HOLD = auto()


@dataclass
class GroupOrders:
    label: str
    mode: Mode
    target: Point2
    hold_point: Point2
    front: Point2 = Point2((1.0, 0.0))
    bio_position: Point2 | None = None
    anchor: Point2 | None = None
    local_result: EngagementResult | None = None
    unit_results: dict[int, EngagementResult] | None = None
    retreating: bool = False
    staging: Point2 | None = None
    pausing: bool = False
    hold_radius: float = 3.5

    @property
    def aggressive(self) -> bool:
        return self.mode in (Mode.ATTACK, Mode.DEFEND)

    def advance_result(self, unit: Unit) -> EngagementResult | None:
        if self.unit_results is None:
            return self.local_result
        return self.unit_results.get(unit.tag)