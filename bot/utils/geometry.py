import math
from collections.abc import Callable

import numpy as np

from cython_extensions.units_utils import cy_closer_than
from cython_extensions.geometry import cy_distance_to, cy_distance_to_squared, cy_towards

from sc2.position import Point2
from sc2.unit import Unit
from sc2.units import Units


def is_already_attacking(unit: Unit, target: Unit) -> bool:
    return unit.is_attacking and unit.order_target == target.tag


def is_already_attack_moving_to(unit: Unit, position: Point2, epsilon: float = 0.5) -> bool:
    return (
        unit.is_attacking
        and isinstance(unit.order_target, Point2)
        and cy_distance_to_squared(unit.order_target, position) <= epsilon * epsilon
    )


def is_already_moving_to(unit: Unit, position: Point2, epsilon: float = 0.5) -> bool:
    return (
        unit.is_moving
        and isinstance(unit.order_target, Point2)
        and cy_distance_to_squared(unit.order_target, position) <= epsilon * epsilon
    )


def plain_point(point) -> Point2:
    return Point2((float(point[0]), float(point[1])))


def segment_walkable(start: Point2, end: Point2, walkable: Callable[[Point2], bool], step: float = 0.75) -> bool:
    length = cy_distance_to(start, end)
    samples = max(1, int(math.ceil(length / step)))
    for i in range(1, samples + 1):
        t = i / samples
        if not walkable(Point2((start.x + (end.x - start.x) * t, start.y + (end.y - start.y) * t))):
            return False
    return True


def path_length(start, path) -> float | None:
    if path is None or len(path) == 0:
        return None
    length, x, y = 0.0, float(start[0]), float(start[1])
    for point in path:
        px, py = float(point[0]), float(point[1])
        length += math.hypot(px - x, py - y)
        x, y = px, py
    return length


def ground_free(grid: np.ndarray, x: float, y: float) -> bool:
    i, j = int(x), int(y)
    return 0 <= i < grid.shape[0] and 0 <= j < grid.shape[1] and grid[i, j] != np.inf


def reachable_from_ground(grid: np.ndarray, position, reach: float = 0.75) -> bool:
    x, y = float(position[0]), float(position[1])
    return any(
        ground_free(grid, x + dx, y + dy)
        for dx, dy in ((0.0, 0.0), (reach, 0.0), (-reach, 0.0), (0.0, reach), (0.0, -reach))
    )


def open_ground_near(grid: np.ndarray, point: Point2, max_distance: float, limit: int = 12) -> list[Point2]:
    x0, x1 = max(1, int(point.x - max_distance)), min(grid.shape[0] - 1, int(point.x + max_distance) + 1)
    y0, y1 = max(1, int(point.y - max_distance)), min(grid.shape[1] - 1, int(point.y + max_distance) + 1)
    if x1 <= x0 or y1 <= y0:
        return []
    free = grid[x0 - 1: x1 + 1, y0 - 1: y1 + 1] != np.inf
    room = free[1:-1, 1:-1] & free[:-2, 1:-1] & free[2:, 1:-1] & free[1:-1, :-2] & free[1:-1, 2:]
    xs, ys = np.nonzero(room)
    cx, cy = xs + x0 + 0.5, ys + y0 + 0.5
    gap = np.hypot(cx - point.x, cy - point.y)
    near = np.nonzero(gap <= max_distance)[0]
    near = near[np.argsort(gap[near], kind="stable")][:limit]
    return [Point2((float(cx[i]), float(cy[i]))) for i in near]


def terrain_view(heights: np.ndarray, a, b, tolerance: float = 8.0, step: float = 1.0) -> bool:
    ax, ay, bx, by = float(a[0]), float(a[1]), float(b[0]), float(b[1])
    rows, cols = heights.shape

    def height(x: float, y: float) -> float:
        return float(heights[min(max(int(y), 0), rows - 1), min(max(int(x), 0), cols - 1)])

    from_height, to_height = height(ax, ay), height(bx, by)
    if from_height + tolerance < to_height:
        return False
    ceiling = max(from_height, to_height) + tolerance
    length = math.hypot(bx - ax, by - ay)
    for i in range(1, int(length // step)):
        t = i * step / length
        if height(ax + (bx - ax) * t, ay + (by - ay) * t) > ceiling:
            return False
    return True


class Crowd:
    def __init__(self, units: Units):
        self._index = {u.tag: i for i, u in enumerate(units)}
        self._xy = np.array([u.position_tuple for u in units], dtype=float).reshape(-1, 2)

    def centre_of_neighbours(self, unit: Unit, radius: float) -> Point2 | None:
        i = self._index[unit.tag]
        gap = self._xy - self._xy[i]
        near = (gap * gap).sum(axis=1) < radius * radius
        near[i] = False
        if not near.any():
            return None
        x, y = self._xy[near].mean(axis=0)
        return Point2((float(x), float(y)))


def spread_out_point(
    unit: Unit,
    friendlies: Units,
    towards: Point2,
    radius: float = 1.25,
    nudge: float = 1.5,
    walkable: Callable[[Point2], bool] | None = None,
    crowd: Crowd | None = None,
) -> Point2:
    if crowd is not None:
        centroid = crowd.centre_of_neighbours(unit, radius)
        if centroid is None or cy_distance_to_squared(unit.position, towards) < radius * radius:
            return towards
    else:
        nearby = cy_closer_than(friendlies.tags_not_in({unit.tag}), radius, unit.position)
        if len(nearby) == 0 or cy_distance_to_squared(unit.position, towards) < radius * radius:
            return towards
        centroid = Point2((
            sum(f.position.x for f in nearby) / len(nearby),
            sum(f.position.y for f in nearby) / len(nearby),
        ))
    if cy_distance_to_squared(unit.position, centroid) < 0.01:
        return towards
    away_point: Point2 = Point2(cy_towards(unit.position, centroid, -nudge))
    hop: Point2 = Point2(cy_towards(away_point, towards, 3))
    if walkable is not None and not segment_walkable(unit.position, hop, walkable):
        return towards
    return hop