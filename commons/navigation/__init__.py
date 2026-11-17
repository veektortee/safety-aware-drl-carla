"""CARLA 0.9.15 global route planner bundled into the project (MIT licensed, see file headers).

Exports:
    GlobalRoutePlanner(wmap, sampling_resolution) -> .trace_route(origin, destination)
        returns list[(carla.Waypoint, RoadOption)]. Building it walks the whole map topology,
        so build it once per map and reuse it.
    RoadOption: IntEnum VOID=-1, LEFT=1, RIGHT=2, STRAIGHT=3, LANEFOLLOW=4,
        CHANGELANELEFT=5, CHANGELANERIGHT=6.
"""

from commons.navigation.road_option import RoadOption
from commons.navigation.global_route_planner import GlobalRoutePlanner

__all__ = ["GlobalRoutePlanner", "RoadOption"]
