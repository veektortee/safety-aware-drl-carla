"""Small fakes of the CARLA objects used by the NAV modules (no CARLA server needed).

carla's plain value types (Location, Rotation, Transform, Vector3D, BoundingBox and the enums)
work offline and are used directly; Waypoint / Map / World / actors are faked here.
"""
import fnmatch
import math
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import carla  # noqa: E402

from commons.navigation.road_option import RoadOption  # noqa: E402
from commons.vehicle_model import DT, MAX_STEER_DEG, WHEELBASE, AccelModel  # noqa: E402


class FakeMarking:
    def __init__(self, mtype=carla.LaneMarkingType.Broken, lane_change=carla.LaneChange.Both):
        self.type = mtype
        self.lane_change = lane_change
        self.color = carla.LaneMarkingColor.White
        self.width = 0.15


class FakeWaypoint:
    def __init__(self, x, y, yaw_deg, lane_width=3.5, is_junction=False, road_id=1, lane_id=-1,
                 section_id=0, lane_type=carla.LaneType.Driving, lane_change=carla.LaneChange.NONE, z=0.0):
        self.transform = carla.Transform(carla.Location(x=x, y=y, z=z), carla.Rotation(yaw=yaw_deg))
        self.lane_width = lane_width
        self.is_junction = is_junction
        self.road_id = road_id
        self.lane_id = lane_id
        self.section_id = section_id
        self.lane_type = lane_type
        self.lane_change = lane_change
        self.left_lane_marking = FakeMarking()
        self.right_lane_marking = FakeMarking()
        self._left = None
        self._right = None

    def get_left_lane(self):
        return self._left

    def get_right_lane(self):
        return self._right

    def next(self, d):
        """Straight continuation of the lane (enough for lane-follow fallbacks)."""
        loc, yaw = self.transform.location, math.radians(self.transform.rotation.yaw)
        return [FakeWaypoint(loc.x + d * math.cos(yaw), loc.y + d * math.sin(yaw), self.transform.rotation.yaw,
                             self.lane_width, False, self.road_id, self.lane_id)]


def polyline_waypoints(points, spacing=2.0, **kw):
    """Waypoints every `spacing` m along a polyline through `points`, yaw = segment direction."""
    out = []
    for (x0, y0), (x1, y1) in zip(points[:-1], points[1:]):
        L = math.hypot(x1 - x0, y1 - y0)
        yaw = math.degrees(math.atan2(y1 - y0, x1 - x0))
        n = max(1, int(round(L / spacing)))
        for k in range(n):
            t = k / n
            out.append(FakeWaypoint(x0 + t * (x1 - x0), y0 + t * (y1 - y0), yaw, **kw))
    x0, y0 = points[-2]
    x1, y1 = points[-1]
    out.append(FakeWaypoint(x1, y1, math.degrees(math.atan2(y1 - y0, x1 - x0)), **kw))
    return out


def straight_waypoints(length=100.0, spacing=2.0, x0=0.0, y0=0.0, yaw_deg=0.0, **kw):
    """Waypoints along a straight line starting at (x0, y0) with heading yaw_deg."""
    n = int(round(length / spacing)) + 1
    c, s = math.cos(math.radians(yaw_deg)), math.sin(math.radians(yaw_deg))
    return [FakeWaypoint(x0 + i * spacing * c, y0 + i * spacing * s, yaw_deg, **kw) for i in range(n)]


def right_turn_route(l1=40.0, radius=10.0, l2=40.0, spacing=2.0, direction=1):
    """Straight along +x, a 90 deg turn (direction=+1 right / -1 left, in a junction), straight on.

    Returns (waypoints, options, turn_start_s, turn_end_s).
    """
    wps = straight_waypoints(l1, spacing, road_id=1)
    opts = [RoadOption.LANEFOLLOW] * len(wps)
    # circle centre to the right (+y) of the end of the first straight for a right turn
    cx, cy = l1, direction * radius
    arc = 0.5 * math.pi * radius
    n = int(math.ceil(arc / spacing))
    for k in range(1, n + 1):
        th = (k / n) * 0.5 * math.pi                     # heading change so far
        x = cx + radius * math.sin(th)
        y = cy - direction * radius * math.cos(th)
        yaw = math.degrees(direction * th)
        wps.append(FakeWaypoint(x, y, yaw, is_junction=k < n, road_id=100))
        opts.append(RoadOption.RIGHT if direction > 0 else RoadOption.LEFT)
    ex, ey = wps[-1].transform.location.x, wps[-1].transform.location.y
    yaw2 = 90.0 * direction
    tail = straight_waypoints(l2, spacing, ex, ey, yaw2, road_id=2)[1:]
    wps += tail
    opts += [RoadOption.LANEFOLLOW] * len(tail)
    return wps, opts, l1, l1 + arc


class FakeStraightMap:
    """A straight road along +x: lanes given as dicts(lane_id, y, yaw, lane_type, width, road_id).

    get_waypoint returns a FakeWaypoint at the lane centre with the location's x, or None when
    project_to_road is False and the point is on no lane of an accepted type.
    """

    def __init__(self, lanes, junction_x=None):
        self.lanes = lanes
        self.junction_x = junction_x      # (x_from, x_to) junction area, optional

    def _accepts(self, lane, lane_type):
        if lane_type is None:
            return lane["lane_type"] == carla.LaneType.Driving
        return bool(int(lane["lane_type"]) & int(lane_type))

    def _wp(self, lane, x):
        junc = self.junction_x is not None and self.junction_x[0] <= x <= self.junction_x[1]
        wp = FakeWaypoint(x, lane["y"], lane["yaw"], lane_width=lane["width"], is_junction=junc,
                          road_id=lane.get("road_id", 1), lane_id=lane["lane_id"],
                          lane_type=lane["lane_type"], lane_change=lane.get("lane_change", carla.LaneChange.NONE))
        return wp

    def get_waypoint(self, location, project_to_road=True, lane_type=carla.LaneType.Driving):
        cands = [ln for ln in self.lanes if self._accepts(ln, lane_type)]
        for ln in cands:
            if abs(location.y - ln["y"]) <= 0.5 * ln["width"]:
                return self._wp(ln, location.x)
        if not project_to_road or not cands:
            return None
        ln = min(cands, key=lambda q: abs(location.y - q["y"]))
        return self._wp(ln, location.x)


class FakeActor:
    def __init__(self, actor_id, type_id, x, y, yaw_deg=0.0, extent=(2.3, 1.0, 0.75), vel=(0.0, 0.0), z=0.0):
        self.id = actor_id
        self.type_id = type_id
        self._tf = carla.Transform(carla.Location(x=x, y=y, z=z), carla.Rotation(yaw=yaw_deg))
        self.bounding_box = carla.BoundingBox(carla.Location(0.0, 0.0, extent[2]),
                                              carla.Vector3D(extent[0], extent[1], extent[2]))
        self._vel = carla.Vector3D(vel[0], vel[1], 0.0)

    def get_transform(self):
        return self._tf

    def get_location(self):
        return self._tf.location

    def get_velocity(self):
        return self._vel

    def set_transform(self, tf):
        self._tf = tf


def walker(actor_id, x, y, vel=(0.0, 0.0)):
    return FakeActor(actor_id, "walker.pedestrian.0001", x, y, 0.0, (0.25, 0.25, 0.9), vel)


class FakeActorList(list):
    def filter(self, pattern):
        return FakeActorList(a for a in self if fnmatch.fnmatch(a.type_id, pattern))


class FakeTrafficLight:
    def __init__(self, actor_id, stop_waypoints, state=carla.TrafficLightState.Red):
        self.id = actor_id
        self.type_id = "traffic.traffic_light"
        self.state = state
        self._stops = stop_waypoints

    def get_state(self):
        return self.state

    def get_stop_waypoints(self):
        return self._stops


class FakeStopSign:
    def __init__(self, actor_id, x, y, yaw_deg=0.0, volume_center=(0.0, 0.0), extent=(1.5, 1.5, 1.0)):
        self.id = actor_id
        self.type_id = "traffic.stop"
        self._tf = carla.Transform(carla.Location(x=x, y=y, z=0.0), carla.Rotation(yaw=yaw_deg))
        self.trigger_volume = carla.BoundingBox(carla.Location(volume_center[0], volume_center[1], 0.0),
                                                carla.Vector3D(*extent))

    def get_transform(self):
        return self._tf


class FakeWorld:
    def __init__(self, actors=()):
        self.actors = FakeActorList(actors)

    def get_actors(self):
        return self.actors


class KinematicCar:
    """Kinematic bicycle (rear-axle state) + measured AccelModel, dt = DT.

    Like a CARLA vehicle, `transform` / `x, y` refer to the vehicle centre (WHEELBASE / 2 ahead of
    the rear axle).
    """

    def __init__(self, x=0.0, y=0.0, yaw_deg=0.0, v=0.0, pedal_delay_steps=0):
        self.yaw, self.v = math.radians(yaw_deg), v
        self._rx = x - 0.5 * WHEELBASE * math.cos(self.yaw)
        self._ry = y - 0.5 * WHEELBASE * math.sin(self.yaw)
        self.model = AccelModel()
        self.id = 9999
        self.type_id = "vehicle.tesla.model3"
        self._pedals = [0.0] * int(pedal_delay_steps)   # FIFO: longitudinal command delay

    @property
    def x(self):
        return self._rx + 0.5 * WHEELBASE * math.cos(self.yaw)

    @property
    def y(self):
        return self._ry + 0.5 * WHEELBASE * math.sin(self.yaw)

    @property
    def transform(self):
        return carla.Transform(carla.Location(x=self.x, y=self.y, z=0.0), carla.Rotation(yaw=math.degrees(self.yaw)))

    def get_transform(self):
        return self.transform

    def step(self, action):
        steer, u = float(action[0]), float(action[1])
        if self._pedals:
            self._pedals.append(u)
            u = self._pedals.pop(0)
        a = self.model.accel(u, max(self.v, 0.0))
        if self.v <= 0.05 and u <= 0.3:     # no rolling backwards / creeping at standstill
            a = max(a, 0.0) if u > 0.0 else 0.0
        delta = steer * math.radians(MAX_STEER_DEG)
        self.yaw += self.v * math.tan(delta) / WHEELBASE * DT
        self._rx += self.v * math.cos(self.yaw) * DT
        self._ry += self.v * math.sin(self.yaw) * DT
        self.v = max(0.0, self.v + a * DT)


def run_tests(namespace):
    """Plain-python runner: call every test_* function in `namespace`, report, exit code."""
    tests = [(k, f) for k, f in sorted(namespace.items()) if k.startswith("test_") and callable(f)]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"PASS {name}")
        except Exception as exc:  # noqa: BLE001
            failed += 1
            import traceback
            print(f"FAIL {name}: {exc!r}")
            traceback.print_exc()
    print(f"{len(tests) - failed}/{len(tests)} passed")
    return 1 if failed else 0
