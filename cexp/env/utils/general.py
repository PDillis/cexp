import re
import math
import numpy as np

import carla

import xml.etree.ElementTree as ET

from agents.navigation.global_route_planner import GlobalRoutePlanner
from agents.navigation.global_route_planner_dao import GlobalRoutePlannerDAO
from agents.navigation.local_planner import RoadOption


def tryint(s):
    try:
        return int(s)
    except (ValueError, TypeError):
        return s


def alphanum_key(s):
    """ Turn a string into a list of string and number chunks.
        "z23a" -> ["z", 23, "a"]
    """
    return [tryint(c) for c in re.split('([0-9]+)', s)]


def alphanum_key_dict(s):
    """ Turn a string into a list of string and number chunks.
        "z23a" -> ["z", 23, "a"]
    """
    return [tryint(c) for c in re.split('([0-9]+)', s[0])]


def sort_nicely(l):
    """ Sort the given list in the way that humans expect.
    """
    l.sort(key=alphanum_key)


def sort_nicely_dict(l):
    """ Sort the given list in the way that humans expect.
    """
    l = sorted(l, key=alphanum_key_dict)
    return l


def convert_json_to_transform(actor_dict):

    return carla.Transform(location=carla.Location(x=float(actor_dict['x']),
                                                   y=float(actor_dict['y']),
                                                   z=float(actor_dict['z'])),
                           rotation=carla.Rotation(roll=0.0, pitch=0.0, yaw=float(actor_dict['yaw'])))


def convert_transform_to_location(transform_vec):

    location_vec = []
    for transform_tuple in transform_vec:
        location_vec.append((transform_tuple[0].location, transform_tuple[1]))

    return location_vec


def distance_vehicle(waypoint, vehicle_position):

    dx = waypoint.location.x - vehicle_position.x
    dy = waypoint.location.y - vehicle_position.y

    return math.sqrt(dx * dx + dy * dy)


def get_forward_speed(vehicle):
    """ Convert the vehicle transform directly to forward speed """

    velocity = vehicle.get_velocity()
    transform = vehicle.get_transform()
    vel_np = np.array([velocity.x, velocity.y, velocity.z])
    pitch = np.deg2rad(transform.rotation.pitch)
    yaw = np.deg2rad(transform.rotation.yaw)
    orientation = np.array([np.cos(pitch) * np.cos(yaw), np.cos(pitch) * np.sin(yaw), np.sin(pitch)])
    speed = np.dot(vel_np, orientation)
    return speed


# Previously srunner.challenge
def _location_to_gps(lat_ref, lon_ref, location):
    """
    Convert from world coordinates to GPS coordinates
    :param lat_ref: latitude reference for the current map
    :param lon_ref: longitude reference for the current map
    :param location: location to translate
    :return: dictionary with lat, lon and height
    """
    EARTH_RADIUS_EQUATOR = 6378137.0   # pylint: disable=invalid-name
    scale = math.cos(lat_ref * math.pi / 180.0)
    mx = scale * lon_ref * math.pi * EARTH_RADIUS_EQUATOR / 180.0
    my = scale * EARTH_RADIUS_EQUATOR * math.log(math.tan((90.0 + lat_ref) * math.pi / 360.0))
    mx += location.x
    my += location.y

    lon = mx * 180.0 / (math.pi * EARTH_RADIUS_EQUATOR * scale)
    lat = 360.0 * math.atan(math.exp(my / (EARTH_RADIUS_EQUATOR * scale))) / math.pi - 90.0
    z = location.z

    return {'lat': lat, 'lon': lon, 'z': z}


def location_route_to_gps(route, lat_ref, lon_ref):
    """
        Locate each waypoint of the route into gps, (lat long ) representations.
    :param route:
    :param lat_ref:
    :param lon_ref:
    :return:
    """
    gps_route = []

    for transform, connection in route:
        gps_point = _location_to_gps(lat_ref, lon_ref, transform.location)
        gps_route.append((gps_point, connection))

    return gps_route


def _get_latlon_ref(world):
    """
    Convert from waypoints world coordinates to CARLA GPS coordinates
    :return: tuple with lat and lon coordinates
    """
    xodr = world.get_map().to_opendrive()
    tree = ET.ElementTree(ET.fromstring(xodr))

    # default reference
    lat_ref = 42.0
    lon_ref = 2.0

    for opendrive in tree.iter("OpenDRIVE"):
        for header in opendrive.iter("header"):
            for georef in header.iter("geoReference"):
                if georef.text:
                    str_list = georef.text.split(' ')
                    for item in str_list:
                        if '+lat_0' in item:
                            lat_ref = float(item.split('=')[1])
                        if '+lon_0' in item:
                            lon_ref = float(item.split('=')[1])
    return lat_ref, lon_ref


def clean_route(route):

    curves_start_end = []
    inside = False
    start = -1
    current_curve = RoadOption.LANEFOLLOW
    index = 0
    while index < len(route):

        command = route[index][1]
        if command != RoadOption.LANEFOLLOW and not inside:
            inside = True
            start = index
            current_curve = command

        if command != current_curve and inside:
            inside = False
            # End now is the index.
            curves_start_end.append([start, index, current_curve])
            if start == -1:
                raise ValueError("End of curve without start")

            start = -1
        else:
            index += 1

    return curves_start_end


def downsample_route(route, sample_factor):
    """
    Downsample the route by some factor.
    :param route: the trajectory , has to contain the waypoints and the road options
    :param sample_factor: the downsampling factor
    :return: returns the ids of the final route that can
    """
    route_size = len(route)

    turn_positions_and_labels = clean_route(route)
    ids_to_sample = []

    lane_follow_set = set(range(0, route_size))

    # we take all the positions that are actually non lane follow and downsample by the factor, sample_factor
    for start, end, conditions in turn_positions_and_labels:
        initial_condition_range = [x for x in range(start, end)]

        lane_follow_set = lane_follow_set - set(initial_condition_range)
        # now we resample the turn points
        ids_to_sample += initial_condition_range[::sample_factor]

    # We take all the lane following segments and reduce them
    ids_to_sample += list(lane_follow_set)[::sample_factor]
    ids_to_sample = sorted(ids_to_sample)

    return ids_to_sample


def interpolate_trajectory(world, waypoints_trajectory, hop_resolution=1.0):
    """
        Given some raw keypoints interpolate a full dense trajectory to be used by the user.
    :param world: an reference to the CARLA world so we can use the planner
    :param waypoints_trajectory: the current coarse trajectory
    :param hop_resolution: is the resolution, how dense is the provided trajectory going to be made
    :return: the full interpolated route both in GPS coordinates and also in its original form.
    """

    dao = GlobalRoutePlannerDAO(world.get_map(), hop_resolution)
    grp = GlobalRoutePlanner(dao)
    grp.setup()
    # Obtain route plan
    route = []
    for i in range(len(waypoints_trajectory) - 1):   # Goes until the one before the last.

        waypoint = waypoints_trajectory[i]
        waypoint_next = waypoints_trajectory[i + 1]
        interpolated_trace = grp.trace_route(waypoint, waypoint_next)
        for wp_tuple in interpolated_trace:
            route.append((wp_tuple[0].transform, wp_tuple[1]))

    # Increase the route position to avoid fails

    lat_ref, lon_ref = _get_latlon_ref(world)

    return location_route_to_gps(route, lat_ref, lon_ref), route
