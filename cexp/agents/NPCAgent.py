import logging
import numpy as NP
from cexp.agents.agent import Agent
from cexp.env.datatools.affordances import  get_driving_affordances
from cexp.env.scenario_identification import get_distance_closest_crossing_waker
from enum import Enum

import carla
from cexp.agents.local_planner import LocalPlanner

# TODO make a sub class for a non learnable agent

"""
    Interface for the CARLA basic npc agent.
"""

class AgentState(Enum):
    """
    AGENT_STATE represents the possible states of a roaming agent
    """
    NAVIGATING = 1
    BLOCKED_BY_VEHICLE = 2
    BLOCKED_RED_LIGHT = 3
    BLOCKED_BY_PEDESTRIAN = 4


class NPCAgent(Agent):

    def __init__(self, sensors_dict):
        super().__init__(self)
        self._sensors_dict = sensors_dict
        self._pedestrian_forbidden_distance = 10.0
        self._pedestrian_max_detected_distance = 50.0
        self._vehicle_forbidden_distance = 10.0
        self._vehicle_max_detected_distance = 50.0
        self._tl_forbidden_distance = 10.0
        self._tl_max_detected_distance = 50.0
        self._speed_detected_distance = 10.0

    def setup(self, config_file_path):
        self.route_assigned = False
        self._agent = None

        self._distance_pedestrian_crossing = -1
        self._closest_pedestrian_crossing = None

    # TODO we set the sensors here directly.
    def sensors(self):
        return self._sensors_dict

    def make_state(self, exp, target_speed = 20.0):
        """
            Based on the exp object it makes all the affordances.
        :param exp:
        :return:
        """
        self._vehicle = exp._ego_actor

        if self._agent is None:
            self._agent = True
            self._state = AgentState.NAVIGATING
            args_lateral_dict = {
                'K_P': 1,
                'K_D': 0.02,
                'K_I': 0,
                'dt': 1.0 / 20.0}
            self._local_planner = LocalPlanner(
                self._vehicle, opt_dict={'target_speed': target_speed,
                                         'lateral_control_dict': args_lateral_dict})
            self._hop_resolution = 2.0
            self._path_seperation_hop = 2
            self._path_seperation_threshold = 0.5
            self._grp = None

        if not self.route_assigned:
            plan = []
            for transform, road_option in exp._route:
                wp = exp._ego_actor.get_world().get_map().get_waypoint(transform.location)
                plan.append((wp, road_option))

            self._local_planner.set_global_plan(plan)
            self.route_assigned = True

        # TODO: are these necessary??
        """
        self._distance_pedestrian_crossing, self._closest_pedestrian_crossing = \
            get_distance_closest_crossing_waker(exp)

        # if the closest pedestrian dies we reset
        if self._closest_pedestrian_crossing is not None and \
                not self._closest_pedestrian_crossing.is_alive:
            self._closest_pedestrian_crossing = None
            self._distance_pedestrian_crossing = -1

        if self._distance_pedestrian_crossing != -1 and self._distance_pedestrian_crossing < 13.0:
            if self._distance_pedestrian_crossing < 4.5:
                self._local_planner.set_speed(0.0)
            else:
                self._local_planner.set_speed(self._distance_pedestrian_crossing / 4.5)
        """

        return get_driving_affordances(exp, self._pedestrian_forbidden_distance, self._pedestrian_max_detected_distance,
                                       self._vehicle_forbidden_distance, self._vehicle_max_detected_distance,
                                       self._tl_forbidden_distance, self._tl_max_detected_distance,
                                       self._local_planner.get_target_waypoint(),
                                       self._local_planner._default_target_speed, self._local_planner._target_speed, self._speed_detected_distance)


    def make_reward(self, exp):
        # Just basically return None since the reward is not used for a non

        return None

    def run_step(self, affordances):
        hazard_detected = False
        is_vehicle_hazard = affordances['is_vehicle_hazard']
        is_red_tl_hazard = affordances['is_red_tl_hazard']
        is_pedestrian_hazard = affordances['is_pedestrian_hazard']
        relative_angle = affordances['relative_angle']
        target_speed = affordances['target_speed']
        # once we meet a speed limit sign, the target speed changes

        #if target_speed != self._local_planner._target_speed:
        #    self._local_planner.set_speed(target_speed)
        forward_speed = affordances['forward_speed']
        
        if is_vehicle_hazard:
            self._state = AgentState.BLOCKED_BY_VEHICLE
            hazard_detected = True
        
        if is_red_tl_hazard:
            self._state = AgentState.BLOCKED_RED_LIGHT
            hazard_detected = True

        if is_pedestrian_hazard:
            self._state = AgentState.BLOCKED_BY_PEDESTRIAN
            hazard_detected = True

        if hazard_detected:
            control = self.emergency_stop()
        
        else:
            self._state = AgentState.NAVIGATING
            control = self._local_planner.run_step(relative_angle, target_speed)
            
        logging.debug("Output %f %f %f " % (control.steer,control.throttle, control.brake))

        return control


    def reinforce(self, rewards):
        """
        This agent cannot learn so there is no reinforce
        """
        pass

    def reset(self):
        print (" Correctly reseted the agent")
        self.route_assigned = False
        self._agent = None


    def emergency_stop(self):
        """
        Send an emergency stop command to the vehicle
        :return:
        """
        control = carla.VehicleControl()
        control.steer = 0.0
        control.throttle = 0.0
        control.brake = 1.0
        control.hand_brake = False

        return control
