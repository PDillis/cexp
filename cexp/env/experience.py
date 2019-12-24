import carla
import math
import os
import numpy as np
import py_trees
import traceback
import time
import logging


from srunner.scenariomanager.timer import GameTime, TimeOut
from srunner.scenariomanager.carla_data_provider import CarlaActorPool, CarlaDataProvider
from srunner.tools.config_parser import ActorConfigurationData, ScenarioConfiguration
from srunner.scenarios.master_scenario import MasterScenario
from srunner.scenarios.background_activity import BackgroundActivity

from srunner.scenarios.object_crash_vehicle import DynamicObjectCrossing
from srunner.scenarios.object_crash_intersection import VehicleTurningRight, VehicleTurningLeft
from srunner.challenge.utils.route_manipulation import interpolate_trajectory, _get_latlon_ref

from cexp.env.scorer import record_route_statistics_default, get_current_completion
from cexp.env.scenario_identification import distance_to_intersection, get_current_road_angle, get_distance_closest_crossing_waker

from cexp.env.datatools import affordances

from agents.navigation.local_planner import RoadOption
from cexp.env.datatools.data_writer import Writer

from cexp.env.sensors.sensor_interface import CANBusSensor, CallBack, SensorInterface


number_class_translation = {

    "Scenario1": [None],
    "Scenario2": [None],
    "Scenario3": [DynamicObjectCrossing],
    "Scenario4": [VehicleTurningRight, VehicleTurningLeft],
    "Scenario5": [None],
    "Scenario6": [None],
    "Scenario7": [None],
    "Scenario8": [None],
    "Scenario9": [None],
    "Scenario10": [None]

}


def convert_json_to_transform(actor_dict):

    return carla.Transform(location=carla.Location(x=float(actor_dict['x']), y=float(actor_dict['y']),
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

# TODO this is actually a benchmark paramter .... either seconds or seconds per meter.

SECONDS_GIVEN_PER_METERS = 0.8

def estimate_route_timeout(route):
    route_length = 0.0  # in meters
    prev_point = route[0][0]
    for current_point, _ in route[1:]:
        dist = current_point.location.distance(prev_point.location)
        route_length += dist
        prev_point = current_point

    #print (" final time ", SECONDS_GIVEN_PER_METERS * route_length)

    return int(SECONDS_GIVEN_PER_METERS * route_length)

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



class Experience(object):

    def __init__(self, client, vehicle_model, route, sensors, scenario_definitions,
                 exp_params, agent_name):
        """
        The experience is like a instance of the environment
         contains all the objects (vehicles, sensors) and scenarios of the the current experience
        :param vehicle_model: the model that is going to be used to spawn the ego CAR
        """

        # We save the agent name for data savings
        self._agent_name = agent_name

        # save all the experiment parameters to be used later
        self._exp_params = exp_params
        # carla recorder mode save the full carla logs to do some replays
        if self._exp_params['carla_recording']:
            client.start_recorder('env_{}_number_{}_batch_{:0>4d}.log'.format(self._exp_params['env_name'],
                                                                              self._exp_params['env_number'],
                                                                              self._exp_params['exp_number']))
        # this parameter sets all the sensor threads and the main thread into saving data
        self._save_data = exp_params['save_data']
        # we can also toogle if we want to save sensors or not.
        self._save_sensors = exp_params['save_sensors']
        # Start objects that are going to be created
        self.world = None
        # You try to reconnect a few times.
        self.MAX_CONNECTION_ATTEMPTS = 7
        # Scenario definitions to perform the scenario building
        self.scenario_definitions = scenario_definitions
        self._ego_actor = None
        self._instanced_sensors = []
        # set the client object connected to the
        self._client = client
        # We also set the town name to be used
        self._town_name = exp_params['town_name']

        self._vehicle_model = vehicle_model
        # if data is being saved we create the writer object

        # if we are going to save, we keep track of a dictionary with all the data
        self._writer = Writer(exp_params['package_name'], exp_params['env_name'], exp_params['env_number'],
                              exp_params['exp_number'], agent_name,
                              other_vehicles=exp_params['save_opponents'])
        self._environment_data = {'exp_measurements': None,  # The exp measurements are specific of the experience
                                  'ego_controls': None,
                                  'scenario_controls': None}
        # identify this exp
        self._exp_id = self._exp_params['exp_number']

        # We try running all the necessary initalization, if we fail we clean the
        try:
            # Sensor interface, a buffer that contains all the read sensors
            self._sensor_interface = SensorInterface(number_threads_barrier=len(sensors))
            # Load the world
            self._load_world()
            # Set the actor pool so the scenarios can prepare themselves when needed
            CarlaActorPool.set_client(client)
            CarlaActorPool.set_world(self.world)
            # Set the world for the global data provider
            CarlaDataProvider.set_world(self.world)
            # We get the lat lon ref that is important for the route
            self._lat_ref, self._lon_ref = _get_latlon_ref(self.world)
            # We instance the ego actor object
            _, self._route = interpolate_trajectory(self.world, route)
            # elevate the z transform to avoid spawning problems
            elevate_transform = self._route[0][0]
            elevate_transform.location.z += 0.5
            self._spawn_ego_car(elevate_transform)
            # We setup all the instanced sensors
            self._setup_sensors(sensors, self._ego_actor)
            # We set all the traffic lights to green to avoid having this traffic scenario.
            self._reset_map()
            # Data for building the master scenario
            self._timeout = estimate_route_timeout(self._route)
            self._master_scenario = self.build_master_scenario(self._route,
                                                               exp_params['town_name'],
                                                               self._timeout)
            other_scenarios = self.build_scenario_instances(scenario_definitions, self._timeout)
            self._list_scenarios = [self._master_scenario] + other_scenarios
            # Route statistics, when the route is finished there will
            # be route statistics on this object. and nothing else
            self._route_statistics = None
            # We tick the world to have some starting points
            self.tick_world()
        except RuntimeError as r:
            # We clean the dataset if there is any exception on creation
            traceback.print_exc()
            if self._save_data:
                self._clean_bad_dataset()
            # Re raise the exception
            raise r


    def tick_scenarios(self):

        # We tick the scenarios to get them started
        for scenario in self._list_scenarios:
            scenario.scenario.scenario_tree.tick_once()

    def get_status(self):
        """
            Returns the current status of the vehicle
        """
        if self._master_scenario is None:
            raise ValueError('You should not run a route without a master scenario')

        if self._master_scenario.scenario.scenario_tree.status == py_trees.common.Status.INVALID:
            logging.debug("Exp No:{} The current scenario is INVALID".format(self._exp_id))
            status = 'INVALID'
        elif self._master_scenario.scenario.scenario_tree.status == py_trees.common.Status.SUCCESS:
            logging.debug("Exp No:{} The current scenario is SUCCESSFUL".format(self._exp_id))
            status = 'SUCCESS'
        elif self._master_scenario.scenario.scenario_tree.status == py_trees.common.Status.FAILURE:
            logging.debug("Exp No:{} The current scenario is FAILURE".format(self._exp_id))
            status = 'FAILURE'
        else:
            status = 'RUNNING'

        return status


    def tick_scenarios_control(self, controls):
        """
        Here we tick the scenarios and also change the control based on the scenario properties
        """

        GameTime.on_carla_tick(self.world.get_snapshot().timestamp)
        CarlaDataProvider.on_carla_tick()
        #print ("Timeout ", self._timeout,  " Timestamp ", self.world.get_snapshot().timestamp)
        # update all scenarios
        for scenario in self._list_scenarios:  #
            scenario.scenario.scenario_tree.tick_once()
            controls = scenario.change_control(controls)

        if self._save_data:
            self._environment_data['ego_controls'] = controls

        return controls


    def apply_control(self, controls):

        if self._save_data:
            self._environment_data['scenario_controls'] = controls
        self._ego_actor.apply_control(controls)

        if self._exp_params['debug']:
            spectator = self.world.get_spectator()
            ego_trans = self._ego_actor.get_transform()
            spectator.set_transform(carla.Transform(ego_trans.location + carla.Location(z=50),
                                                    carla.Rotation(pitch=-90)))


    def tick_world(self):
        # Save all the measurements that are interesting
        # TODO this may go to another function
        # TODO maybe add not on every iterations, identify every second or half second.
        # TODO this may be requiried even if no data is saved

        actor_list = self.world.get_actors()  # we get all objects in this world
        vehicle_list = actor_list.filter("*vehicle*")  # vehicle objects
        tl_list = actor_list.filter("*traffic_light*")  # traffic light objects
        pedestrian_list = actor_list.filter("*pedestrian*")  # pedestrian objects
        if self._save_data:
            closest_waypoint, directions = self._get_current_wp_direction(self._ego_actor.get_transform().location,
                                                           self._route)

            dist_scenario3, _ = get_distance_closest_crossing_waker(self)

            # HERE we may adapt the npc to stop dist_scenario3

            self._environment_data['exp_measurements'] = {
                'directions': directions,
                'distance_intersection': distance_to_intersection(self._ego_actor,
                                                                  self._ego_actor.get_world().get_map()),
                'road_angle': get_current_road_angle(self._ego_actor,
                                                     self._ego_actor.get_world().get_map()),
                'distance_crossing_walker': dist_scenario3,
                'distance_closest_scenario4': -1
            }


        self._sync(self.world.tick())

    def _sync(self, frame):
        while frame > self.world.get_snapshot().timestamp.frame:
            pass
        assert frame == self.world.get_snapshot().timestamp.frame
        self.frame = frame

    def save_experience(self):

        if self._save_data:
            self._sensor_interface.wait_sensors_written(self._writer)
            self._writer.save_experience(self.world, self._environment_data)

    def is_running(self):
        """
            The master scenario tests if the route is still running for this experiment
        """
        if self._master_scenario is None:
            raise ValueError('You should not run a route without a master scenario')

        return self._master_scenario.scenario.scenario_tree.status == py_trees.common.Status.RUNNING \
                or self._master_scenario.scenario.scenario_tree.status == py_trees.common.Status.INVALID

    """
        FUNCTIONS FOR BUILDING 
    """

    def _spawn_ego_car(self, start_transform):
        """
        Spawn or update all scenario actors according to
        a certain start position.
        """
        # If ego_vehicle already exists, just update location
        # Otherwise spawn ego vehicle
        self._ego_actor = CarlaActorPool.request_new_actor(self._vehicle_model, start_transform,
                                                           hero=True)

        CarlaDataProvider.set_ego_vehicle_route(
            convert_transform_to_location(self._route))
        logging.debug("Created Ego Vehicle")


    def _setup_sensors(self, sensors, vehicle):
        """
        Create the sensors defined by the user and attach them to the ego-vehicle
        :param sensors: list of sensors
        :param vehicle: ego vehicle
        :return:
        """
        bp_library = self.world.get_blueprint_library()
        for sensor_spec in sensors:
            # These are the pseudosensors (not spawned)
            if sensor_spec['type'].startswith('sensor.can_bus'):
                # The speedometer pseudo sensor is created directly here
                sensor = CANBusSensor(vehicle, sensor_spec['reading_frequency'])
            # These are the sensors spawned on the carla world
            else:
                bp = bp_library.find(sensor_spec['type'])
                if sensor_spec['type'].startswith('sensor.camera'):
                    bp.set_attribute('image_size_x', str(sensor_spec['width']))
                    bp.set_attribute('image_size_y', str(sensor_spec['height']))
                    bp.set_attribute('fov', str(sensor_spec['fov']))
                    bp.set_attribute('sensor_tick', "0.05")
                    sensor_location = carla.Location(x=sensor_spec['x'], y=sensor_spec['y'],
                                                     z=sensor_spec['z'])
                    sensor_rotation = carla.Rotation(pitch=sensor_spec['pitch'],
                                                     roll=sensor_spec['roll'],
                                                     yaw=sensor_spec['yaw'])
                elif sensor_spec['type'].startswith('sensor.lidar'):
                    bp.set_attribute('range', '200')
                    bp.set_attribute('rotation_frequency', '10')
                    bp.set_attribute('channels', '32')
                    bp.set_attribute('upper_fov', '15')
                    bp.set_attribute('lower_fov', '-30')
                    bp.set_attribute('points_per_second', '500000')
                    bp.set_attribute('sensor_tick', "0.05")
                    sensor_location = carla.Location(x=sensor_spec['x'], y=sensor_spec['y'],
                                                     z=sensor_spec['z'])
                    sensor_rotation = carla.Rotation(pitch=sensor_spec['pitch'],
                                                     roll=sensor_spec['roll'],
                                                     yaw=sensor_spec['yaw'])
                elif sensor_spec['type'].startswith('sensor.other.gnss'):
                    sensor_location = carla.Location(x=sensor_spec['x'], y=sensor_spec['y'],
                                                     z=sensor_spec['z'])
                    sensor_rotation = carla.Rotation()

                # create sensor
                sensor_transform = carla.Transform(sensor_location, sensor_rotation)
                sensor = self.world.spawn_actor(bp, sensor_transform,
                                                vehicle)

            # setup callback
            if self._save_sensors:  # We have the options to not save sensors data
                sensor.listen(CallBack(sensor_spec['id'], sensor, self._sensor_interface,
                                   writer=self._writer))
            else:
                sensor.listen(CallBack(sensor_spec['id'], sensor, self._sensor_interface,
                                       writer=None))
            self._instanced_sensors.append(sensor)

        # check that all sensors have initialized their data structure
        while not self._sensor_interface.all_sensors_ready():
            logging.debug(" waiting for one data reading from sensors...")
            self._sync(self.world.tick())

    def _get_current_wp_direction(self, vehicle_position, route):

        # for the current position and orientation try to get the closest one from the waypoints
        closest_id = 0
        closest_waypoint = None
        min_distance = 100000
        for index in range(len(route)):
            waypoint = route[index][0]
            computed_distance = distance_vehicle(waypoint, vehicle_position)
            if computed_distance < min_distance:
                min_distance = computed_distance
                closest_id = index
                closest_waypoint = waypoint

        direction = route[closest_id][1]
        if direction == RoadOption.LEFT:
            direction = 3.0
        elif direction == RoadOption.RIGHT:
            direction = 4.0
        elif direction == RoadOption.STRAIGHT:
            direction = 5.0
        else:
            direction = 2.0

        return closest_waypoint, direction


    def _reset_map(self):
        """
        We set all the traffic lights to green to avoid having this scenario.

        """
        ### This was used for L0

        #for actor in self.world.get_actors():
        #    if 'traffic_light' in actor.type_id:
        #        actor.set_state(carla.TrafficLightState.Green)
        #        actor.set_green_time(100000)
        pass
        # TODO for now we are just randomizing the seeds and that is it


    def build_master_scenario(self, route, town_name, timeout):
        # We have to find the target.
        # we also have to convert the route to the expected format
        master_scenario_configuration = ScenarioConfiguration()
        master_scenario_configuration.target = route[-1][0]  # Take the last point and add as target.
        #print (" BEFORE CONVERSION ")
        #print (clean_route(route))
        master_scenario_configuration.route = convert_transform_to_location(route)

        master_scenario_configuration.town = town_name
        master_scenario_configuration.ego_vehicle = ActorConfigurationData('vehicle.lincoln.mkz2017',
                                                                           self._ego_actor.get_transform())
        master_scenario_configuration.trigger_point = self._ego_actor.get_transform()
        CarlaDataProvider.register_actor(self._ego_actor)

        return MasterScenario(self.world, self._ego_actor, master_scenario_configuration,
                              timeout=timeout)


    def _load_world(self):
        # time continues
        attempts = 0

        while attempts < self.MAX_CONNECTION_ATTEMPTS:
            try:
                self.world = self._client.load_world(self._town_name)
                logging.debug("=============================")
                logging.debug("---------New Episode---------")
                logging.debug("=============================")
                break
            except Exception:
                import traceback
                traceback.print_exc()
                attempts += 1
                print('======[WARNING] The server is not ready [{}/{} attempts]!!'.format(attempts,
                                                                      self.MAX_CONNECTION_ATTEMPTS))
                time.sleep(2.0)
                continue

        settings = self.world.get_settings()
        settings.no_rendering_mode = self._exp_params['non_rendering_mode']
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 0.05

        self.world.set_weather(self._exp_params['weather_profile'])
        self.world.apply_settings(settings)

        # We also set the client to record carla loggings for this episode
        root_path = os.environ["SRL_DATASET_PATH"]
        env_full_path = os.path.join(root_path, self._exp_params['package_name'],
                                           self._exp_params['env_name'],
                                           str(self._exp_params['exp_number'])
                                     + '_' + self._agent_name)

    # Todo make a scenario builder class

    def _build_background(self, background_definition, timeout):
        scenario_configuration = ScenarioConfiguration()
        scenario_configuration.route = None
        scenario_configuration.town = self._town_name
        # TODO The random seed should be set
        # print ("BUILDING BACKGROUND OF DEFINITION ", background_definition)
        configuration_instances = []
        for key, numbers in background_definition.items():
            if 'walker' not in key:
                model = key
                transform = carla.Transform()
                autopilot = True
                random = True
                actor_configuration_instance = ActorConfigurationData(model, transform,
                                                                      autopilot, random,
                                                                      amount=background_definition[key])
                configuration_instances.append(actor_configuration_instance)

        scenario_configuration.other_actors = configuration_instances
        return BackgroundActivity(self.world, self._ego_actor, scenario_configuration,
                                  timeout=timeout, debug_mode=False)

    # TODO adding also scenario
    def build_scenario_instances(self, scenario_definition_vec, timeout):

        """
            Based on the parsed route and possible scenarios, build all the scenario classes.
        :param scenario_definition_vec: the dictionary defining the scenarios
        :param town: the town where scenarios are going to be
        :return:
        """
        list_instanced_scenarios = []
        if scenario_definition_vec is None:
            return list_instanced_scenarios


        for scenario_name in scenario_definition_vec:
            # The BG activity encapsulates several scenarios that contain vehicles going arround
            if scenario_name == 'background_activity':  # BACKGROUND ACTIVITY SPECIAL CASE

                background_definition = scenario_definition_vec[scenario_name]
                list_instanced_scenarios.append(self._build_background(background_definition,
                                                                       timeout))

            else:

                # Sample the scenarios to be used for this route instance.
                # tehre can be many instances of the same scenario
                scenario_definition_instances = scenario_definition_vec[scenario_name]

                if scenario_definition_instances is None:
                    raise ValueError(" Not Implemented ")


                for scenario_definition in scenario_definition_instances:

                    # TODO scenario 4 is out

                    ScenarioClass = number_class_translation[scenario_name][0]

                    egoactor_trigger_position = convert_json_to_transform(
                        scenario_definition)
                    scenario_configuration = ScenarioConfiguration()
                    scenario_configuration.other_actors = None  # TODO the other actors are maybe needed
                    scenario_configuration.town = self._town_name
                    scenario_configuration.trigger_point = egoactor_trigger_position
                    scenario_configuration.ego_vehicle = ActorConfigurationData(
                                                            'vehicle.lincoln.mkz2017',
                                                            self._ego_actor.get_transform())
                    try:
                        scenario_instance = ScenarioClass(self.world, self._ego_actor,
                                                          scenario_configuration,
                                                          criteria_enable=False, timeout=timeout)
                    except Exception as e:
                        #if  self._exp_params['debug'] > 1:
                        #     raise e
                        #else:
                        print("Skipping scenario '{}' due to setup error: {}".format(
                            'Scenario3', e))
                        continue
                    # registering the used actors on the data provider so they can be updated.

                    CarlaDataProvider.register_actors(scenario_instance.other_actors)

                    list_instanced_scenarios.append(scenario_instance)

        return list_instanced_scenarios

    def get_summary(self):

        return self._route_statistics


    def record(self):
        self._route_statistics = record_route_statistics_default(self._master_scenario,
                                                                 self._exp_params['env_name'] + '_' +
                                                                 str(self._exp_params['env_number']) + '_' +
                                                                 str(self._exp_params['exp_number']))

        if self._save_data:
            self._writer.save_summary(self._route_statistics)
            if self._exp_params['remove_wrong_data']:
                if self._route_statistics['result'] == 'FAILURE':
                    self._clean_bad_dataset()


    def cleanup(self, ego=True):
        """
        Remove and destroy all actors
        """

        for scenario in self._list_scenarios:
            # Reset scenario status for proper cleanup
            scenario.scenario.terminate()
            # Do not call del here! Directly enforce the actor removal
            scenario.remove_all_actors()
            scenario = None

        self._client.stop_recorder()
        # We need enumerate here, otherwise the actors are not properly removed
        for i, _ in enumerate(self._instanced_sensors):
            if self._instanced_sensors[i] is not None:
                self._instanced_sensors[i].stop()
                self._instanced_sensors[i].destroy()
                self._instanced_sensors[i] = None
        self._instanced_sensors = []
        self._sensor_interface.destroy()
        #  We stop the sensors first to avoid problems

        CarlaActorPool.cleanup()
        CarlaDataProvider.cleanup()

        if ego and self._ego_actor is not None:
            self._ego_actor.destroy()
            self._ego_actor = None
            logging.debug("Removed Ego Vehicle")

        if self.world is not None:
            self.world = None



    def _clean_bad_dataset(self):
        # TODO for now only deleting on failure.
        # Basically remove the folder associated with this exp if the status was not success,
        # or if did not achieve the correct ammount of points
        logging.debug( "FAILED , DELETING")
        self._writer.delete()

