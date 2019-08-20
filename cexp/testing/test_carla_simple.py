#!/usr/bin/env python

# Copyright (c) 2019 Computer Vision Center (CVC) at the Universitat Autonoma de
# Barcelona (UAB).
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

import glob
import os
import sys

try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla

import random
import pygame
try:
    import numpy as np
except ImportError:
    raise RuntimeError('cannot import numpy, make sure numpy package is installed')

try:
    import queue
except ImportError:
    import Queue as queue


class CarlaSyncMode(object):
    """
    Context manager to synchronize output from different sensors. Synchronous
    mode is enabled as long as we are inside this context

        with CarlaSyncMode(world, sensors) as sync_mode:
            while True:
                data = sync_mode.tick(timeout=1.0)

    """

    def __init__(self, world, *sensors, **kwargs):
        self.world = world
        self.sensors = sensors
        self.frame = None
        self.delta_seconds = 1.0 / kwargs.get('fps', 20)
        self._queues = []
        self._settings = None
        self._settings = self.world.get_settings()
        self.frame = self.world.apply_settings(carla.WorldSettings(
            no_rendering_mode=True,
            synchronous_mode=True))
        def make_queue(register_event):
            q = queue.Queue()
            register_event(q.put)
            self._queues.append(q)

        make_queue(self.world.on_tick)
        for sensor in self.sensors:
            make_queue(sensor.listen)

    
    def __enter__(self):
        self._settings = self.world.get_settings()
        self.frame = self.world.apply_settings(carla.WorldSettings(
            no_rendering_mode=True,
            synchronous_mode=True))

        def make_queue(register_event):
            q = queue.Queue()
            register_event(q.put)
            self._queues.append(q)

        make_queue(self.world.on_tick)
        for sensor in self.sensors:
            make_queue(sensor.listen)
        return self

    def tick(self, timeout):
        self.frame = self.world.tick()
        data = [self._retrieve_data(q, timeout) for q in self._queues]
        assert all(x.frame == self.frame for x in data)
        return data

    def __exit__(self, *args, **kwargs):
        self.world.apply_settings(self._settings)

    def _retrieve_data(self, sensor_queue, timeout):
        while True:
            data = sensor_queue.get(timeout=timeout)
            if data.frame == self.frame:
                return data


def get_font():
    fonts = [x for x in pygame.font.get_fonts()]
    default_font = 'ubuntumono'
    font = default_font if default_font in fonts else fonts[0]
    font = pygame.font.match_font(font)
    return pygame.font.Font(font, 14)


def load_world(client):

    world = client.get_world()
    m = world.get_map()
    start_pose = random.choice(m.get_spawn_points())
    blueprint_library = world.get_blueprint_library()

    vehicle = world.spawn_actor(
        random.choice(blueprint_library.filter('vehicle.*')),
        start_pose)
    vehicle.set_simulate_physics(False)

    rgb_bp = blueprint_library.find('sensor.camera.rgb')
    rgb_bp.set_attribute('image_size_x', str(1024))
    rgb_bp.set_attribute('image_size_y', str(780))
    rgb_bp.set_attribute('fov', str(120))
    rgb_bp.set_attribute('sensor_tick', "0.05")
    camera_rgb = world.spawn_actor(
        rgb_bp,
        carla.Transform(carla.Location(x=-5.5, z=2.8), carla.Rotation(pitch=-15)),
        attach_to=vehicle)

    camera_semseg = world.spawn_actor(
        blueprint_library.find('sensor.camera.semantic_segmentation'),
        carla.Transform(carla.Location(x=-5.5, z=2.8), carla.Rotation(pitch=-15)),
        attach_to=vehicle)

    #actor_list.append(camera_semseg)

    return world, camera_rgb, camera_semseg

def main():

    clock = pygame.time.Clock()

    client = carla.Client('localhost', 2000)
    client.set_timeout(2.0)

    try:

        world, camera_rgb, camera_semseg = load_world(client)
        # Create a synchronous mode context.
        sync_mode = CarlaSyncMode(world, camera_rgb, camera_semseg, fps=30)

        for i in range(2000):
            clock.tick()
            print (i)


            # Advance the simulation and wait for the data.
            snapshot, image_rgb, image_semseg = sync_mode.tick(timeout=2.0)


            if i % 200 == 0:
                world, camera_rgb, camera_semseg = load_world(client)
                sync_mode = CarlaSyncMode(world, camera_rgb, camera_semseg, fps=30)

            image_semseg.convert(carla.ColorConverter.CityScapesPalette)
            fps = round(1.0 / snapshot.timestamp.delta_seconds)



    finally:

        pygame.quit()
        print('done.')


if __name__ == '__main__':

    try:

        main()

    except KeyboardInterrupt:
        print('\nCancelled by user. Bye!')