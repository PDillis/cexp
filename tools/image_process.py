#!/usr/bin/env python
import json
import numpy as np
import glob
import re
import scipy

import argparse
import math

import time
import os
from collections import deque
import scipy.ndimage
from cexp.cexp import CEXP

from cexp.env.environment import NoDataGenerated




# Configurations for this script


sensors = {'RGB': 3, 'labels': 3, 'depth': 0}
resolution = [800, 600]

""" Position to cut the image before reshapping """
""" This is used to cut out the sky (Kind of useless for learning) """
IMAGE_CUT = [90, 485]


def purge(dir, pattern):
    for f in os.listdir(dir):
        if re.search(pattern, f):
            os.remove(os.path.join(dir, f))



def join_classes(labels_image, join_dic):
    compressed_labels_image = np.copy(labels_image)
    for key, value in join_dic.iteritems():
        compressed_labels_image[np.where(labels_image == key)] = value

    return compressed_labels_image


def join_classes_for(labels_image, join_dic):
    compressed_labels_image = np.copy(labels_image)
    # print compressed_labels_image.shape
    for i in range(labels_image.shape[0]):
        for j in range(labels_image.shape[1]):
            compressed_labels_image[i, j, 0] = join_dic[labels_image[i, j, 0]]

    return compressed_labels_image

def reshape_images(image_type, image_name, out_path):

    """
    Function for reshaping all the images of an episode and save it again on the

    Params:
        image_type: The type of images that is going to be reshaped.

    """

    if image_type == 'labels':
        interp_type = 'nearest'
    else:
        interp_type = 'bicubic'
    print ("reshape ", image_name)
    image = scipy.ndimage.imread(image_name)

    if image.shape[0] == 600:
        center = image[IMAGE_CUT[0]:IMAGE_CUT[1], ...]
        center = scipy.misc.imresize(center, (88, 200), interp=interp_type)
        scipy.misc.imsave(out_path, center)

    else:
        print (" size not for processing")


# ***** main loop *****
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Path viewer')


    parser.add_argument(
        '--dataset',
        help=' the json configuration file name',
        default=None
    )
    """ You should pass this extra arguments if you want to delete the semantic segmenation labels"""
    parser.add_argument(
        '-ds', '--delete-semantic-segmentation',
        dest='delete_semantic_segmentation',
        action='store_true',
        help='Flag to tell the system to NOT erase the semantic segmentation labels from the dataset'
    )
    """ You should pass this extra arguments if you want to delete the depth labels"""
    parser.add_argument(
        '-dd', '--delete-depth',
        dest='delete_depth',
        action='store_true',
        help='Flag to tell the system to NOT erase the semantic segmentation labels from the dataset'
    )
    parser.add_argument(
        '-np','--new-package',
        help=' Set the new package_name ',
        default=None
    )

    args = parser.parse_args()

    first_time = True
    count = 0
    steering_pred = []
    steering_gt = []
    step_size = 1
    image_queue = deque()

    actions_queue = deque()

    params = {'save_dataset': False,
              'docker_name': 'carlalatest:latest',
              'gpu': 0,
              'batch_size': 1,
              'remove_wrong_data': False,
              'non_rendering_mode': False,
              'carla_recording': False
              }

    # Start a screen to show everything. The way we work i  s that we do IMAGES x Sensor.
    # But maybe a more arbitrary configuration may be useful
    ts = []
    env_batch = CEXP(args.dataset, params=params, execute_all=True,
                     ignore_previous_execution=True)
    # Here some docker was set
    env_batch.start(no_server=True)  # no carla server mode.
    # count, we count the environments that are read
    for env in env_batch:
        steer_vec = []
        throttle_vec = []
        brake_vec = []
        # it can be personalized to return different types of data.
        print("Environment Name: ", env)
        try:
            env_data = env.get_data()  # returns a basically a way to read all the data properly
        except NoDataGenerated:
            print("No data generate for episode ", env)
        except ValueError:
            print("Json decode error ", env)
            env.remove_data()
        else:

            for exp in env_data:
                print("    Exp: ", exp[1])
                for batch in exp[0]:
                    print("      Batch: ", batch[1])
                    batch_path_name = os.path.join(env.get_path(), exp[1], batch[1])
                    if 'processed' in os.listdir(batch_path_name):
                        continue

                    for data_point in batch[0]:
                        # assume standard name
                        for key in data_point.keys():
                            if 'depth' in key or 'labels' in key or 'rgb' in key:
                                if args.delete_depth and 'depth' in key:
                                    os.remove(data_point[key])
                                if args.delete_semantic_segmentation and 'labels' in key:
                                    os.remove(data_point[key])

                                #Process in the orign or copy ??
                                path_vector = ['/'] + data_point[key].split('/')
                                if args.new_package is not None:
                                    path_vector[-5] = args.new_package
                                out_name = os.path.join(*path_vector)

                                if 'rgb' in key:
                                    reshape_images("rgb", data_point[key], out_name)

                                if not args.delete_depth and 'labels' in key:
                                    reshape_images("labels", data_point[key], out_name)

                                if not args.delete_depth and 'depth' in key:
                                    reshape_images("depth", data_point[key], out_name)

                    with open(os.path.join(batch_path_name, 'processed'), 'w') as f:
                        pass

