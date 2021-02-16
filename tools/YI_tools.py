import os
import glob
from torchvision.utils import save_image
import numpy as np
import scipy.misc
import torch
import cv2
from scipy.ndimage.filters import gaussian_filter, median_filter
import re
import shutil
import json
import math
import matplotlib.pyplot as plt
import subprocess

def tryint(s):
    try:
        return int(s)
    except:
        return s

def alphanum_key(s):
    """ Turn a string into a list of string and number chunks.
        "z23a" -> ["z", 23, "a"]
    """
    return [tryint(c) for c in re.split('([0-9]+)', s) ]



if False:
    dataset_path = '/datatmp/fcodevilla/Datasets/MSN/dataset_dynamic_l0'
    all_episodes_path_list = glob.glob(os.path.join(dataset_path, '*'))
    for episode_path in all_episodes_path_list:
        json_path_list = glob.glob(os.path.join(episode_path, '0_Multi', '0','summary.json'))
        if json_path_list == []:
            subprocess.call(['rm', '-r', episode_path])
            #print(episode_path.split('/')[-1])

if False:
    data_path = '/home/yixiao/Datasets/ICML/sample_benchmark_ped_resized'
    all_episodes_path_list = glob.glob(os.path.join(data_path, '*'))

    output_folder = os.path.join('/home/yixiao/Datasets/ICML/videos')
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for episode_path in all_episodes_path_list:
        summary_json_path = os.path.join(episode_path, '0_Agent', '0', 'summary.json')
        if not os.path.exists(summary_json_path):
            print('Not finished', episode_path.split('/')[-1])
            continue

        if not os.path.exists(os.path.join(output_folder, episode_path.split('/')[-1] + '_' + 'rgb' + '.mp4')):
            subprocess.call(['ffmpeg', '-f', 'image2', '-i', os.path.join(episode_path, '0_Agent', '0', 'rgb_central' + '%06d.png'), '-vcodec', 'mpeg4', '-y',
                             os.path.join(output_folder, episode_path.split('/')[-1] + '_' + 'rgb' + '.mp4')])

if False:
    dataset_path = '/datatmp/fcodevilla/Datasets/MSN/dataset_vehicles_l0'
    #dataset_path = '/datatmp/Datasets/yixiao_smalldataset'

    s_1 = 0
    s_2 = 0
    s_3 = 0
    s_4 = 0
    s_5 = 0

    b_1 = 0
    b_2 = 0

    t_1 = 0
    t_2 = 0
    t_3 = 0

    all_episodes_path_list = glob.glob(os.path.join(dataset_path, '*'))
    for episode_path in all_episodes_path_list:
        episode = episode_path.split('/')[-1]
        json_path_list = glob.glob(os.path.join(episode_path, '0_Agent', '0','measurement*.json'))
        for json_file in json_path_list:
            with open(json_file) as json_:
                data = json.load(json_)
                steer = data['steer']
                brake = data['brake']
                throttle = data['throttle']
                if steer < -0.3:
                    s_1 += 1
                elif steer >= -0.3 and steer < -0.05:
                    s_2 += 1
                elif steer >= -0.05 and steer < 0.05:
                    s_3 += 1
                elif steer >= 0.05 and steer < 0.3:
                    s_4 += 1
                elif steer >= 0.3:
                    s_5 += 1

                if brake <= 0.1:
                    b_1 += 1
                else:
                    b_2 += 1

                if throttle < 0.4:
                    t_1 += 1
                elif throttle >= 0.4 and throttle <= 0.6:
                    t_2 += 1
                elif throttle > 0.6:
                    t_3 += 1

    print(s_1,s_2,s_3,s_4,s_5)
    print(b_1, b_2)
    print(t_1, t_2, t_3)


if False:
    dataset_path = '/datatmp/fcodevilla/Datasets/MSN/dataset_vehicles_l0'
    #dataset_path = '/datatmp/Datasets/yixiao_smalldataset'

    all_episodes_path_list = glob.glob(os.path.join(dataset_path, '*'))
    for episode_path in all_episodes_path_list:
        episode = episode_path.split('/')[-1]
        json_path_list = glob.glob(os.path.join(episode_path, '0_Agent', '0','measurement*.json'))
        for json_file in json_path_list:
            with open(json_file) as json_:
                data = json.load(json_)
                steer = data['steer']
                if steer < -0.3:
                    data['steer_class_num'] = 0
                elif steer >= -0.3 and steer < -0.05:
                    data['steer_class_num'] = 1
                elif steer >= -0.05 and steer < 0.05:
                    data['steer_class_num'] = 2
                elif steer >= 0.05 and steer < 0.3:
                    data['steer_class_num'] = 3
                elif steer >= 0.3:
                    data['steer_class_num'] = 4

                brake = data['brake']
                if brake <= 0.1:
                    data['brake_class_num'] = 0

                else:
                    data['brake_class_num'] = 1

                throttle = data['throttle']
                if throttle < 0.4:
                    data['throttle_class_num'] = 0
                elif throttle >= 0.4 and throttle <= 0.6:
                    data['throttle_class'] = [0, 1, 0]
                    data['throttle_class_num'] = 1
                elif throttle > 0.6:
                    data['throttle_class'] = [0, 0, 1]
                    data['throttle_class_num'] = 2

            with open(json_file, 'w') as f:
                json.dump(data, f, indent=4)


# to get the histogram of data distribution
if False:
    dataset_path = '/datatmp/fcodevilla/Datasets/MSN/dataset_vehicles_affordances_Town01_l1'

    all_episodes_path_list = glob.glob(os.path.join(dataset_path, '*'))
    """
    all_episodes_path_list_skip = [os.path.join(dataset_path, 'ClearNoon_challenge_route00001'),
                              os.path.join(dataset_path, 'ClearNoon_challenge_route00002'),
                              os.path.join(dataset_path, 'ClearNoon_challenge_route00003'),
                              os.path.join(dataset_path, 'ClearNoon_challenge_route00004'),
                              os.path.join(dataset_path, 'ClearNoon_navigation_route00003'),
                              os.path.join(dataset_path, 'ClearNoon_navigation_route00004'),
                              os.path.join(dataset_path, 'ClearNoon_navigation_route00005'),
                              os.path.join(dataset_path, 'ClearNoon_navigation_route00006'),
                              os.path.join(dataset_path, 'ClearSunset_challenge_route00005'),
                              os.path.join(dataset_path, 'ClearSunset_challenge_route00006'),
                              os.path.join(dataset_path, 'ClearSunset_challenge_route00007'),
                              os.path.join(dataset_path, 'ClearSunset_challenge_route00008'),
                              os.path.join(dataset_path, 'ClearSunset_navigation_route00007'),
                              os.path.join(dataset_path, 'ClearSunset_navigation_route00008'),
                              os.path.join(dataset_path, 'ClearSunset_navigation_route00009'),
                              os.path.join(dataset_path, 'ClearSunset_navigation_route00010'),
                              os.path.join(dataset_path, 'HardRainNoon_challenge_route00009'),
                              os.path.join(dataset_path, 'HardRainNoon_challenge_route00011'),
                              os.path.join(dataset_path, 'HardRainNoon_challenge_route00012'),
                              os.path.join(dataset_path, 'HardRainNoon_challenge_route00013'),
                              os.path.join(dataset_path, 'HardRainNoon_navigation_route00013'),
                              os.path.join(dataset_path, 'HardRainNoon_navigation_route00014'),
                              os.path.join(dataset_path, 'HardRainNoon_navigation_route00016'),
                              os.path.join(dataset_path, 'HardRainNoon_navigation_route00017'),
                              os.path.join(dataset_path, 'WetNoon_challenge_route00014'),
                              os.path.join(dataset_path, 'WetNoon_challenge_route00015'),
                              os.path.join(dataset_path, 'WetNoon_challenge_route00016'),
                              os.path.join(dataset_path, 'WetNoon_challenge_route00017'),
                              os.path.join(dataset_path, 'WetNoon_navigation_route00019'),
                              os.path.join(dataset_path, 'WetNoon_navigation_route00021'),
                              os.path.join(dataset_path, 'WetNoon_navigation_route00022'),
                              os.path.join(dataset_path, 'WetNoon_navigation_route00023')]
    """

    true =0
    false =0
    total_data =0
    values = []
    for episode_path in all_episodes_path_list:
        if episode_path in all_episodes_path_list_skip:
            continue
        episode = episode_path.split('/')[-1]
        json_path_list = glob.glob(os.path.join(episode_path, '0_Agent', '0','measurement*.json'))
        for json_file in json_path_list:
            with open(json_file) as json_:
                data = json.load(json_)
                values.append(data['relative_angle'])
                values.append(data['relative_angle'] + np.deg2rad(30))
                values.append(data['relative_angle'] + np.deg2rad(-30))
                if abs(data['relative_angle']) > math.pi:
                    print(json_file)
                total_data+=3

    print('Finish appending of values')
    print('total data number', total_data)

    print(min(values))
    print(max(values))

    # the histogram of the data
    n, bins, patches = plt.hist(values, 50)

    plt.xlabel('meters')
    plt.ylabel('data number')
    plt.title('dataset_vehicles_affordances_Town01_l1 (Train): Relative Angle')
    plt.text(60, .025, r'$\mu=100,\ \sigma=15$')
    plt.xlim([min(values), max(values)])
    plt.grid(True)
    plt.show()

    plt.savefig('dataset_vehicles_affordances_Town01_l1_Relative_angle_train.png')


# This is for checking training dataset "SUCCESS"
if False:
    dataset_path = '/datatmp/fcodevilla/Datasets/MSN/dataset_vehicles_affordances_l0'

    all_episodes_path_list = glob.glob(os.path.join(dataset_path, '*'))

    can_used = 0
    for episode_path in all_episodes_path_list:
        episode = episode_path.split('/')[-1]
        summary_json_path = os.path.join(episode_path, '0_Agent', '0', 'summary.json')
        if not os.path.exists(summary_json_path):
            print('Not completed', episode)
            continue
        with open(summary_json_path) as summary_json:
            summary_data = json.load(summary_json)
            if summary_data['result'] == "SUCCESS":
                if summary_data['score_composed'] != 100.0:
                    print(episode)
                    #print(summary_data)
                else:
                    can_used +=1

            else:
                print(episode)
                print(summary_data['result'])

    print(can_used)



if False:
    dataset_path = '/datatmp/fcodevilla/Datasets/MSN/dataset_new_town_l0'
    save_path = '/datatmp/fcodevilla/Datasets/MSN/dataset_new_town_l0_resized'

    all_episodes_path_list = glob.glob(os.path.join(dataset_path, '*'))

    for i in range(0, len(all_episodes_path_list)):
        print(i)
        episode = all_episodes_path_list[i].split('/')[-1]
        print('Checking images size in ', episode)
        images_list = glob.glob(os.path.join(all_episodes_path_list[i],'0_Agent', '0', '*.png'))
        for image_path in images_list:
            #print(image_path)
            image = scipy.misc.imread(image_path)
            image_size = image.shape
            #print(image_size)
            if image_size[0] != int(88):
                image_name = image_path.split('/')[-1]
                image = image[90:485:3]
                image = scipy.misc.imresize(image, (88, 200))
                saving_img_dir = os.path.join(save_path, episode, '0_Agent', '0')
                if not os.path.exists(saving_img_dir):
                    os.makedirs(saving_img_dir)
                scipy.misc.imsave(os.path.join(saving_img_dir, image_name), image)

if False:
    plt.pie([109828/402049,(402049-109828)/402049], labels=['True', 'False'], autopct='%1.1f%%')
    plt.title('dataset dataset_vehicles_affordances_Town01_l1 (Train): Hazard Stop')
    plt.show()
    plt.savefig('dataset_vehicles_affordances_Town01_l1_Hazard_stop_train.png')

if False:
    dataset_path = '/home/yixiao/Datasets/ICML/corl2017_Town01_navigation'
    #all_episodes_path_list = glob.glob(os.path.join(dataset_path, '*'))
    #"""
    all_episodes_path_list = [os.path.join(dataset_path, 'ClearNoon_navigation_route00007'),
                              os.path.join(dataset_path, 'ClearNoon_navigation_route00016'),
                              os.path.join(dataset_path, 'ClearSunset_navigation_route00010'),
                              os.path.join(dataset_path, 'ClearSunset_navigation_route00007'),
                              os.path.join(dataset_path, 'HardRainNoon_navigation_route00005'),
                              os.path.join(dataset_path, 'HardRainNoon_navigation_route00022'),
                              os.path.join(dataset_path, 'WetNoon_navigation_route00023'),
                              os.path.join(dataset_path, 'WetNoon_navigation_route00007')


        #os.path.join(dataset_path, 'ClearNoon_navigation_route00009'),
        #os.path.join(dataset_path, 'ClearNoon_navigation_route00005'),
        #os.path.join(dataset_path, 'ClearNoon_navigation_route00014'),
        #os.path.join(dataset_path, 'ClearNoon_navigation_route00004'),
        #os.path.join(dataset_path, 'ClearSunset_navigation_route00005'),
        #os.path.join(dataset_path, 'ClearSunset_navigation_route00006'),
        #os.path.join(dataset_path, 'ClearSunset_navigation_route00008'),
        #os.path.join(dataset_path, 'HardRainNoon_navigation_route00014'),
        #os.path.join(dataset_path, 'HardRainNoon_navigation_route00002'),
        #os.path.join(dataset_path, 'WetNoon_navigation_route00013'),
        #os.path.join(dataset_path, 'WetNoon_navigation_route00014'),
        #os.path.join(dataset_path, 'WetNoon_navigation_route00021')
                              ]
        
    #"""

    is_pedestrian_hazard = 0
    is_vehicle_hazard = 0
    is_red_tl_hazard = 0
    total_data_number = 0
    for episode_path in all_episodes_path_list:
        #if episode_path in all_episodes_path_list_skip:
        #    continue
        episode = episode_path.split('/')[-1]
        #if episode.split('_')[-2] == 'navigation' and episode.split('_')[-1] == 'route00015':
        #    continue
        print('Calculating in ', episode)
        files_list = glob.glob(os.path.join(episode_path, '0_Agent', '0','measurement*.json'))
        episode_data_number = 0
        for file_path in files_list:
            with open(file_path) as json_file:
                data = json.load(json_file)
                if data['is_pedestrian_hazard']:
                    is_pedestrian_hazard += 3
                if data['is_red_tl_hazard']:
                    is_red_tl_hazard += 3
                if data['is_vehicle_hazard']:
                    is_vehicle_hazard += 3
                if data['target_speed'] != 20.0:
                    print(data['target_speed'])
                episode_data_number += 3
                total_data_number += 3
        print(episode_data_number)

    print("Total number of data", total_data_number)
    print("is_pedestrian_hazard", is_pedestrian_hazard)
    print("is_red_tl_hazard", is_red_tl_hazard)
    print("is_vehicle_hazard", is_vehicle_hazard)

    print("percentage of is_pedestrian_hazard true: ", is_pedestrian_hazard / total_data_number)
    print("percentage of is_red_tl_hazard true: ", is_red_tl_hazard / total_data_number)
    print("percentage of is_vehicle_hazard true: ", is_vehicle_hazard / total_data_number)

if False:
    dataset_path = '/home/yixiao/Datasets/ICML/dataset_dynamic_l0_resized'
    #all_episodes_path_list = glob.glob(os.path.join(dataset_path, '*'))

    all_episodes_path_list = [os.path.join(dataset_path, 'ClearNoon_routes_route00059'),
                              os.path.join(dataset_path, 'ClearNoon_routes_route00061'),
                              os.path.join(dataset_path, 'ClearNoon_routes_route00069'),
                              os.path.join(dataset_path, 'ClearNoon_routes_route00081'),
                              os.path.join(dataset_path, 'ClearNoon_routes_route00092'),
                              os.path.join(dataset_path, 'ClearNoon_routes_route00094'),
                              os.path.join(dataset_path, 'ClearSunset_routes_route00041'),
                              os.path.join(dataset_path, 'ClearSunset_routes_route00043'),
                              os.path.join(dataset_path, 'ClearSunset_routes_route00046'),
                              os.path.join(dataset_path, 'ClearSunset_routes_route00054'),
                              os.path.join(dataset_path, 'ClearSunset_routes_route00055'),
                              os.path.join(dataset_path, 'ClearSunset_routes_route00078'),
                              os.path.join(dataset_path, 'HardRainNoon_routes_route00034'),
                              os.path.join(dataset_path, 'HardRainNoon_routes_route00043'),
                              os.path.join(dataset_path, 'HardRainNoon_routes_route00054'),
                              os.path.join(dataset_path, 'HardRainNoon_routes_route00061'),
                              os.path.join(dataset_path, 'HardRainNoon_routes_route00072'),
                              os.path.join(dataset_path, 'HardRainNoon_routes_route00082'),
                              os.path.join(dataset_path, 'HardRainNoon_routes_route00087'),
                              os.path.join(dataset_path, 'HardRainNoon_routes_route00099'),
                              os.path.join(dataset_path, 'WetNoon_routes_route00032'),
                              os.path.join(dataset_path, 'WetNoon_routes_route00037'),
                              os.path.join(dataset_path, 'WetNoon_routes_route00044'),
                              os.path.join(dataset_path, 'WetNoon_routes_route00046'),
                              os.path.join(dataset_path, 'WetNoon_routes_route00048'),
                              os.path.join(dataset_path, 'WetNoon_routes_route00049'),
                              os.path.join(dataset_path, 'WetNoon_routes_route00052'),
                              os.path.join(dataset_path, 'WetNoon_routes_route00053'),
                              os.path.join(dataset_path, 'WetNoon_routes_route00060'),
                              os.path.join(dataset_path, 'WetNoon_routes_route00071'),
                              os.path.join(dataset_path, 'WetNoon_routes_route00072'),
                              os.path.join(dataset_path, 'WetNoon_routes_route00074'),
                              os.path.join(dataset_path, 'WetNoon_routes_route00088'),
                              os.path.join(dataset_path, 'WetNoon_routes_route00094'),
                              ]

    for episode_path in all_episodes_path_list:
        episode = episode_path.split('/')[-1]
        print('Checking in ', episode)
        files_list = glob.glob(os.path.join(episode_path, '0_Agent', '0', 'measurement*.json'))
        rgb_central_list = glob.glob(os.path.join(episode_path, '0_Agent', '0', 'rgb_central*.png'))
        rgb_left_list = glob.glob(os.path.join(episode_path, '0_Agent', '0', 'rgb_left*.png'))
        rgb_right_list = glob.glob(os.path.join(episode_path, '0_Agent', '0', 'rgb_right*.png'))
        labels_central_list = glob.glob(os.path.join(episode_path, '0_Agent', '0', 'labels_central*.png'))
        labels_left_list = glob.glob(os.path.join(episode_path, '0_Agent', '0', 'labels_left*.png'))
        labels_right_list = glob.glob(os.path.join(episode_path, '0_Agent', '0', 'labels_right*.png'))
        if len(files_list) == len(rgb_central_list) == len(rgb_left_list) == len(rgb_right_list) == \
                len(labels_central_list) == len(labels_left_list) == len(labels_right_list):
            continue

        else:
            print(len(files_list), len(rgb_central_list), len(rgb_left_list), len(rgb_right_list),
                  len(labels_central_list), len(labels_left_list), len(labels_right_list))



if False:
    dataset_path = '/datatmp/fcodevilla/Datasets/MSN/dataset_vehicles_affordances_Town01_l1'
    all_episodes_path_list = glob.glob(os.path.join(dataset_path, '*'))
    all_episodes_path_list = glob.glob(os.path.join(dataset_path, 'HardRainNoon_navigation_route00024'))

    for episode_path in all_episodes_path_list:
        episode = episode_path.split('/')[-1]
        # if episode.split('_')[-2] == 'navigation' and episode.split('_')[-1] == 'route00015':
        #    continue
        print('Calculating in ', episode)
        files_list = sorted(glob.glob(os.path.join(episode_path, '0_Agent', '0', 'measurement*.json')))
        rgb_list_0 = sorted(glob.glob(os.path.join(episode_path, '0_Agent', '0', 'rgb_central*.png')))
        rgb_list_1 = sorted(glob.glob(os.path.join(episode_path, '0_Agent', '0', 'rgb_left*.png')))
        rgb_list_2 = sorted(glob.glob(os.path.join(episode_path, '0_Agent', '0', 'rgb_right*.png')))
        rgb_list_3 = sorted(glob.glob(os.path.join(episode_path, '0_Agent', '0', 'labels_central*.png')))
        rgb_list_4 = sorted(glob.glob(os.path.join(episode_path, '0_Agent', '0', 'labels_left*.png')))
        rgb_list_5 = sorted(glob.glob(os.path.join(episode_path, '0_Agent', '0', 'labels_right*.png')))


        print(len(files_list))
        print(len(rgb_list_0), len(rgb_list_1), len(rgb_list_2),len(rgb_list_3),len(rgb_list_4) ,len(rgb_list_5))


if False:
    image_path = '/datatmp/Experiments/yixiao/labels_central000000.png'
    image = scipy.misc.imread(image_path)
    print(image[:,:,0].max())
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            if image[i,j,0] != 12:
                image[i, j, 0] = 0
            else:
                image[i, j, 0] = 1

    scipy.misc.imsave(os.path.join('/datatmp/Experiments/yixiao/', 'temp12.png'), image[:, :, 0]/2*255)

if False:
    dataset_path = '/home/yixiao/Datasets/ICML/sample_benchmark_ped'
    save_path = '/home/yixiao/Datasets/ICML/sample_benchmark_ped_resized'

    #all_episodes_path_list = glob.glob(os.path.join(dataset_path, '*'))
    all_episodes_path_list = ['/home/yixiao/Datasets/ICML/sample_benchmark_ped/WetNoon_navigation_route00011'
                              ]


    for episode_path in all_episodes_path_list:
        episode = episode_path.split('/')[-1]
        print('Resizing images in ', episode)
        metadata_file = glob.glob(os.path.join(episode_path, '*.json'))
        metadata_name = metadata_file[0].split('/')[-1]
        if not os.path.exists(os.path.join(save_path, episode)):
            os.mkdir(os.path.join(save_path, episode))
        shutil.copy(metadata_file[0], os.path.join(os.path.join(save_path, episode), metadata_name))
        images_list = glob.glob(os.path.join(episode_path,'0_Agent', '0', '*.png'))
        files_list = glob.glob(os.path.join(episode_path, '0_Agent', '0','*.json'))
        for image_path in images_list:
            image_name = image_path.split('/')[-1]
            image = scipy.misc.imread(image_path)
            image = image[65:460,:,:]
            image = scipy.misc.imresize(image, (88, 200))
            saving_img_dir = os.path.join(save_path, episode, '0_Agent','0')
            if not os.path.exists(saving_img_dir):
                os.makedirs(saving_img_dir)
            scipy.misc.imsave(os.path.join(saving_img_dir, image_name), image)
        for file_path in files_list:
            file_name = file_path.split('/')[-1]
            saving_files_dir = os.path.join(save_path, episode, '0_Agent','0')
            if not os.path.exists(saving_files_dir):
                os.makedirs(saving_files_dir)
            shutil.copy(file_path, os.path.join(saving_files_dir, file_name))
        print(episode_path, 'Done!')


if False:
    dataset_path = '/datatmp/Datasets/yixiao_cexp/Town01_Empty'
    save_path = '/datatmp/Datasets/yixiao_cexp/Town01_Empty_resized'
    episodes_list = glob.glob(os.path.join(dataset_path, 'route*'))
    episodes_list.sort(key=alphanum_key)
    for episode in episodes_list[14:25]:
        route = episode.split('/')[-1]
        images_list = glob.glob(os.path.join(episode, '*.png'))
        files_list = glob.glob(os.path.join(episode, '*.json'))
        for image_path in images_list:
            image_name = image_path.split('/')[-1]
            image = scipy.misc.imread(image_path)
            image = image[90:485:3]
            image = scipy.misc.imresize(image, (88, 200))
            saving_dir= os.path.join(save_path, route)
            if not os.path.exists(saving_dir):
                os.mkdir(saving_dir)
            scipy.misc.imsave(os.path.join(saving_dir, image_name), image)
        for file in files_list:
            file_name = file.split('/')[-1]
            saving_files_dir = os.path.join(save_path, route)
            shutil.copy(file, os.path.join(saving_files_dir, file_name))
        print(route, "Done!")



if False:
    image_type = 'depth'
    images_dir = '/home/adas/Desktop/Demo_useful/RGBD(LiDAR)_Town2/53_76/depth'

    depth_log = False

    images_filenames = [os.path.join(images_dir, f) for f in glob.glob1(images_dir, "image_*.png")]
    images_filenames = sorted(images_filenames)

    count = 0
    for i in images_filenames:
        image = scipy.misc.imread(i)
        if image_type == 'depth':
            image = image.astype(np.float32)                                   #shape -> (600, 800, 3)
            decoded_image = np.dot(image[:, :, :3], [1.0, 256.0, 65536.0])

            depth_clip = np.clip(decoded_image, 1.0 / 1000.0 * 16777215.0, 100.0 / 1000.0 * 16777215.0)

            depth_clip[np.where(depth_clip == 100.0 / 1000.0 * 16777215.0)] = 0
            depth_clip[np.where(depth_clip == 1.0 / 1000.0 * 16777215.0)] = 0

            depth_clip = depth_clip.astype(np.float32)
            mask = (depth_clip == 0)
            mask = np.ndarray.astype(mask, dtype=np.uint8)
            depth_clip = cv2.inpaint(depth_clip, mask, 0.1, cv2.INPAINT_NS)

            # This step is to simulate projected LiDAR sensor depth maps of Kitti dataset with the pixel range of (0, 25600); Distance range is 100 meters
            depth_map = np.around(depth_clip / (100.0 / 1000.0 * 16777215.0) * (256.0 * 100.0))

            depth_map = median_filter(depth_map, size=6)

            #im_grad = np.gradient(depth_map)
            #grad_xy = np.sqrt(np.power(im_grad[0], 2) + np.power(im_grad[1], 2))
            #boundary_mask = (grad_xy >= 0.8).astype(int)
            #depth_map = boundary_mask * depth_map

            normalized_depth = depth_map / (256.0 * 100.0)

            result = normalized_depth [115:510, :]

            if depth_log:
                logdepth = np.ones(normalized_depth.shape) + \
                           (np.log(normalized_depth) / 5.70378)
                result = np.clip(logdepth, 0.0, 1.0)

            scipy.misc.imsave(os.path.join('/home/adas/Desktop/Demo_useful/RGBD(LiDAR)_Town2/53_76', 'cut', 'images_'+ str(count).zfill(8) + ".png"), result)     # shape -> (600, 800)
            count += 1


        elif image_type == 'labels':
            classes = {0: [0, 0, 0],         # None
                       1: [70, 70, 70],      # Buildings
                       2: [190, 153, 153],   # Fences
                       3: [72, 0, 90],       # Other
                       4: [220, 20, 60],     # Pedestrians
                       5: [153, 153, 153],   # Poles
                       6: [157, 234, 50],    # RoadLines
                       7: [128, 64, 128],    # Roads
                       8: [244, 35, 232],    # Sidewalks
                       9: [107, 142, 35],    # Vegetation
                       10: [0, 0, 255],      # Vehicles
                       11: [102, 102, 156],  # Walls
                       12: [220, 220, 0]     # TrafficSigns
                      }
            result = numpy.zeros((image.shape[0], image.shape[1], 3))
            for key, value in classes.items():
                result[numpy.where(image[:,:,0] == key)] = value
                scipy.misc.imsave(os.path.join("running_converted_images", "image_" + str(count)+ ".png"),result)


