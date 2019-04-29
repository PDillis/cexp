
import json
import os

if __name__ == '__main__':

    root_route_file_position = '/network/home/codevilf/experience_database_generator/database/nocrash'
    # root_route_file_position = 'srunner/challenge/'
    #filename_town01 = os.path.join(root_route_file_position, 'Town01_navigation.json')

    # The sensor information should be on get data
    sensors = [{'type': 'sensor.camera.rgb',
                'x': 2.0, 'y': 0.0,
                'z': 1.40, 'roll': 0.0,
                'pitch': -15.0, 'yaw': 0.0,
                'width': 800, 'height': 600,
                'fov': 100,
                'id': 'rgb'}
               ]

    # For each of the routes to be evaluated.

    # Tows to be generated
    town_sets = {'Town01': 'Town01_navigation.xml',
                 'Town02': 'Town02_navigation.xml'}


    # Weathers to be generated later
    weather_sets = {'training': ["Clear Noon",
                                  "Wet Noon",
                                  "Hard Rain Noon",
                                   "Clear Sunset"],
                    'new_weather':  ["Wet Sunset",
                                    "Soft Rain Sunset"]
                    }



    name_dict = {'training':{'Town01': 'training',
                             'Town02': 'newtown'
                             },
                 'new_weather': {'Town01': 'newweather',
                                 'Town02': 'newweathertown'

                 }

    }

    # TODO add generation of the other posibilities ( Non-empty)

    for town_name in town_sets.keys():

        for w_set_name in weather_sets.keys():
            # get the actual set  from th name
            w_set = weather_sets[w_set_name]
            new_json = {"envs": {},
                        "additional_sensors": sensors,
                        "package_name": 'nocrash_' + name_dict[w_set_name][town_name] + '_empty' }

            for weather in w_set:

                for env_number in range(25):

                    env_dict = {
                        "route": {
                            "file": town_sets[town_name],
                            "id": env_number
                        },
                        "scenarios": {
                            "file": "None"
                        },
                        "town_name": "Town01",
                        "vehicle_model": "vehicle.lincoln.mkz2017",
                        "weather_profile": weather
                    }

                    new_json["envs"].update({weather + '_route' + str(env_number).zfill(5): env_dict})

            filename = os.path.join(root_route_file_position, 'nocrash_' + name_dict[w_set_name][town_name] + '_empty')

            with open(filename, 'w') as fo:
                # with open(os.path.join(root_route_file_position, 'all_towns_traffic_scenarios3_4.json'), 'w') as fo:
                fo.write(json.dumps(new_json, sort_keys=True, indent=4))
