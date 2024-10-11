
import sys
import os
sys.path.append(os.getcwd())
from tools.data_converter.whales import Whales
import yaml
if __name__ == '__main__':
    root_path = 'data/whales/'
    old_path = 'data/config-old'
    whales = Whales(dataroot=root_path, verbose=True)
    for key in whales.config.keys():
        yaml.safe_dump(whales.config[key], open(os.path.join(old_path, key + '.yaml'), 'w'), default_flow_style=False)
        whales.config[key]['description'] = "Config of WHALES dataset, we inherit config from OpenCDA https://github.com/ucla-mobility/OpenCDA and add new configs for WHALES. See https://github.com/chensiweiTHU/WHALES"
        del whales.config[key]['world']['weather']
        del whales.config[key]['world']['port']
        del whales.config[key]['world']['seed']
        del whales.config[key]['world']['sync_mode']

        del whales.config[key]['platoon_base']
        del whales.config[key]['scenario']
        del whales.config[key]['carla_traffic_manager']

        del whales.config[key]['vehicle_base']['v2x']
        del whales.config[key]['vehicle_base']['controller']
        del whales.config[key]['vehicle_base']['behavior']
        del whales.config[key]['vehicle_base']['safety_manager']
        del whales.config[key]['vehicle_base']['map_manager']
        del whales.config[key]['vehicle_base']['sensing']['perception']['load_bbox']
        del whales.config[key]['vehicle_base']['sensing']['perception']['activate']
        del whales.config[key]['vehicle_base']['sensing']['perception']['camera']['visualize']
        del whales.config[key]['vehicle_base']['sensing']['perception']['camera']['positions']
        del whales.config[key]['vehicle_base']['sensing']['localization']

        del whales.config[key]['rsu_base']
        curr_yaml = os.path.join(root_path, key + '/config.yaml')
        yaml.safe_dump(whales.config[key], open(curr_yaml, 'w'), default_flow_style=False)
    print('finished!')