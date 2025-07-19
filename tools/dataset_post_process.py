
import sys
import os
sys.path.append(os.getcwd())
from tools.data_converter.dolphins import Dolphins
import yaml
if __name__ == '__main__':
    root_path = 'data/whales/'
    old_path = 'data/config-old'
    dolphins = Dolphins(dataroot=root_path, verbose=True)
    for key in dolphins.config.keys():
        yaml.safe_dump(dolphins.config[key], open(os.path.join(old_path, key + '.yaml'), 'w'), default_flow_style=False)
        dolphins.config[key]['description'] = "Config of WHALES dataset, we inherit config from OpenCDA https://github.com/ucla-mobility/OpenCDA and add new configs for WHALES. See https://github.com/chensiweiTHU/WHALES"
        del dolphins.config[key]['world']['weather']
        del dolphins.config[key]['world']['port']
        del dolphins.config[key]['world']['seed']
        del dolphins.config[key]['world']['sync_mode']

        del dolphins.config[key]['platoon_base']
        del dolphins.config[key]['scenario']
        del dolphins.config[key]['carla_traffic_manager']

        del dolphins.config[key]['vehicle_base']['v2x']
        del dolphins.config[key]['vehicle_base']['controller']
        del dolphins.config[key]['vehicle_base']['behavior']
        del dolphins.config[key]['vehicle_base']['safety_manager']
        del dolphins.config[key]['vehicle_base']['map_manager']
        del dolphins.config[key]['vehicle_base']['sensing']['perception']['load_bbox']
        del dolphins.config[key]['vehicle_base']['sensing']['perception']['activate']
        del dolphins.config[key]['vehicle_base']['sensing']['perception']['camera']['visualize']
        del dolphins.config[key]['vehicle_base']['sensing']['perception']['camera']['positions']
        del dolphins.config[key]['vehicle_base']['sensing']['localization']

        del dolphins.config[key]['rsu_base']
        curr_yaml = os.path.join(root_path, key + '/config.yaml')
        yaml.safe_dump(dolphins.config[key], open(curr_yaml, 'w'), default_flow_style=False)
    print('finished!')