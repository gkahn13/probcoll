#  path is folder of gps_quadrotor package
#  params is global configuration file

import yaml
import os

path = os.path.dirname(os.path.abspath(__file__))
params = {}

def load_params(yaml_path):
    global params
    params.clear()
    with open(yaml_path, "r") as f:
        params.update(yaml.load(f))

def set_params(new_params, clear=True):
    global param
    if clear: params.clear()
    params.update(new_params)
