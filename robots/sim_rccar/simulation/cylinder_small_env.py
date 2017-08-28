import os
from robots.sim_rccar.simulation.cylinder_env import CylinderEnv
from robots.sim_rccar.simulation.car_env import CarEnv

class CylinderSmallEnv(CylinderEnv):

    def __init__(self, params={}):
        self._model_path = params.get('model_path',
                                      os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                   'models/cylinder_small.egg'))
        CarEnv.__init__(
            self,
            params=params)
        self._end = 6.0

if __name__ == '__main__':
    params = {'visualize': True, 'run_as_task': True, 'model_path': 'models/cylinder_small.egg', 'use_depth': True}
    env = CylinderSmallEnv(params)
