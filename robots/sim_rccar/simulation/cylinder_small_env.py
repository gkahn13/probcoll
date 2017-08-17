from cylinder_env import CylinderEnv
from car_env import CarEnv

class CylinderSmallEnv(CylinderEnv):

    def __init__(self, params={}):
        self._model_path = params.get('model_path', 'robots/sim_rccar/simulation/models/cylinder_small.egg')
        CarEnv.__init__(
            self,
            params=params)
        self._end = 6.0

if __name__ == '__main__':
    params = {'visualize': True, 'run_as_task': True, 'model_path': 'models/cylinder_small.egg', 'use_depth': True}
    env = CylinderSmallEnv(params)
