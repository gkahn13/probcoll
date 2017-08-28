import os
from robots.sim_rccar.simulation.car_env import CarEnv

class SquareEnv(CarEnv):

    def __init__(self, params={}):
        self._model_path = params.get('model_path',
                                      os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                   'models/square.egg'))
        CarEnv.__init__(
            self,
            params=params)

    def _setup_light(self):
        pass
        
    def _default_pos(self):
        return (20.0, -19., 0.25)

    def _default_restart_pos(self):
        return [
                [ 20., -20., 0.3, 0.0, 0.0, 3.14],
                [-20., -20., 0.3, 0.0, 0.0, 3.14],
                [ 20.,  15., 0.3, 0.0, 0.0, 3.14],
                [-20.,  15., 0.3, 0.0, 0.0, 3.14]
            ]

if __name__ == '__main__':
    params = {'visualize': True, 'run_as_task': True, 'model_path': 'models/square.egg', 'use_depth': True, 'size': [640, 480]}
    env = SquareEnv(params)
