from car_env import CarEnv

class SquareBankedEnv(CarEnv):

    def __init__(self, params={}):
        self._model_path = params.get('model_path', 'robots/sim_rccar/simulation/models/square_banked.egg')
        CarEnv.__init__(
            self,
            params=params)

    def _setup_light(self):
        pass
        
    def _default_pos(self):
        return (20.0, -15., 0.25)

    def _default_restart_pos(self):
        return [
                [ 20., -15., 0.3, 0.0, 0.0, 3.14],
                [-20., -15., 0.3, 0.0, 0.0, 3.14],
                [ 20.,  15., 0.3, 0.0, 0.0, 3.14],
                [-20.,  15., 0.3, 0.0, 0.0, 3.14]
            ]

if __name__ == '__main__':
    params = {'visualize': True, 'run_as_task': True, 'model_path': 'models/square_banked.egg', 'use_depth': True, 'size': [640, 480]}
    env = SquareEnv(params)
