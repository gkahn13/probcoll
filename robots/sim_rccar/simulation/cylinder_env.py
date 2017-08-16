import numpy as np
from car_env import CarEnv

class CylinderEnv(CarEnv):

    def __init__(self, params={}):
        self._model_path = params.get('model_path', 'robots/sim_rccar/simulation/models/cylinder.egg')
        CarEnv.__init__(
            self,
            params=params)
        self._end = 6.0

    def _default_pos(self):
        return (0.0, -6., 0.25)

    def _get_done(self):
        return self._collision or np.array(self._vehicle_pointer.getPos())[1] >= self._end

    def _get_info(self):
        info = {}
        info['pos'] = np.array(self._vehicle_pointer.getPos())
        info['hpr'] = np.array(self._vehicle_pointer.getHpr())
        info['vel'] = self._get_speed()
        info['coll'] = self._collision and np.array(self._vehicle_pointer.getPos())[1] < self._end
        return info

    def _default_restart_pos(self):
        ran = np.linspace(-3.5, 3.5, 20)
        np.random.shuffle(ran)
        restart_pos = []
        for val in ran:
            restart_pos.append([val, -6.0, 0.25, 0.0, 0.0, 3.14])
        return restart_pos

    def _next_random_restart_pos_hpr(self):
        val = np.random.uniform(-3.5, 3.5)
        return [val, -6.0, 0.25], [0.0, 0.0, 3.14]

if __name__ == '__main__':
    params = {'visualize': True, 'run_as_task': True, 'model_path': 'models/cylinder.egg', 'use_depth': True}
    env = CylinderEnv(params)
