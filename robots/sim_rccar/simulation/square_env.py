#!/usr/bin/env python
from car_env import CarEnv

class SquareEnv(CarEnv):

    def __init__(self, params={}):
        self._model_path = 'robots/sim_rccar/simulation/models/square_hallway.egg'
        CarEnv.__init__(
            self,
            params=params)

    def _default_pos(self):
        return (42.5, -42.5, 0.2)
