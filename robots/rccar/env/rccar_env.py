import time
import numpy as np

from robots.rccar.env.sensors.sensors_handler import SensorsHandler

class RCcarEnv:
    def __init__(self, params):
        self._sensors = SensorsHandler()
        self._collision = False
        self._params = params
        self._do_back_up = params.get('do_back_up', True)
        self._next_time = None

    # Helper functions

    def _do_action(self, action, t, vel_control=False):
        if self._next_time is not None:
            time.sleep(self._next_time - time.time())
        if vel_control:
            self._sensors.set_vel_cmd(action)
        else:
            self._sensors.set_motor_cmd(action)
        self._next_time = time.time() + t

    def _back_up(self):
        back_up_vel = self._params['back_up'].get('vel', -2.0) 
        back_up_steer_ran = self._params['back_up'].get('steer', (-5.0, 5.0))
        back_up_steer = np.random.uniform(*back_up_steer_ran)
        duration = self._params['back_up'].get('duration', 1.0)
        print(duration, back_up_steer, back_up_vel)
        self._do_action((back_up_steer, back_up_vel), t=duration, vel_control=True)
        self._do_action((0.0, 0.0), t=1.0, vel_control=False)

    def _get_observation(self):
        return self._sensors.get_image()

    def _get_reward(self):
        reward = -1.0 * int(self._sensors.get_crash())
        return reward

    def _get_done(self):
        return self._sensors.get_crash()

    def _get_info(self):
        info = {}
        info['coll'] = self._sensors.get_crash()
        info['flipped'] = self._sensors.get_flip()
        info['steer'] = self._sensors.get_motor_data()[1]
        info['motor'] = self._sensors.get_motor_data()[2]
        info['vel'] = self._sensors.get_motor_data()[3]
        info['acc'] = self._sensors.get_imu_data()[:3]
        info['ori'] = self._sensors.get_imu_data()[3:]
        return info

    # Environment functions

    def close(self):
        self._sensors.close()

    def reset(self):
        if self._do_back_up:
            if self._sensors.get_crash():
                self._sensors.reset_crash()
                self._back_up()
        return self._get_observation()

    def step(self, action):
        self._do_action(action, t=self._params['dt'], vel_control=self._params['use_vel'])
        observation = self._get_observation()
        reward = self._get_reward()
        done = self._get_done()
        info = self._get_info()
        return observation, reward, done, info
