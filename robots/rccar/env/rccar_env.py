import time
import numpy as np
from pynput import keyboard

from robots.rccar.env.sensors.sensors_handler import SensorsHandler

class RCcarEnv:
    def __init__(self, params):
        self._sensors = SensorsHandler()
        self._collision = False
        self._params = params
        self._do_back_up = params.get('do_back_up', True)
        self._max_time_step = params.get('max_time_step', None)
        self._time_step = 0
#        self._next_time = None
        self._last_obs_time = time.time() # TODO
        self._interrupt_end = False
        self._listener = keyboard.Listener(on_press=self._on_press)
        self._listener.start()

    # Keyboard Listener function

    def _on_press(self, key):
        pass
#        if str(key)[1] == 'p':
#            self.stop()
#            self._interrupt_end = True

    # Helper functions

    def _check_remote_interrupt(self, action):
        if action[1] > 0 and self._sensors.get_motor_data()[2] <= -5.0:
            self._interrupt_end = True

    def _do_action(self, action, t, absolute=False, vel_control=False):
        if not self._get_done():
            if vel_control:
                self._sensors.set_vel_cmd(action)
            else:
                self._sensors.set_motor_cmd(action)
            if absolute:
                sleep_time = t
            else:
                sleep_time = (self._last_obs_time + t) - time.time()
                print(sleep_time)
            if sleep_time > 0:
                time.sleep(sleep_time)

    def stop(self):
        self._do_action((0.0, 0.0), t=0.0, absolute=True, vel_control=False)

    def _continue_check(self):
        time.sleep(0.5)
        while self._sensors.get_motor_data()[2] > -5.0:
            pass
        print("Continuing!")
        time.sleep(0.5)

    def _back_up(self):
        if self._sensors.get_flip():
            print("Car has flipped!")
            self._continue_check()
            self._sensors.reset_crash()
        back_up_vel = self._params['back_up'].get('vel', -2.0) 
        back_up_steer_ran = self._params['back_up'].get('steer', (-5.0, 5.0))
        back_up_steer = np.random.uniform(*back_up_steer_ran)
        duration = self._params['back_up'].get('duration', 1.0)
        self._do_action((back_up_steer, back_up_vel), t=duration, absolute=True, vel_control=True)
        self._do_action((0.0, 0.0), t=1.0, absolute=True, vel_control=False)
        if self._sensors.get_flip():
            print("Car has flipped!")
            self._continue_check()
            self._sensors.reset_crash()
            self._back_up()

    def _get_observation(self):
        self._last_obs_time = time.time()
        return self._sensors.get_image()

    def _get_reward(self):
        reward = -1.0 * int(self._sensors.get_crash())
        return reward

    def _get_done(self):
        return self._sensors.get_crash() or self._interrupt_end or self._time_step >= self._max_time_step

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
        self.stop()
        self._sensors.close()
        self._listener.stop()

    def reset(self):
        if self._interrupt_end:
            self._continue_check() 
        self._interrupt_end = False
        self._time_step = 0
        if self._do_back_up:
            self._sensors.reset_crash()
            self._back_up()
            time.sleep(0.5)
        else:
            self._do_action((0.0, 0.0), t=1.0, vel_control=False)
        self._sensors.reset_crash()
        return self._get_observation()

    def step(self, action):
        if self._time_step >= 2 and abs(self._get_info()['vel']) < 0.1:
            self._time_step = 0
            input("Car is stuck press Enter when ready")
        self._do_action(action, t=self._params['dt'], vel_control=self._params['use_vel'])
        self._time_step += 1
        self._check_remote_interrupt(action)
        observation = self._get_observation()
        reward = self._get_reward()
        done = self._get_done()
        info = self._get_info()
        if done:
            self._do_action((0.0, 0.0), t=1.0, vel_control=False)
        return observation, reward, done, info
