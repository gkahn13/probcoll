import std_msgs.msg as sm

import general.ros.ros_utils as ros_utils

from general.policy.policy import Policy
from general.state_info.sample import Sample
from general.utility.logger import get_logger

class TeleopMPCPolicyRCcar(Policy):

    def __init__(self, meta_data):
        use_obs = False
        Policy.__init__(self, meta_data['mpc']['H'], use_obs, meta_data)
        self._logger = get_logger(self.__class__.__name__, 'warn')

        rccar_topics = meta_data['rccar']['topics']
        self.teleop_cmd_steer_callback = \
            ros_utils.RosCallbackMostRecent(rccar_topics['teleop_cmd_steer'], sm.Float32, clear_on_get=False)
        self.teleop_cmd_vel_callback = \
            ros_utils.RosCallbackMostRecent(rccar_topics['teleop_cmd_vel'], sm.Float32, clear_on_get=False)

    def act(self, x, obs, t, noise, ref_traj=None):
        cmd_steer_msg = self.teleop_cmd_steer_callback.get()
        cmd_vel_msg = self.teleop_cmd_vel_callback.get()

        cmd_steer = cmd_steer_msg.data if cmd_steer_msg is not None else 50.
        cmd_vel = cmd_vel_msg.data if cmd_vel_msg is not None else 0.

        sample = Sample(meta_data=self._meta_data, T=2)
        sample.set_U([cmd_steer], t=0, sub_control='cmd_steer')
        sample.set_U([cmd_vel], t=0, sub_control='cmd_vel')
        u = sample.get_U(t=0)

        return u + noise.sample(u)

    def get_info(self):
        return dict()
