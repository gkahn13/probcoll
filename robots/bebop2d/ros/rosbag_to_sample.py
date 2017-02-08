#################################
### Convert rosbag to samples ###
#################################

import os, subprocess
import collections

import numpy as np

import rospy, rosbag
import cv_bridge
import bebop_msgs.msg as bm

import general.ros.ros_utils as ros_utils

from robots.bebop2d.agent.agent_bebop2d import AgentBebop2d

from general.state_info.sample import Sample

from config import load_params, params

FOLDER = '/home/gkahn/code/gps_quadrotor/experiments/bebop2d/lfd_cmd_vel_init'


def spaced_messages(time_msgs, dt):
    """ Space the messages by dt, starting from the end """
    spaced_time_msgs = [time_msgs[-1]]

    for t, msg in time_msgs[::-1]:
        if abs(t - spaced_time_msgs[0][0]) >= dt:
            spaced_time_msgs.insert(0, (t, msg))

    return spaced_time_msgs

def aligned_messages(src_time_msgs, targ_time_msgs, dt):
    """ Align target messages to src messages """
    aligned_targ_time_msgs = []

    for src_t, _ in src_time_msgs:
        time_diff = [abs(targ_t - src_t) for targ_t, _ in targ_time_msgs]
        closest = min(time_diff)
        if closest >= dt / 2.:
            print('Closest is {0}, dt is {1}'.format(closest, dt))
        closest_targ_idx = np.argmin(time_diff)
        aligned_targ_time_msgs.append(targ_time_msgs[closest_targ_idx])

    return aligned_targ_time_msgs

def get_measured_vel(rosbag_fname, msg_dict):
    bebop_topics = params['bebop']['topics']
    ### record all measured_vel
    measured_vel_callback = ros_utils.RosCallbackAll(bebop_topics['measured_vel'],
                                                     bm.Ardrone3PilotingStateSpeedChanged,
                                                     max_num_msgs=np.inf)
    ### rosbag play
    subprocess.call('rosbag play {0} -i'.format(rosbag_fname), shell=True, stdout=subprocess.PIPE)
    ### get all msgs
    measured_vel_msgs = measured_vel_callback.get()
    assert(len(measured_vel_msgs) == len(msg_dict[bebop_topics['measured_vel']]))
    msg_dict[bebop_topics['measured_vel']] = \
        [(t, msg) for (t, _), msg in zip(msg_dict[bebop_topics['measured_vel']], measured_vel_msgs)]

def rosbag_to_sample(rosbag_fname):
    print('Parsing {0}'.format(rosbag_fname))

    bebop_topics = params['bebop']['topics']

    ### read all messages
    bag = rosbag.Bag(rosbag_fname, 'r')
    msg_dict = collections.defaultdict(list)
    start_time = None
    for topic, msg, t in bag.read_messages():
        if start_time is None:
            start_time = t.to_sec()
        else:
            assert(start_time <= t.to_sec())
        msg_dict[topic].append((t.to_sec() - start_time, msg))
    bag.close()
    # get_measured_vel(rosbag_fname, msg_dict) # necessary hack
    if len(msg_dict[bebop_topics['start_rollout']]) == 0:
        print('\tNo start_rollout, returning None')
        return None
    ### sort by time
    for k, v in msg_dict.items():
        msg_dict[k] = sorted(v, key=lambda x: x[0])
    ### prune start and end
    is_coll = (bebop_topics['collision'] in msg_dict.keys())
    start_time = msg_dict[bebop_topics['start_rollout']][0][0]
    end_time = msg_dict[bebop_topics['collision']][0][0] if is_coll else np.inf
    for topic, time_msg in msg_dict.items():
        msg_dict[topic] = [(t, msg) for t, msg in time_msg if start_time <= t <= end_time]

    ### necessary info for Samples
    dt = params['dt']
    T = params['prediction']['dagger']['T']

    ### space by dt
    image_time_msgs = spaced_messages(msg_dict[bebop_topics['image']], dt)[-T:]
    cmd_vel_time_msgs = aligned_messages(image_time_msgs, msg_dict[bebop_topics['cmd_vel']], dt)[-T:]
    # measured_vel_time_msgs = aligned_messages(image_time_msgs, msg_dict[bebop_topics['measured_vel']], dt)[-T:]

    ### create sample
    sample = Sample(T=len(image_time_msgs), meta_data=params)
    cvb = cv_bridge.CvBridge()
    for t, (image_time_msg, cmd_vel_time_msg) in enumerate(zip(image_time_msgs, cmd_vel_time_msgs)):
        image_msg = image_time_msg[1]
        cmd_vel_msg = cmd_vel_time_msg[1]

        ### image
        im = AgentBebop2d.process_image(image_msg, cvb)
        sample.set_O(im.ravel(), t=t, sub_obs='camera')

        ### collision
        sample.set_O([0], t=t, sub_obs='collision')

        ### linearvel
        sample.set_X([cmd_vel_msg.linear.x, cmd_vel_msg.linear.y], t=t)
        sample.set_U([cmd_vel_msg.linear.x, cmd_vel_msg.linear.y], t=t)
    sample.set_O([int(is_coll)], t=-1, sub_obs='collision')
    # sample.set_O([int(np.random.random() > 0.5)], t=-1, sub_obs='collision') # tmp

    assert(sample.isfinite())

    return sample

if __name__ == '__main__':
    rospy.init_node('rosbag_to_sample', anonymous=True)

    yaml_path = os.path.join(FOLDER, 'params_bebop2d.yaml')
    load_params(yaml_path)

    samples = [rosbag_to_sample(os.path.join(FOLDER, f))
               for f in sorted(os.listdir(FOLDER)) if os.path.splitext(f)[1] == '.bag']
    samples = filter(lambda x: x is not None, samples)
    Sample.save(os.path.join(FOLDER, 'samples.npz'), samples)
