import os
import rospy
import std_msgs
import subprocess
import os
import signal
import time
import tf as ros_tf 
import numpy as np

from robots.rccar.tf.planning.planner_primitives_rccar import PlannerPrimitivesRCcar
from robots.rccar.tf.planning.planner_random_rccar import PlannerRandomRCcar
from general.tf.planning.planner_cem import PlannerCem
from general.algorithm.probcoll import Probcoll
from general.policy.open_loop_policy import OpenLoopPolicy
from general.policy.random_policy import RandomPolicy
from robots.rccar.algorithm.probcoll_model_rccar import ProbcollModelRCcar
from robots.rccar.ros import ros_utils
from general.state_info.conditions import Conditions
from general.state_info.sample import Sample
from robots.rccar.agent.agent_rccar import AgentRCcar
from robots.rccar.dynamics.dynamics_rccar import DynamicsRCcar
from robots.rccar.world.world_rccar import WorldRCcar
from config import params

class ProbcollRCcar(Probcoll):

    def __init__(self, save_dir=None, data_dir=None):
        Probcoll.__init__(self, save_dir=save_dir, data_dir=data_dir)

    def _setup(self):
        rospy.init_node('ProbcollRCcar', anonymous=True)

        if params['world']['sim']:
            self._run_simulation() 
        
        probcoll_params = params['probcoll']
        world_params = params['world']
        cond_params = probcoll_params['conditions']
        self._max_iter = probcoll_params['max_iter']
        self._dynamics = DynamicsRCcar() # Try to remove dynamics
        self._agent = AgentRCcar(self._dynamics)
        self._world = WorldRCcar(self._agent, self._bag_file, wp=world_params)
        self._conditions = Conditions(cond_params=cond_params)
        self._use_dynamics = False

        assert(self._world.randomize)

        ### load prediction neural net
        self.probcoll_model = ProbcollModelRCcar(save_dir=self._save_dir, data_dir=self._data_dir)
        rccar_topics = params['rccar']['topics']
        self.coll_callback = ros_utils.RosCallbackEmpty(rccar_topics['collision'], std_msgs.msg.Empty)
        self.good_rollout_callback = ros_utils.RosCallbackEmpty(rccar_topics['good_rollout'], std_msgs.msg.Empty)
        self.bad_rollout_callback = ros_utils.RosCallbackEmpty(rccar_topics['bad_rollout'], std_msgs.msg.Empty)
    
    ##########################
    ### Threaded Functions ###
    ##########################

    def _run_simulation(self):
        try:
            command = [
                    "python",
                    '{0}/{1}.py'.format(params['sim']['sim_path'], params['sim']['sim_env']),
                    "--yaml_path",
                    params['sim']['config_file']
                ]
            p = subprocess.Popen(command)
            self._jobs.append(p)
        except Exception as e:
            self._logger.warning('Error starting simulation!')
            self._logger.warning(e)
   
    def _async_training(self):
        try:
            p = subprocess.Popen(
                ["python", "main.py", "train", "rccar", "--asynch"])
            self._jobs.append(p)
        except Exception as e:
            self._logger.warning('Error starting async training!')
            self._logger.warning(e)

    def close(self):
        try:
            self._agent.close()
        except rospy.service.ServiceException:
            self._logger.info("Service Ended")
        rospy.signal_shutdown("Probcoll is done")
        for p in self._jobs:
            p.kill()
        self.probcoll_model.close()
    
    ###################
    ### Run methods ###
    ###################

    def run_testing(self, itr):
        if (itr != 0 and (itr == self._max_iter - 1 \
                or itr % params['world']['testing']['itr_freq'] == 0)): 
            self._logger.info('Itr {0} testing'.format(itr))
            if self._agent.sim:
                if self._async_on:
                    self._logger.debug('Recovering probcoll model')
                    self.probcoll_model.recover()
                T = params['probcoll']['T']
                conditions = []
                if params['world']['testing'].get('position_ranges', None) is not None:
                    ranges = params['world']['testing']['position_ranges']
                    num_pos = params['world']['testing']['num_pos']
                    if params['world']['testing']['range_type'] == 'random':
                        for _ in xrange(num_pos):
                            ran = ranges[np.random.randint(len(ranges))]
                            conditions.append(np.random.uniform(ran[0], ran[1]))
                    elif params['world']['testing']['range_type'] == 'fix_spacing':
                        num_ran = len(ranges)
                        num_per_ran = num_pos//num_ran
                        for i in xrange(num_ran):
                            ran = ranges[i]
                            low = np.array(ran[0])
                            diff = np.array(ran[1]) - np.array(ran[0])
                            for j in xrange(num_per_ran):
                                val = diff * ((j + 0.0)/num_per_ran) + low
                                conditions.append(val) 
                elif params['world']['testing'].get('positions', None) is not None:
                    conditions = params['world']['testing']['positions']
                samples = []
                reset_pos, reset_quat = self._agent.get_sim_state_data() 
                for cond in xrange(len(conditions)):
                    self._logger.info('\t\tTesting cond {0} itr {1}'.format(cond, itr))
                    start = time.time()
                    pos_ori = conditions[cond]
                    pos = pos_ori[:3]
                    quat = ros_tf.transformations.quaternion_from_euler(pos_ori[3], 0, 0)
                    self._agent.execute_control(None, reset=True, pos=pos, quat=quat)
                    x0 = self._conditions.get_cond(0)
                    sample_T = Sample(meta_data=params, T=T)
                    sample_T.set_X(x0, t=0)
                    for t in xrange(T):
                        x0 = sample_T.get_X(t=t)

                        rollout, rollout_no_noise = self._agent.sample_policy(x0, self._mpc_policy, 0., T=1, use_noise=False)
                        
                        o = rollout.get_O(t=0)
                        u = rollout_no_noise.get_U(t=0)

                        sample_T.set_U(u, t=t)
                        sample_T.set_O(o, t=t)
                        
                        if not self._use_dynamics:
                            sample_T.set_X(rollout.get_X(t=0), t=t)
                        
                        if self._world.is_collision(sample_T, t=t):
                            self._logger.warning('\t\t\tCrashed at t={0}'.format(t))
                            break

                        if self._use_dynamics:
                            if t < T-1:
                                x_tp1 = self._dynamics.evolve(x0, u)
                                sample_T.set_X(x_tp1, t=t+1)

                    else:
                        self._logger.info('\t\t\tLasted for t={0}'.format(t))

                    sample = sample_T.match(slice(0, t + 1))

                    if not self._is_good_rollout(sample, t):
                        self._logger.warning('\t\t\tNot good rollout. Repeating rollout.'.format(t))
                        continue

                    samples.append(sample)

                    assert(samples[-1].isfinite())
                    elapsed = time.time() - start
                    self._logger.info('\t\t\tFinished cond {0} of testing ({1:.1f}s, {2:.3f}x real-time)'.format(
                        cond,
                        elapsed,
                        t*params['dt']/elapsed))

                self._itr_save_samples(itr, samples, prefix='testing_')
                self._agent.execute_control(None, reset=True, pos=reset_pos, quat=reset_quat)
            else:
                pass

    ####################
    ### Save methods ###
    ####################

    def _bag_file(self, itr, cond, rep, create=True):
        return os.path.join(self._itr_dir(itr, create=create), 'bagfile_itr{0}_cond{1}_rep{2}.bag'.format(itr, cond, rep))

    def _itr_save_worlds(self, itr, world_infos):
        pass

    #####################
    ### World methods ###
    #####################

    def _reset_world(self, itr, cond, rep):
        if self._agent.sim:
            back_up = params["world"]["do_back_up"] 
        else:
            if cond == 0 and rep == 0:
                self._logger.info('Press A or B to start')
                self._ros_is_good_rollout()
            back_up = self.coll_callback.get() is not None # only back up if experienced a crash
        self._world.reset(back_up, itr=itr, cond=cond, rep=rep)

    def _update_world(self, sample, t):
        return

    def _is_good_rollout(self, sample, t):
        if self._agent.sim:
            return True
        else:
            self._agent.execute_control(None) # stop the car
            self._logger.info('Is good rollout? (A for yes, B for no)')
            return self._ros_is_good_rollout()

    def _ros_is_good_rollout(self):
        self.good_rollout_callback.get()
        self.bad_rollout_callback.get()
        while not rospy.is_shutdown():
            good_rollout = self.good_rollout_callback.get()
            bad_rollout = self.bad_rollout_callback.get()
            if good_rollout and not bad_rollout:
                return True
            elif bad_rollout and not good_rollout:
                return False
            rospy.sleep(0.1)

    #########################
    ### Create controller ###
    #########################

    def _create_mpc(self):
        """ Must initialize MPC """
        self._logger.info('\t\t\tCreating MPC')
        if self._planner_type == 'random_policy':
            mpc_policy = RandomPolicy()
        elif self._planner_type == 'random':
            planner = PlannerRandomRCcar(self.probcoll_model, params['planning'])
            mpc_policy = OpenLoopPolicy(planner)
        elif self._planner_type == 'primitives':
            planner = PlannerPrimitivesRCcar(self.probcoll_model, params['planning'])
            mpc_policy = OpenLoopPolicy(planner)
        elif self._planner_type == 'cem':
            planner = PlannerCem(self.probcoll_model, params['planning'])
            mpc_policy = OpenLoopPolicy(planner)
        else:
            raise NotImplementedError('planner_type {0} not implemented for rccar'.format(self._planner_type))

        return mpc_policy

    ####################
    ### Info methods ###
    ####################

    def _get_world_info(self):
        ### just returns empty dict, but function call terminates bag recording
        return self._world.get_info()

