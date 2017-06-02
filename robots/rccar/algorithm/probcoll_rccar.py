import os
import rospy
import std_msgs
import multiprocessing
import subprocess
import os
import signal

from robots.rccar.tf.planning.planner_primitives_rccar import PlannerPrimitivesRCcar
from general.tf.planning.planner_random import PlannerRandom
from general.tf.planning.planner_cem import PlannerCem
from general.algorithm.probcoll import Probcoll
from general.policy.open_loop_policy import OpenLoopPolicy
#from robots.rccar.algorithm.cost_probcoll_rccar import CostProbcollRCcar
from robots.rccar.algorithm.probcoll_model_rccar import ProbcollModelRCcar
import robots.rccar.ros.ros_utils as ros_utils
from general.state_info.conditions import Conditions
from general.state_info.sample import Sample
from robots.rccar.agent.agent_rccar import AgentRCcar
from robots.rccar.dynamics.dynamics_rccar import DynamicsRCcar
#from robots.rccar.planning.primitives_rccar import PrimitivesRCcar
#from robots.rccar.planning.cost.cost_velocity_rccar import cost_velocity_rccar
from robots.rccar.world.world_rccar import WorldRCcar
from config import params

class ProbcollRCcar(Probcoll):

    def __init__(self, read_only=False):
        Probcoll.__init__(self, read_only=read_only)

    def _setup(self):
        rospy.init_node('ProbcollRCcar', anonymous=True)

        self._jobs = []
        
        if params['world']['sim']:
            p = multiprocessing.Process(target=self._run_simulation)
            p.daemon = True
            self._jobs.append(p)
            p.start()
        
        probcoll_params = params['probcoll']
        world_params = params['world']
        cond_params = probcoll_params['conditions']
        self._asynchronous = probcoll_params['asynchronous_training']
        self._max_iter = probcoll_params['max_iter']
        self._dynamics = DynamicsRCcar() # Try to remove dynamics
        self._agent = AgentRCcar(self._dynamics)
        self._world = WorldRCcar(self._agent, self._bag_file, wp=world_params)
        self._conditions = Conditions(cond_params=cond_params)
        self._use_dynamics = False

        assert(self._world.randomize)

        ### load prediction neural net
        self._probcoll_model = ProbcollModelRCcar(read_only=self._read_only)

        self._async_on = False

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
                    "roslaunch",
                    params["sim"]["launch_file"],
                    "car_name:={0}".format(params["exp_name"]),
                    "config:={0}".format(params["config_file"]),
                    "env:={0}".format(params["sim"]["sim_env"])
                ]
            subprocess.call(command)
        except Exception as e:
            print(e)
        finally:
            self._logger.info("Ending Simulation!")

    def _close(self):
        for p in self._jobs:
            os.kill(p.pid, signal.SIGKILL)
            p.join()
        self._probcoll_model.close()
    
    ###################
    ### Run methods ###
    ###################

    def _run_training(self):
        if self._asynchronous:
            self._probcoll_model.recover()
            if not self._async_on:
                self._probcoll_model.async_training()
                self._async_on = True
        else:
            self._probcoll_model.train(reset=params['model']['reset_every_train'])
    
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
        self._logger.debug('\t\t\tCreating MPC')
        if self._planner_type == 'random':
            planner = PlannerRandom(self._probcoll_model, params['planning'])
            mpc_policy = OpenLoopPolicy(planner)
        elif self._planner_type == 'primitives':
            planner = PlannerPrimitivesRCcar(self._probcoll_model, params['planning'])
            mpc_policy = OpenLoopPolicy(planner)
        elif self._planner_type == 'cem':
            planner = PlannerCem(self._probcoll_model, params['planning'])
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

