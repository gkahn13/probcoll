import os, sys, pickle

import rospy

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from general.algorithm.prediction.analyze import Analyze

from robots.bebop2d.algorithm.prediction.probcoll_model_bebop2d import ProbcollModelBebop2d
from robots.bebop2d.algorithm.prediction.cost_probcoll_bebop2d import CostProbcollBebop2d
from robots.bebop2d.traj_opt.ilqr.cost.cost_velocity_bebop2d import cost_velocity_bebop2d
from robots.bebop2d.dynamics.dynamics_bebop2d import DynamicsBebop2d
from robots.bebop2d.world.world_bebop2d import WorldBebop2d
from robots.bebop2d.agent.agent_bebop2d import AgentBebop2d
from robots.bebop2d.traj_opt.traj_opt_bebop2d import TrajoptBebop2d
from robots.bebop2d.policy.primitives_mpc_policy_bebop2d import PrimitivesMPCPolicyBebop2d

from rll_quadrotor.policy.cem_mpc_policy import CEMMPCPolicy
from general.policy.noise_models import ZeroNoise

from config import params

class AnalyzeBebop2d(Analyze):

    def __init__(self, on_replay=False):
        rospy.init_node('analyze_bebop2d', anonymous=True)

        Analyze.__init__(self, on_replay=on_replay, parent_exp_dir='/media/gkahn/ExtraDrive1/data/')

        self._world = WorldBebop2d(None, wp=params['world'])
        self._dynamics = DynamicsBebop2d()
        self._agent = AgentBebop2d(self._dynamics)
        self._trajopt = TrajoptBebop2d(self._dynamics, self._world, self._agent)

    def _create_mpc(self, cost_cp):
        # planner_type = params['prediction']['dagger']['planner_type']
        planner_type = 'primitives' # TODO
        if planner_type == 'primitives':
            additional_costs = []
            mpc_policy = PrimitivesMPCPolicyBebop2d(self._trajopt,
                                                    cost_cp,
                                                    additional_costs=additional_costs,
                                                    meta_data=params,
                                                    use_threads=False,
                                                    plot=True,
                                                    epsilon_greedy=params['prediction']['dagger']['epsilon_greedy'])
        elif planner_type == 'cem':
            additional_costs = []
            if 'cost_velocity' in params['trajopt']:
                additional_costs.append(cost_velocity_bebop2d(params['mpc']['H'],
                                                              params['trajopt']['cost_velocity']['velocity'],
                                                              params['trajopt']['cost_velocity']['weights'],
                                                              weight_scale=1.0))
            else:
                raise Exception('No additional cost function in yaml file')
            costs = [cost_cp] + additional_costs
            mpc_policy = CEMMPCPolicy(None,
                                      self._dynamics,
                                      costs,
                                      meta_data=params)
        else:
            raise NotImplementedError('planner_type {0} not implemented for bebop2d'.format(planner_type))

        return mpc_policy

    #######################
    ### Data processing ###
    #######################

    def _evaluate_prediction_trajectory_model(self):
        """
        Evaluates models on samples
        Returns samples with actions replaced with the MPC chosen actions
        """
        itrs_samples = self._load_samples()
        # itrs_samples = [samples[60:63] for samples in itrs_samples] # TODO temp

        itrs_mpc_samples = [[] for _ in itrs_samples]

        for itr, samples in enumerate(itrs_samples):

            ### load NN
            model_file = self._itr_model_file(itr)
            if not ProbcollModelBebop2d.checkpoint_exists(model_file):
                break
            bootstrap = ProbcollModelBebop2d(read_only=True, finalize=False)
            bootstrap.load(model_file=model_file)

            ### create MPC
            cp_params = params['prediction']['dagger']['cost_probcoll']
            cost_cp = CostProbcollBebop2d(bootstrap,
                                            weight=float(cp_params['weight']),
                                            eval_cost=cp_params['eval_cost'],
                                            pre_activation=cp_params['pre_activation'])
            mpc_policy = self._create_mpc(cost_cp)

            ### take actions
            noise = ZeroNoise(params)
            for i, sample in enumerate(samples):
                self._logger.info('Evaluating itr {0} sample {1}'.format(itr, i))
                mpc_sample = sample.copy()
                for t, (x_t, o_t) in enumerate(zip(sample.get_X(), sample.get_O())):
                    u_t = mpc_policy.act(x_t, o_t, t, noise=noise)
                    mpc_sample.set_U(u_t, t=t)
                itrs_mpc_samples[itr].append(mpc_sample)

            bootstrap.close()

        return itrs_samples, itrs_mpc_samples

    #############
    ### Files ###
    #############

    def _plot_pred_traj_file(self, itr, sample_num):
        return os.path.join(self._save_dir, self._image_folder, 'pred_traj_itr{0}_sample{1}.png'.format(itr, sample_num))

    ################
    ### Plotting ###
    ################

    def _plot_statistics(self):
        ### get samples
        samples_itrs = self._load_samples()
        itrs = np.arange(len(samples_itrs))

        ### pkl dict to save to
        pkl_dict = {}

        ### 4 plots
        f, axes = plt.subplots(4, 1, sharex=True, figsize=(15,15))

        ### durations
        durations_itrs = [[s._T * params['dt'] for s in samples] for samples in samples_itrs]
        dur_means = [np.mean(durations) for durations in durations_itrs]
        dur_stds = [np.std(durations) for durations in durations_itrs]
        if len(itrs) > 1:
            axes[0].errorbar(itrs, dur_means, yerr=dur_stds)
        else:
            axes[0].errorbar([-1]+list(itrs), dur_means*2, yerr=dur_stds*2)
        axes[0].set_title('Time until crash')
        axes[0].set_ylabel('Duration (s)')
        pkl_dict['durations'] = durations_itrs

        ### crash speeds
        crash_speed_itrs = []
        crash_speed_means = []
        crash_speed_stds = []
        for samples in samples_itrs:
            crashes = [s._T < params['prediction']['dagger']['T'] - 1 for s in samples]
            final_speeds = [np.linalg.norm(s.get_U(t=-1)) for s in samples]
            speeds = [final_speed for crash, final_speed in zip(crashes, final_speeds) if crash]
            crash_speed_itrs.append(speeds)
            if len(speeds) > 0:
                crash_speed_means.append(np.mean(speeds))
                crash_speed_stds.append(np.std(speeds))
            else:
                crash_speed_means.append(0)
                crash_speed_stds.append(0)
        if len(itrs) > 1:
            axes[1].errorbar(itrs, crash_speed_means, crash_speed_stds)
        else:
            axes[1].errorbar([-1]+list(itrs), crash_speed_means*2, crash_speed_stds*2)
        axes[1].set_title('Speed at crash')
        axes[1].set_ylabel('Speed (m/s)')
        pkl_dict['crash_speeds'] = crash_speed_itrs

        ### cumulative energy
        crash_energies = []
        for samples in samples_itrs:
            crashes = [s._T < params['prediction']['dagger']['T'] - 1 for s in samples]
            final_speeds = [np.linalg.norm(s.get_U(t=-1)) for s in samples]
            crash_speeds = [crash * final_speed for crash, final_speed in zip(crashes, final_speeds)]
            crash_energies.append(np.sum(crash_speeds) * np.sum(crash_speeds))
        cum_energy = np.cumsum(crash_energies)
        if len(itrs) > 1:
            axes[2].plot(itrs, cum_energy)
        else:
            axes[2].plot([-1]+list(itrs), list(cum_energy)*2)
        axes[2].set_title('Cumulative energy from crashing')
        axes[2].set_ylabel('Energy')

        x_vels_itrs = []
        for samples_itr in samples_itrs:
            x_vels_itr = []
            for s in samples_itr:
                x_vels_itr += abs(s.get_U()[:,0]).tolist()
            x_vels_itrs.append(x_vels_itr)
        x_vel_min = np.hstack(x_vels_itrs).min()
        x_vel_max = np.hstack(x_vels_itrs).max()
        num_bins = 11
        bins = np.linspace(x_vel_min, x_vel_max, num_bins)
        x_vels_hist_itrs = []
        for x_vels_itr in x_vels_itrs:
            x_vels_hist_itr, _ = np.histogram(x_vels_itr, bins, weights=(1./len(x_vels_itr))*np.ones(len(x_vels_itr)))
            x_vels_hist_itrs.append(x_vels_hist_itr)
        bar_width = 0.8 / len(bins)
        for i, (speed, pcts) in enumerate(zip(bins, np.array(x_vels_hist_itrs).T)):
            axes[3].bar(itrs + i*bar_width, pcts, bar_width,
                        color=cm.jet(i/float(len(bins))), label='{0:.2f}'.format(speed))
        axes[3].legend(bbox_to_anchor=(1.1, 1.05))
        axes[3].set_title('Primitives histogram by speed')
        axes[3].set_xlabel('Iteration')
        axes[3].set_ylabel('Pct')
        pkl_dict['U'] = [[s.get_U() for s in samples_itr] for samples_itr in samples_itrs]

        plt.show(block=False)
        plt.pause(0.1)
        if not os.path.exists(os.path.dirname(self._plot_stats_file)):
            os.makedirs(os.path.dirname(self._plot_stats_file))
        f.savefig(self._plot_stats_file)

        with open(self._plot_stats_file_pkl, 'w') as f:
            pickle.dump(pkl_dict, f)

    def _plot_prediction_trajectory_model(self):
        itrs_samples, itrs_mpc_samples = self._evaluate_prediction_trajectory_model()

        for itr, (samples, mpc_samples) in enumerate(zip(itrs_samples, itrs_mpc_samples)):
            for i, (sample, mpc_sample) in enumerate(zip(samples, mpc_samples)):
                T = len(sample.get_U())
                f, axes = plt.subplots(2, T, figsize=(35, 7))
                mng = plt.get_current_fig_manager()
                mng.window.showMinimized()

                ### show image
                for ax, o in zip(axes[0, :], sample.get_O(sub_obs='camera')):
                    o = o.reshape((params['O']['camera']['height'], params['O']['camera']['width']))
                    ax.imshow(o, cmap='gray')
                ### show controls
                for ax, u, u_mpc in zip(axes[1, :], sample.get_U(), mpc_sample.get_U()):
                    ax.arrow(0, 0, -u[1], u[0], fc='k', ec='k', width=0.003)
                    ax.arrow(0, 0, -u_mpc[1], u_mpc[0], fc='r', ec='r', width=0.003)
                    ax.set_xlim((-0.4, 0.4))
                    ax.set_ylim((-0.1, 0.7))
                    ax.set_aspect('equal')

                f.suptitle('Itr {0} Sample {1}'.format(itr, i))

                plt.show(block=False)
                plt.pause(0.05)
                f.savefig(self._plot_pred_traj_file(itr, i), dpi=100)

                plt.close(f)

    ###########
    ### Run ###
    ###########

    def run(self, plot_single, plot_traj, plot_samples, plot_groundtruth):
        self._plot_statistics()
        if plot_traj:
            self._plot_prediction_trajectory_model()

        rospy.signal_shutdown('')
