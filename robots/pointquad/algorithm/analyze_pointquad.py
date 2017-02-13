import os, copy, pickle

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from general.algorithm.analyze import Analyze
import general.utility.colormaps as cmaps

from robots.pointquad.algorithm.probcoll_model_pointquad import ProbcollModelPointquad
from robots.pointquad.dynamics.dynamics_pointquad import DynamicsPointquad
from robots.pointquad.world.world_pointquad import WorldPointquad
from robots.pointquad.agent.agent_pointquad import AgentPointquad
# from robots.pointquad.policy.primitives_mpc_policy_pointquad import PrimitivesMPCPolicyPointquad
from robots.pointquad.algorithm.cost_probcoll_pointquad import CostProbcollPointquad

from general.state_info.sample import Sample
from general.policy.noise_models import ZeroNoise

from config import params

class AnalyzePointquad(Analyze):

    def __init__(self, on_replay=False):
        Analyze.__init__(self, on_replay=on_replay, parent_exp_dir=None)#'/media/gkahn/ExtraDrive1/data/')

        self._world = WorldPointquad()
        self._world.clear()

        self._dynamics = DynamicsPointquad()
        self._agent = AgentPointquad(self._world, self._dynamics, dyn_noise=params['probcoll']['dyn_noise'])

    #######################
    ### Data processing ###
    #######################

    def _evaluate_probcoll_model(self):
        ### get prediction model evaluations for all iterations
        itr = 0
        itrs_Xs, itrs_prob_means, itrs_prob_stds, itrs_costs = [], [], [], []
        while True:
            ### load NN
            model_file = self._itr_model_file(itr)
            if not os.path.exists(model_file):
                break
            bootstrap = ProbcollModelPointquad(read_only=True, finalize=False)
            bootstrap.load(model_file=model_file)

            wd = self._load_world_file(itr, 0, 0)
            self._world.env.clear()
            for pose, radius, height in zip(wd['cyl_poses'], wd['cyl_radii'], wd['cyl_heights']):
                self._world.env.add_cylinder(pose, radius, height)
            for pose, extents in zip(wd['box_poses'], wd['box_extents']):
                self._world.env.add_box(pose, extents)

            default_sample = Sample(meta_data=params, T=bootstrap.T)
            for k, v in params['probcoll']['conditions']['default'].items():
                default_sample.set_X(v, t=slice(0, bootstrap.T), sub_state=k)

            ### evaluate on grid
            dx, dy = 30, 30
            positions = np.dstack(np.meshgrid(np.linspace(-0.5, 3.0, dx),
                                             np.linspace(1.0, -1.0, dy))).reshape(dx*dy, 2)

            samples = []
            for pos in positions:
                s = copy.deepcopy(default_sample)

                pos = np.concatenate((pos, [s.get_X(t=0, sub_state='position')[-1]]))
                s.set_X(pos, t=slice(0, bootstrap.T), sub_state='position')
                s.set_O(self._agent.get_observation(s.get_X(t=0), noise=False), t=slice(0, bootstrap.T))
                s.set_U(np.zeros((bootstrap.T, bootstrap.dU)), t=slice(0, bootstrap.T))

                samples.append(s)

            cp_params = params['probcoll']['cost_probcoll']
            eval_cost = cp_params['eval_cost']
            pre_activation = cp_params['pre_activation']
            probs_mean_batch, probs_std_batch = bootstrap.eval_sample_batch(samples,
                                                                            num_avg=10,
                                                                            pre_activation=pre_activation)

            bootstrap.close()

            sigmoid = lambda x: (1. / (1. + np.exp(-x)))
            costs = []
            for probs_mean, probs_std in zip(probs_mean_batch, probs_std_batch):
                cost = 0.
                for t in [bootstrap.T - 1]:
                    speed = 1.
                    cost += eval(eval_cost)
                costs.append(cost / bootstrap.T)
            costs = np.array(costs)

            # if itr == 3:
            #     import IPython; IPython.embed()

            Xs = np.array([s.get_X(t=0, sub_state='position')[:2] for s in samples]).reshape(dx, dy, 2)
            itrs_Xs.append(Xs)
            if pre_activation:
                probs_mean_batch = sigmoid(probs_mean_batch)
                # probs_std_batch = sigmoid(probs_std_batch)
                probs_std_batch /= probs_std_batch.max()
            itrs_prob_means.append(probs_mean_batch.max(axis=1).reshape(dx, dy))
            itrs_prob_stds.append(probs_std_batch.max(axis=1).reshape(dx, dy))
            itrs_costs.append(costs.reshape(dx, dy))

            # if itr >= 3:
            #     break

            itr += 1

        return itrs_Xs, itrs_prob_means, itrs_prob_stds, itrs_costs

    def _evaluate_prediction_trajectory_model(self):
        ### get prediction model evaluations for all iterations
        itr = 0
        itrs_Xs, itrs_thetas, itrs_speeds, itrs_prob_means, itrs_prob_stds, itrs_costs = [], [], [], [], [], []
        while True:
            ### load NN
            model_file = self._itr_model_file(itr)
            if not os.path.exists(model_file):
                break
            bootstrap = ProbcollModelPointquad(read_only=True, finalize=False)
            bootstrap.load(model_file=model_file)

            wd = self._load_world_file(itr, 0, 0)
            self._world.env.clear()
            for pose, radius, height in zip(wd['cyl_poses'], wd['cyl_radii'], wd['cyl_heights']):
                self._world.env.add_cylinder(pose, radius, height)
            for pose, extents in zip(wd['box_poses'], wd['box_extents']):
                self._world.env.add_box(pose, extents)

            ### create MPC
            cp_params = params['probcoll']['cost_probcoll']
            cost_cp = CostProbcollPointquad(bootstrap, None,
                                              weight=float(cp_params['weight']),
                                              eval_cost=cp_params['eval_cost'],
                                              pre_activation=cp_params['pre_activation'])
            mpc_policy = PrimitivesMPCPolicyPointquad(self._trajopt,
                                                    cost_cp,
                                                    additional_costs=[],
                                                    meta_data=params,
                                                    use_threads=False,
                                                    plot=False)

            default_sample = Sample(meta_data=params, T=bootstrap.T)
            for k, v in params['probcoll']['conditions']['default'].items():
                default_sample.set_X(v, t=slice(0, bootstrap.T), sub_state=k)

            ### evaluate on grid
            dx, dy = 20, 20
            samples = []
            itr_costs = []
            thetas, speeds = [], []
            for i, xpos in enumerate(np.linspace(-0.5, 1.5, dx)):
                for j, ypos in enumerate(np.linspace(0.6, -0.6, dy)):
                    s = copy.deepcopy(default_sample)

                    pos = [xpos, ypos, s.get_X(t=0, sub_state='position')[-1]]
                    s.set_X(pos, t=slice(0, bootstrap.T), sub_state='position')
                    s.set_O(self._agent.get_observation(s.get_X(t=0), noise=False), t=slice(0, bootstrap.T))
                    x = s.get_X(t=0)
                    o = s.get_O(t=0)

                    # assume straight line trajectories
                    p_samples = []
                    p_speeds, p_thetas = [], []
                    for primitive in mpc_policy._mpc_policies:
                        p_speeds.append(primitive.speed)
                        p_thetas.append(primitive.theta)
                        primitive.act(x, o, t=0, noise=ZeroNoise(params))
                        p_samples.append(primitive._curr_traj)

                    prim_costs = mpc_policy._primitives_cost(p_samples)
                    min_cost_idx = np.argmin(prim_costs)

                    samples.append(p_samples[min_cost_idx])
                    itr_costs.append(prim_costs[min_cost_idx])
                    thetas.append(p_thetas[min_cost_idx])
                    speeds.append(p_speeds[min_cost_idx])


            ### evaluate on grid
            cp_params = params['probcoll']['cost_probcoll']
            eval_cost = cp_params['eval_cost']
            pre_activation = cp_params['pre_activation']
            probs_mean_batch, probs_std_batch = bootstrap.eval_sample_batch(samples,
                                                                            num_avg=10,
                                                                            pre_activation=pre_activation)

            bootstrap.close()

            sigmoid = lambda x: (1. / (1. + np.exp(-x)))
            # costs = []
            # for probs_mean, probs_std in zip(probs_mean_batch, probs_std_batch):
            #     cost = 0.
            #     for t in [bootstrap.T - 1]:
            #         speed = 1.
            #         cost += eval(eval_cost)
            #     costs.append(cost / bootstrap.T)
            # costs = np.array(costs)

            itrs_Xs.append(np.array([s.get_X(t=0, sub_state='position')[:2] for s in samples]))
            itrs_thetas.append(np.array(thetas))
            itrs_speeds.append(np.array(speeds) / (params['trajopt']['cost_velocity']['velocity'][0] * dx * dy))  # normalize so fits into grid density
            # itrs_speeds.append(np.ones(len(speeds), dtype=float) / (dx * dy)) # TODO
            if pre_activation:
                probs_mean_batch = sigmoid(probs_mean_batch)
                # probs_std_batch = sigmoid(probs_std_batch)
                probs_std_batch /= probs_std_batch.max()
            itrs_prob_means.append(np.array(probs_mean_batch))
            itrs_prob_stds.append(np.array(probs_std_batch))
            itrs_costs.append(np.array(itr_costs))

            itr += 1

            # if itr >= 2: # TODO
            #     break

        return itrs_Xs, itrs_thetas, itrs_speeds, itrs_prob_means, itrs_prob_stds, itrs_costs

    def _evaluate_probcoll_model_on_samples(self, bootstrap, itr, samples, mpcs):
        ### load NN
        H = params['model']['T']
        model_file = self._itr_model_file(itr)
        if not ProbcollModelPointquad.checkpoint_exists(model_file):
            return [None]*len(samples), [None]*len(samples)

        bootstrap.load(model_file=model_file)

        means, stds = [], []
        for sample, mpc in zip(samples, mpcs):
            eval_samples = []
            for o_t, U in zip(sample.get_O(), mpc['controls']):
                eval_sample = Sample(T=H, meta_data=params)
                eval_sample.set_U(U, t=slice(0, H))
                eval_sample.set_O(o_t, t=0)
                eval_samples.append(eval_sample)

            mean, std = bootstrap.eval_sample_batch(eval_samples, num_avg=10, pre_activation=True)
            means.append(mean)
            stds.append(std)

        return means, stds

    def _evaluate_probcoll_model_groundtruth(self):
        """
        For each primitive, execute the primitive for T timesteps to obtain ground truth, and compare label
        """
        ### get prediction model evaluations for all iterations
        itr = 0
        itrs_gt_crashes, itrs_pred_crashes = [], []
        positions, speeds_angles = None, None
        while True:
            ### load NN
            model_file = self._itr_model_file(itr)
            if not os.path.exists(model_file):
                break
            bootstrap = ProbcollModelPointquad(read_only=True, finalize=False)
            bootstrap.load(model_file=model_file)

            wd = self._load_world_file(itr, 0, 0)
            self._world.env.clear()
            for pose, radius, height in zip(wd['cyl_poses'], wd['cyl_radii'], wd['cyl_heights']):
                self._world.env.add_cylinder(pose, radius, height)
            for pose, extents in zip(wd['box_poses'], wd['box_extents']):
                self._world.env.add_box(pose, extents)

            ### create MPC
            cp_params = params['probcoll']['cost']
            cost_cp = CostProbcollPointquad(bootstrap, None,
                                              weight=float(cp_params['weight']),
                                              eval_cost=cp_params['eval_cost'],
                                              pre_activation=cp_params['pre_activation'])
            mpc_policy = PrimitivesMPCPolicyPointquad(self._trajopt,
                                                      cost_cp,
                                                      additional_costs=[],
                                                      meta_data=params,
                                                      use_threads=False,
                                                      plot=False)

            default_sample = Sample(meta_data=params, T=bootstrap.T)
            for k, v in params['probcoll']['conditions']['default'].items():
                default_sample.set_X(v, t=slice(0, bootstrap.T), sub_state=k)

            ### speeds/angles
            speeds_angles = [(p.speed, p.theta) for p in mpc_policy._mpc_policies]
            positions = []

            ### evaluate on grid
            dx, dy = 12, 5
            rollouts = []
            for i, xpos in enumerate(np.linspace(-0.5, 1.5, dx)):
                for j, ypos in enumerate(np.linspace(0.6, -0.6, dy)):
                    s = copy.deepcopy(default_sample)

                    pos = [xpos, ypos, s.get_X(t=0, sub_state='position')[-1]]
                    s.set_X(pos, t=slice(0, bootstrap.T), sub_state='position')
                    s.set_O(self._agent.get_observation(s.get_X(t=0), noise=False), t=slice(0, bootstrap.T))

                    if np.sum(s.get_O(sub_obs='collision', t=0)) > 0:
                        continue

                    positions.append((xpos, ypos))

                    x0 = s.get_X(t=0)
                    rollouts_ij = [self._agent.sample_policy(x0, p, noise=ZeroNoise(params), T=bootstrap.T)
                                   for p in mpc_policy._mpc_policies]
                    rollouts.append(rollouts_ij)

            rollouts = np.array(rollouts)
            gt_crashes = np.array([r.get_O(sub_obs='collision').sum() > 0 for r in rollouts.ravel()]).reshape(rollouts.shape)

            sigmoid = lambda x: 1. / (1. + np.exp(-x))
            pred_crashes = np.array([sigmoid(mean) > 0.5 for mean in bootstrap.eval_sample_batch(rollouts.ravel(),
                                                                                                 num_avg=10,
                                                                                                 pre_activation=True)[0]]).reshape(rollouts.shape)

            bootstrap.close()

            itrs_gt_crashes.append(gt_crashes)
            itrs_pred_crashes.append(pred_crashes)

            itr += 1

        return itrs_gt_crashes, itrs_pred_crashes, positions, speeds_angles

    ################
    ### Plotting ###
    ################

    def _plot_statistics(self):
        ### get samples and stats
        samples_itrs = self._load_samples()
        itrs = np.arange(len(samples_itrs))

        ### pkl dict to save to
        pkl_dict = {}

        ### 5 plots
        f, axes = plt.subplots(5, 1, sharex=True, figsize=(15,15))

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
            crashes = [s._T < params['probcoll']['T'] - 1 for s in samples]
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
        crash_speeds = []
        for samples in samples_itrs:
            crashes = [s._T < params['probcoll']['T'] - 1 for s in samples]
            final_speeds = [np.linalg.norm(s.get_U(t=-1)) for s in samples]
            crash_speeds.append([crash * final_speed for crash, final_speed in zip(crashes, final_speeds)])
        crash_speeds = np.array(crash_speeds)
        cum_energy = (crash_speeds * crash_speeds).sum(axis=1).cumsum()
        if len(itrs) > 1:
            axes[2].plot(itrs, cum_energy)
        else:
            axes[2].plot([-1]+list(itrs), list(cum_energy)*2)
        axes[2].set_title('Cumulative energy from crashing')
        axes[2].set_ylabel('Energy')

        x_dists_itrs = []
        dist_means, dist_stds = [], []
        for samples_itr in samples_itrs:
            x_dists = [s.get_X(t=-1, sub_state='position')[0] - s.get_X(t=0, sub_state='position')[0] for s in samples_itr]
            x_dists_itrs.append(x_dists)
            dist_means.append(np.mean(x_dists))
            dist_stds.append(np.std(x_dists))
        if len(itrs) > 1:
            axes[3].errorbar(itrs, dist_means, dist_stds)
        else:
            axes[3].errorbar([-1]+list(itrs), dist_means*2, dist_stds*2)
        axes[3].set_title('Horizontal distance travelled')
        axes[3].set_ylabel('Distance (m)')
        pkl_dict['x_dists'] = x_dists_itrs

        # speeds = [val[0] for val in stats[0]['speeds_pcts']]
        # bar_width = 0.8 / len(speeds)
        # for i, speed in enumerate(speeds):
        #     pcts = [stat['speeds_pcts'][i][1] for stat in stats]
        #     axes[4].bar(itrs + i*bar_width, pcts, bar_width,
        #                 color=cm.jet(i/float(len(speeds))), label='{0:.2f}'.format(speed))

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
            axes[4].bar(itrs + i*bar_width, pcts, bar_width,
                        color=cm.jet(i/float(len(bins))), label='{0:.2f}'.format(speed))
        axes[4].legend(bbox_to_anchor=(1.1, 1.05))
        axes[4].set_title('Primitives histogram by speed')
        axes[4].set_xlabel('Iteration')
        axes[4].set_ylabel('Pct')
        pkl_dict['U'] = [[s.get_U() for s in samples_itr] for samples_itr in samples_itrs]

        # mpcs_itrs = self._load_mpcs()
        # bootstrap = ProbcollModelPointquad(read_only=True, finalize=False)
        # means_itrs, stds_itrs = [], []
        # for i, (samples, mpcs) in enumerate(zip(samples_itrs, mpcs_itrs)):
        #     means, stds = self._evaluate_probcoll_model_on_samples(bootstrap, i, samples, mpcs)
        #     means_itrs.append(means)
        #     stds_itrs.append(stds)
        # bootstrap.close()
        # pkl_dict['mpc_means'] = means_itrs
        # pkl_dict['mpc_stds'] = stds_itrs
        # def flatten_list(l):
        #     return [val for sublist in l for val in sublist]
        # si = []
        # for s in stds_itrs:
        #     si.append(flatten_list([list(x.ravel()) for x in s]))
        # import IPython; IPython.embed()

        plt.show(block=False)
        plt.pause(0.1)
        if not os.path.exists(os.path.dirname(self._plot_stats_file)):
            os.makedirs(os.path.dirname(self._plot_stats_file))
        f.savefig(self._plot_stats_file)

        with open(self._plot_stats_file_pkl, 'w') as f:
            pickle.dump(pkl_dict, f)

    def _plot_probcoll_model(self):
        itrs_Xs, itrs_prob_means, itrs_prob_stds, itrs_costs = self._evaluate_probcoll_model()
        samples = self._load_samples()

        samples = samples[:len(itrs_Xs)]

        sqrt_num_axes = int(np.ceil(np.sqrt(len(itrs_Xs))))
        f_mean, axes_mean = plt.subplots(sqrt_num_axes, sqrt_num_axes, figsize=(15, 15))
        f_std, axes_std = plt.subplots(sqrt_num_axes, sqrt_num_axes, figsize=(15, 15))
        f_cost, axes_cost = plt.subplots(sqrt_num_axes, sqrt_num_axes, figsize=(15, 15))
        f_std_hist, axes_std_hist = plt.subplots(sqrt_num_axes, sqrt_num_axes, figsize=(15, 15))
        if not hasattr(axes_mean, '__iter__'):
            axes_mean = np.array([axes_mean])
            axes_std = np.array([axes_std])
            axes_std_hist = np.array([axes_std_hist])

        for i, (Xs, prob_means, prob_stds, costs, itr_samples) in \
                enumerate(zip(itrs_Xs, itrs_prob_means, itrs_prob_stds, itrs_costs, samples)):
            ### plot mean
            ax_mean = axes_mean.ravel()[i]
            ax_mean.imshow(1-prob_means,
                           extent=(Xs[:,:,0].min(), Xs[:,:,0].max(), Xs[:,:,1].min(), Xs[:,:,1].max()),
                           interpolation='nearest',
                           cmap=cm.gist_heat)
            ax_mean.axis('equal')

            ### plot std
            ax_std = axes_std.ravel()[i]
            ax_std.imshow(np.clip(prob_stds.max() - prob_stds, 0.0, 1.0), # cap std to 1.0
                          extent=(Xs[:,:,0].min(), Xs[:,:,0].max(), Xs[:,:,1].min(), Xs[:,:,1].max()),
                          interpolation='nearest',
                          cmap=cm.gist_heat)
            ax_std.axis('equal')

            ### plot cost
            ax_cost = axes_cost.ravel()[i]
            ax_cost.imshow(1. - np.clip(costs, 0.0, 1.0),  # cap std to 1.0
                          extent=(Xs[:, :, 0].min(), Xs[:, :, 0].max(), Xs[:, :, 1].min(), Xs[:, :, 1].max()),
                          interpolation='nearest',
                          cmap=cm.gist_heat)
            ax_cost.axis('equal')

            ### plot std hist
            ax_std_hist = axes_std_hist.ravel()[i]
            ax_std_hist.hist(prob_stds.ravel(),
                             weights=np.ones(len(prob_stds.ravel())) * 1. / len(prob_stds.ravel()),
                             bins=10)
            ax_std_hist.set_title('Itr {0}'.format(i))

            for ax in (ax_mean, ax_std, ax_cost):
                ax.set_title('Itr {0}'.format(i))

                plt.show(block=False)
                plt.pause(0.01)

                wd = self._load_worlds()[i][0]
                for pose, radius, height in zip(wd['cyl_poses'], wd['cyl_radii'], wd['cyl_heights']):
                    patch = ax.add_artist(plt.Circle(pose[:2,3], radius, edgecolor='g', facecolor='none'))
                    ax.draw_artist(patch)

            ### plot samples
            for s in itr_samples:
                colls = s.get_O(sub_obs='collision')
                positions = s.get_X(sub_state='position')[:,:2]
                coll_positions = positions[colls.ravel() == 1]
                no_coll_positions = positions[colls.ravel() == 0]

                for ax in (ax_mean, ax_std, ax_cost):
                    if len(coll_positions) > 0:
                        ax.plot(coll_positions[:,0], coll_positions[:,1], 'cx',
                                alpha=0.8, markeredgecolor='c', markeredgewidth=2.0)
                    if len(no_coll_positions) > 0:
                        ax.plot(no_coll_positions[:,0], no_coll_positions[:,1], 'o',
                                alpha=0.8, color='DarkSlateBlue', markeredgewidth=0, markersize=5.0)

        f_mean.suptitle('Bootstrap MEAN plots')
        f_std.suptitle('Bootstrap STD plots')
        f_cost.suptitle('Bootstrap COST plots')
        f_std_hist.suptitle('Bootstrap STD hist plots')

        plt.show(block=False)
        plt.pause(0.1)

        f_mean.savefig(self._plot_pred_mean_file, dpi=400)
        f_std.savefig(self._plot_pred_std_file, dpi=400)
        f_cost.savefig(self._plot_pred_cost_file, dpi=400)
        f_std_hist.savefig(self._plot_pred_std_hist_file, dpi=400)

    def _plot_prediction_trajectory_model(self):
        itrs_Xs, itrs_thetas, itrs_speeds, itrs_prob_means, itrs_prob_stds, itrs_costs = \
            self._evaluate_prediction_trajectory_model()
        samples = self._load_samples()
        samples = samples[:len(itrs_Xs)]

        sqrt_num_axes = int(np.ceil(np.sqrt(len(itrs_Xs))))
        f_mean, axes_mean = plt.subplots(sqrt_num_axes, sqrt_num_axes, figsize=(15, 15))
        f_std, axes_std = plt.subplots(sqrt_num_axes, sqrt_num_axes, figsize=(15, 15))
        f_cost, axes_cost = plt.subplots(sqrt_num_axes, sqrt_num_axes, figsize=(15, 15))
        f_std_hist, axes_std_hist = plt.subplots(sqrt_num_axes, sqrt_num_axes, figsize=(15, 15))
        if not hasattr(axes_mean, '__iter__'):
            axes_mean = np.array([axes_mean])
            axes_std = np.array([axes_std])
            axes_cost = np.array([axes_cost])
            axes_std_hist = np.array([axes_std_hist])

        for i, (Xs, thetas, speeds, prob_means, prob_stds, costs, itr_samples) in \
                enumerate(zip(itrs_Xs, itrs_thetas, itrs_speeds, itrs_prob_means, itrs_prob_stds, itrs_costs, samples)):

            ax_mean = axes_mean.ravel()[i]
            ax_std = axes_std.ravel()[i]
            ax_cost = axes_cost.ravel()[i]
            ax_std_hist = axes_std_hist.ravel()[i]

            for ax in (ax_mean, ax_std, ax_cost):
                ax.set_title('Itr {0}'.format(i))

                plt.show(block=False)
                plt.pause(0.01)

                wd = self._load_world_file(i, 0, 0)
                for pose, radius, height in zip(wd['cyl_poses'], wd['cyl_radii'], wd['cyl_heights']):
                    patch = ax.add_artist(plt.Circle(pose[:2, 3], radius, color='k'))
                    ax.draw_artist(patch)

                ### plot samples
                # for s in itr_samples:
                #     colls = s.get_O(sub_obs='collision')
                #     positions = s.get_X(sub_state='position')[:, :2]
                #     coll_positions = positions[colls.ravel() == 1]
                #     no_coll_positions = positions[colls.ravel() == 0]
                #
                #     for ax in (ax_mean, ax_std, ax_cost):
                #         if len(coll_positions) > 0:
                #             ax.plot(coll_positions[:, 0], coll_positions[:, 1], 'cx',
                #                     alpha=0.2, markeredgecolor='c', markeredgewidth=1.5)
                #         if len(no_coll_positions) > 0:
                #             ax.plot(no_coll_positions[:, 0], no_coll_positions[:, 1], 'o',
                #                     alpha=0.2, color='DarkSlateBlue', markeredgewidth=0, markersize=1.5)

            X, Y = Xs[:, 0], Xs[:, 1]
            U = np.array(speeds) * np.cos(thetas)
            V = np.array(speeds) * np.sin(thetas)

            ### plot mean
            ax_mean.quiver(X, Y, U, V, 1 - prob_means, angles='xy', cmap=cm.RdYlGn)
            ax_mean.axis('equal')

            ### plot std
            std_plot = prob_stds.max() - prob_stds
            # std_plot = np.clip(std_plot, 0.0, np.median(std_plot))  # TODO
            # std_plot /= std_plot.max()
            ax_std.quiver(X, Y, U, V, std_plot, angles='xy', cmap=cm.RdYlGn)
            ax_std.axis('equal')

            ### plot cost
            ax_cost.quiver(X, Y, U, V, 1 - costs, angles='xy', cmap=cm.RdYlGn)
            ax_cost.axis('equal')

            ### plot std hist
            if np.all(np.isfinite(prob_stds)):
                ax_std_hist.hist(prob_stds.ravel(),
                                 weights=np.ones(len(prob_stds.ravel())) * 1. / len(prob_stds.ravel()),
                                 bins=10)
                ax_std_hist.set_title('Itr {0}'.format(i))

        f_mean.suptitle('Bootstrap MEAN plots')
        f_std.suptitle('Bootstrap STD plots')
        f_cost.suptitle('Bootstrap COST plots')
        f_std_hist.suptitle('Bootstrap STD hist plots')

        plt.show(block=False)
        plt.pause(0.1)
        f_mean.savefig(self._plot_pred_mean_file, dpi=400)
        f_std.savefig(self._plot_pred_std_file, dpi=400)
        f_cost.savefig(self._plot_pred_cost_file, dpi=400)
        f_std_hist.savefig(self._plot_pred_std_hist_file, dpi=400)

    def _plot_prediction_groundtruth(self):
        itrs_gt_crashes, itrs_pred_crashes, positions, speeds_angles = self._evaluate_probcoll_model_groundtruth()

        sqrt_num_axes = int(np.ceil(np.sqrt(len(itrs_gt_crashes))))
        f, axes = plt.subplots(sqrt_num_axes, sqrt_num_axes, figsize=(25, 25))
        if not hasattr(axes, '__iter__'):
            axes = np.array([axes])

        for i, (gt_crashes, pred_crashes) in enumerate(zip(itrs_gt_crashes, itrs_pred_crashes)):
            ax = axes.ravel()[i]

            ax.set_title('Itr {0}'.format(i))

            plt.show(block=False)
            plt.pause(0.01)

            wd = self._load_world_file(i, 0, 0)
            for pose, radius, height in zip(wd['cyl_poses'], wd['cyl_radii'], wd['cyl_heights']):
                patch = ax.add_artist(plt.Circle(pose[:2, 3], radius, color='k'))
                ax.draw_artist(patch)


            plot_xs = []
            plot_ys = []
            plot_colors = []
            plot_types = []

            ### for each starting position
            for j, (gt_j, pred_j, pos_j) in enumerate(zip(gt_crashes, pred_crashes, positions)):
                ### for each primitive
                for k, (gt_jk, pred_jk, speed_angle_jk) in enumerate(zip(gt_j, pred_j, speeds_angles)):
                    speed, angle = speed_angle_jk
                    x, y = pos_j
                    plot_xs.append(x + 0.25 * speed * np.cos(angle))
                    plot_ys.append(y + 0.25 * speed * np.sin(angle))

                    if gt_jk and pred_jk:
                        plot_types.append(0)
                        plot_colors.append((1, 0, 1))
                    elif not gt_jk and not pred_jk:
                        plot_types.append(1)
                        plot_colors.append((0, 1, 0))
                    elif gt_jk and not pred_jk:
                        plot_types.append(2)
                        plot_colors.append((1, 0, 0))
                    elif not gt_jk and pred_jk:
                        plot_types.append(3)
                        plot_colors.append((0, 0, 1))

            plot_xs, plot_ys, plot_colors, plot_types = np.array(plot_xs), np.array(plot_ys), np.array(plot_colors), np.array(plot_types)

            for j, label in enumerate(('True positive', 'True negative', 'False negative', 'False positive')):
                b = (plot_types == j)
                ax.scatter(plot_xs[b], plot_ys[b], s=2, c=plot_colors[b], edgecolors=plot_colors[b], label=label)

            ax.legend()
            ax.axis('equal')
            ax.set_title('Itr {0}'.format(i))

        f.suptitle('Ground truth')

        plt.show(block=False)
        plt.pause(0.1)
        f.savefig(self._plot_pred_groundtruth_file, dpi=400)

    def _plot_samples_prediction(self):
        samples = self._load_samples()
        mpcs = self._load_mpcs()

        for i, (itr_samples, itr_mpcs) in enumerate(zip(samples, mpcs)):
            sqrt_num_axes = int(np.ceil(np.sqrt(len(itr_samples))))
            f_mean, axes_mean = plt.subplots(sqrt_num_axes, sqrt_num_axes, figsize=(15, 15))
            f_std, axes_std = plt.subplots(sqrt_num_axes, sqrt_num_axes, figsize=(15, 15))

            plt.show(block=False)
            plt.pause(0.1)

            wd = self._load_worlds()[i][0]

            means, stds = self._evaluate_probcoll_model_on_samples(i, itr_samples, itr_mpcs)

            ### plot samples
            for axes_row, colors_row in zip((axes_mean.ravel(), axes_std.ravel()), (means, stds)):
                for j, (s, ax, colors) in enumerate(zip(itr_samples, axes_row, colors_row)):
                    positions = s.get_X(sub_state='position')[:, :2]

                    if colors is None:
                        colors = 'k'
                    else:
                        # dark is collision or uncertain
                        colors = colors.ravel()
                        colors = np.clip(colors, -1, 1)
                        colors = (colors - colors.min()) / 2.
                        colors = 1 - colors
                        colors = np.array([cmaps.magma(c) for c in colors])

                    ax.scatter(positions[:, 0], positions[:, 1], s=20, color=colors)
                    ax.plot(positions[:, 0], positions[:, 1], 'kx', alpha=0.5, markersize=1.)
                    ax.plot(positions[:, 0], positions[:, 1], 'k-', alpha=0.5, linewidth=0.5)

                    for pose, radius, height in zip(wd['cyl_poses'], wd['cyl_radii'], wd['cyl_heights']):
                        patch = ax.add_artist(plt.Circle(pose[:2, 3], radius, edgecolor='g', facecolor='none'))
                        ax.draw_artist(patch)

                    ax.set_title('Itr {0} s{1}'.format(i, j), fontsize=10)

                    maxabsxy = max(abs(positions).max(), 1.)
                    ax.set_xlim((-maxabsxy+1, maxabsxy+1))
                    ax.set_ylim((-maxabsxy, maxabsxy))
                    # ax.axis('equal')

            f_mean.suptitle('Itr {0}: Bootstrap MEAN SAMPLES PREDICTION plots'.format(i))
            f_std.suptitle('Itr {0}: Bootstrap STD SAMPLES PREDICTION plots'.format(i))

            plt.show(block=False)
            plt.pause(0.1)
            f_mean.savefig(self._plot_mean_samples_prediction_file(i), dpi=400)
            f_std.savefig(self._plot_std_samples_prediction_file(i), dpi=400)
            plt.close(f_mean)
            plt.close(f_std)

    def _plot_samples(self):
        samples = self._load_samples()
        mpcs = self._load_mpcs()

        sqrt_num_axes = int(np.ceil(np.sqrt(len(samples))))
        f_samples, axes_samples = plt.subplots(sqrt_num_axes, sqrt_num_axes, figsize=(15, 15))
        if not hasattr(axes_samples, '__iter__'):
            axes_samples = np.array([axes_samples])

        for i, (itr_samples, itr_mpcs) in enumerate(zip(samples, mpcs)):
            ax = axes_samples.ravel()[i]

            plt.show(block=False)
            plt.pause(0.01)

            wd = self._load_worlds()[i][0]
            for pose, radius, height in zip(wd['cyl_poses'], wd['cyl_radii'], wd['cyl_heights']):
                patch = ax.add_artist(plt.Circle(pose[:2, 3], radius, edgecolor='g', facecolor='none'))
                ax.draw_artist(patch)

            ### plot samples
            num_colls = 0
            x_dists = []
            past_cyl = []
            for s in itr_samples:
                colls = s.get_O(sub_obs='collision')
                positions = s.get_X(sub_state='position')[:, :2]

                coll_positions = positions[colls.ravel() == 1]
                no_coll_positions = positions[colls.ravel() == 0]

                num_colls += 1 if colls.sum() > 0 else 0
                x_dists.append(s.get_X(t=-1, sub_state='position')[0] - s.get_X(t=0, sub_state='position')[0])
                past_cyl.append(s.get_X(t=-1, sub_state='position')[0] >= 1.0)

                if len(coll_positions) > 0:
                    ax.plot(coll_positions[:, 0], coll_positions[:, 1], 'x',
                            alpha=0.8, color='c', markeredgecolor='c', markeredgewidth=2.0)
                if len(no_coll_positions) > 0:
                    ax.plot(no_coll_positions[:, 0], no_coll_positions[:, 1], 'o',
                            alpha=0.8, color='DarkSlateBlue', markeredgewidth=0, markersize=3.0)

            ax.set_title('Itr {0} | {1}/{2} colls | # past cyl {3}'.format(i, num_colls, len(itr_samples),
                                                                           np.sum(past_cyl)),
                         fontsize=10)

            # ax.set_xlim((-1, 3))
            # ax.set_ylim((-2, 2))

        for ax in axes_samples.ravel():
            ax.set_xlim((-1, 8))
            ax.set_ylim((-4, 4))
            ax.axis('equal')

        f_samples.suptitle('Bootstrap SAMPLES plots')

        plt.show(block=False)
        plt.pause(0.1)
        f_samples.savefig(self._plot_samples_file, dpi=400)

    ###########
    ### Run ###
    ###########

    def run(self, plot_single, plot_traj, plot_samples, plot_groundtruth):
        self._plot_statistics()
        if plot_samples:
            print('Plotting samples')
            self._plot_samples()
            # self._plot_samples_prediction()
        if plot_single:
            print('Plotting single')
            self._plot_probcoll_model()
        if plot_traj:
            print('Plotting traj')
            self._plot_prediction_trajectory_model()
        if plot_groundtruth:
            print('Plotting ground truth')
            self._plot_prediction_groundtruth()

