import os, sys, pickle
import rospy, rosbag
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib

from general.algorithm.analyze import Analyze

from config import params

class AnalyzeRCcar(Analyze):

    def __init__(self, on_replay=False):
        rospy.init_node('analyze_rccar', anonymous=True)
        Analyze.__init__(self, on_replay=on_replay, parent_exp_dir=None)

    #######################
    ### Data processing ###
    #######################


    #############
    ### Files ###
    #############

    def _bag_file(self, itr, cond, rep):
        return os.path.join(self._itr_dir(itr), 'bagfile_itr{0}_cond{1}_rep{2}.bag'.format(itr, cond, rep))

    def _plot_trajectories_file(self, itr):
        return os.path.join(self._save_dir, self._image_folder, 'trajectories_{0}.png'.format(itr))

    ################
    ### Plotting ###
    ################

    def _plot_statistics(self):
        ### get samples
        samples_itrs = self._load_samples()
        num_itrs = len(samples_itrs)
        samples_per_itr = len(samples_itrs[0])
        itrs = np.arange(num_itrs)
        ### pkl dict to save to
        pkl_dict = {}

        ### 4 plots
        f, axes = plt.subplots(4, 1, sharex=True, figsize=(15,15))

        ### durations
        durations = []
        times = []
        time_to_crash = 0
        next_time = 0
        lins = np.linspace(0, num_itrs - 1, num_itrs * samples_per_itr)

        for i, samples in enumerate(samples_itrs):
            for j, s in enumerate(samples):
                time_to_crash += s._T * params['dt']
                if (int(s.get_O(t=-1, sub_obs='collision')) == 1) or \
                        ((i == len(samples) - 1) and \
                        (j == len(samples_itrs) - 1)):
                    durations.append(time_to_crash)
                    times.append(next_time)
                    time_to_crash = 0
                    next_time = lins[i * samples_per_itr + j]

        dur_means = []
        dur_stds = []
        avg_itrs = 10
        # TODO hard coded
        for i in xrange(len(durations)):
            lower_index = max(0, i - avg_itrs)
            dur_means.append(np.mean(durations[lower_index:i+1]))
            dur_stds.append(np.std(durations[lower_index:i+1]))
        if len(itrs) > 1:
            axes[0].errorbar(
                times,
                dur_means,
                yerr=dur_stds)
        else:
            axes[0].errorbar([-1]+list(itrs), dur_means*2, yerr=dur_stds*2)
        axes[0].set_title('Time until crash')
        axes[0].set_ylabel('Duration (s)')
        axes[0].set_xlabel('Steps')
        pkl_dict['durations'] = durations

        ### crash speeds
        crash_speed_itrs = []
        crash_speed_means = []
        crash_speed_stds = []
        for samples in samples_itrs:
            crashes = [int(s.get_O(t=-1, sub_obs='collision')) == 1 for s in samples]
            final_speeds = [s.get_U(t=-1, sub_control='cmd_vel')[0] for s in samples]
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
        # TODO size issue
        axes[1].set_title('Speed at crash')
        axes[1].set_ylabel('Speed (m/s)')
        pkl_dict['crash_speeds'] = crash_speed_itrs

        ### cumulative energy
        crash_energies = []
        for samples in samples_itrs:
            crashes = [int(s.get_O(t=-1, sub_obs='collision')) == 1 for s in samples]
            final_speeds = [s.get_U(t=-1, sub_control='cmd_vel')[0] for s in samples]
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
                # TODO think of better way to consolidate
                x_vels_itr += s.get_U(sub_control='cmd_vel').ravel().tolist()
            x_vels_itrs.append(x_vels_itr)
        x_vel_min = np.hstack(x_vels_itrs).min()
        x_vel_max = np.hstack(x_vels_itrs).max()
#        num_bins = len(params["planning"]["primitives"]["speeds"]) + 1
#        bins = np.linspace(x_vel_min, x_vel_max, num_bins)
        bins = np.array(params["planning"]["primitives"]["speeds"] + [np.inf])
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


#        plt.show(block=False)
#        plt.pause(0.1)
        if not os.path.exists(os.path.dirname(self._plot_stats_file)):
            os.makedirs(os.path.dirname(self._plot_stats_file))
        f.savefig(self._plot_stats_file)

        # TODO add sim specifics
        if params["sim"]:
            positions_itrs = self._plot_position(samples_itrs)
            pkl_dict['positions'] = positions_itrs

        with open(self._plot_stats_file_pkl, 'w') as f:
            pickle.dump(pkl_dict, f)

    def _plot_position(self, samples_itrs):
        positions_itrs = [] 
        for itr, samples in enumerate(samples_itrs):
            positions_x, positions_y, collision = [], [], []
            plt.figure()
            plt.title('Trajectories for itr {0}'.format(itr))
            plt.xlabel('X position')
            plt.ylabel('Y position')
            # TODO not make this hard coded
            plt.ylim([-50, 75])
            plt.xlim([-110, 25])
            for s in samples:
                pos = s.get_X(sub_state='position')
                is_coll = s.get_O(t=-1, sub_obs='collision')
                pos_x, pos_y = pos[:, 0], pos[:, 1]
                if is_coll:
                    plt.plot(pos_x, pos_y, color='r')
                    plt.scatter(pos_x[-1], pos_y[-1], marker="x", color='r')
                else:
                    plt.plot(pos_x, pos_y, color='b')
                positions_x.append(pos_x)
                positions_y.append(pos_y)
            positions_itrs.append([positions_x, positions_y, collision])
            plt.savefig(self._plot_trajectories_file(itr)) 
            plt.close()
        return positions_itrs
    
    ###########
    ### Run ###
    ###########

    def run(self, plot_single, plot_traj, plot_samples, plot_groundtruth):
        self._plot_statistics()

        rospy.signal_shutdown('')
