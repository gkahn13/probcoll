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
        durations_itrs = []
        time_to_crash = 0

        # TODO deal with iteraitons
        for samples in samples_itrs:
            durations_itr = []
            for i, s in enumerate(samples):
                time_to_crash += s._T * params['dt']
                if (int(s.get_O(t=-1, sub_obs='collision')) == 1) or (i == len(samples) - 1):
                    durations_itr.append(time_to_crash)
                    time_to_crash = 0
            durations_itrs.append(durations_itr)
 

#        durations_itrs = [[s._T * params['dt'] for s in samples] for samples in samples_itrs]
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
                x_vels_itr += s.get_U(sub_control='cmd_vel').ravel().tolist()
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


        # TODO add sim specifics
        if params["sim"]:
            pass

        plt.show(block=False)
        plt.pause(5.0)
        if not os.path.exists(os.path.dirname(self._plot_stats_file)):
            os.makedirs(os.path.dirname(self._plot_stats_file))
        f.savefig(self._plot_stats_file)

        with open(self._plot_stats_file_pkl, 'w') as f:
            pickle.dump(pkl_dict, f)

    ###########
    ### Run ###
    ###########

    def run(self, plot_single, plot_traj, plot_samples, plot_groundtruth):
        self._plot_statistics()

        rospy.signal_shutdown('')
