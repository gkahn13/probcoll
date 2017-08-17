import os, sys, pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib

from general.analysis.analyze import Analyze
from robots.sim_rccar.algorithm.probcoll_sim_rccar import ProbcollSimRCcar
from config import params

class AnalyzeSimRCcar(Analyze):

    def __init__(self, on_replay=False, save_dir=None, value_heat_map=False):
        self._value_heat_map = value_heat_map
        Analyze.__init__(self, on_replay=on_replay, save_dir=save_dir)

    #######################
    ### Data processing ###
    #######################

    def _success_percentage(self, samples):
        tot = len(samples)
        num_suc = 0.0
        for sample in samples:
            num_suc += (1. - float(sample.get_O(t=-1, sub_obs='collision')))
        return num_suc / tot

    #############
    ### Files ###
    #############

    def _plot_trajectories_file(self, itr, testing=False):
        if testing:
            prefix = 'testing'
        else:
            prefix = ''
        return os.path.join(self._image_folder, '{0}_trajectories_{1}.png'.format(prefix, itr))
    
    ################
    ### Plotting ###
    ################

    def _plot_statistics(self, testing=False, times=None):
        ### get samples
        if testing:
            samples_itrs, sample_times = self._load_testing_samples()
        else:
            samples_itrs, sample_times = self._load_samples()
        
        if times is None:
            times = sample_times

        total_times = self._plot_position(samples_itrs, times, testing=testing)
        if testing:
            self._plot_testing_speed_crash_statistics(samples_itrs, sample_times, times)
        else:
            self._plot_speed_crash_statistics(samples_itrs, times)
        return total_times

    def _plot_position(self, samples_itrs, times, testing=False):
        blue_line = matplotlib.lines.Line2D([], [], color='b', label='collision')
        red_line = matplotlib.lines.Line2D([], [], color='r', label='no collision')
        total_times = []
        total_time = 0
        for itr, samples in samples_itrs:
            if not isinstance(times[itr], list):
                total_time = times[itr]
            else:
                cur_time = total_time
                total_time += sum(times[itr])
            total_times.append(total_time)
            plt.figure()
            if testing:
                plt.title('Trajectories for testing itr {0}\n{1} runtime : {2:.2f} min'.format(
                    itr,
                    params['exp_name'],
                    total_time * params['probcoll']['dt'] / 60.))
            else:
                plt.title('Trajectories for itr {0}\n{1} runtime : {2:.2f} min'.format(
                    itr,
                    params['exp_name'],
                    cur_time * params['probcoll']['dt'] / 60.))
            plt.xlabel('X position')
            plt.ylabel('Y position')
            if params['sim']['sim_env'] == 'square' or params['sim']['sim_env'] == 'square_banked':
                plt.ylim([-22.5, 22.5])
                plt.xlim([-22.5, 22.5])
                plt.legend(handles=[blue_line, red_line], loc='center')
            elif params['sim']['sim_env'] == 'cylinder' or params['sim']['sim_env'] == 'small_cylinder':
                plt.ylim([-7.5, 7.5])
                plt.xlim([-4.5, 4.5])
                plt.legend(handles=[blue_line, red_line], loc='center')
            for s in samples:
                pos = s.get_X(sub_state='position')
                is_coll = s.get_O(t=-1, sub_obs='collision')
                pos_x, pos_y = pos[:, 0], pos[:, 1]
                if is_coll:
                    plt.plot(pos_x, pos_y, color='r')
                    plt.scatter(pos_x[-1], pos_y[-1], marker="x", color='r')
                else:
                    plt.plot(pos_x, pos_y, color='b')
            plt.savefig(self._plot_trajectories_file(itr, testing)) 
            plt.close()
        return total_times        

    def _plot_speed_crash_statistics(self, samples_itrs, times):
        # Make 3 plots
        f, axes = plt.subplots(3, 1, sharex=True, figsize=(15,15))
        avg_speeds = []
        avg_crashes = []
        cumulative_crash_energies = []
        cumulative_crash_energy = 0
        crashes = []
        speeds = []
        for (itr, samples), time_list in zip(samples_itrs, times):
            for s, time in zip(samples, time_list):
                for t in range(time):
                    speeds.append(s.get_U(t=t, sub_control='cmd_vel')[0])
                    crashes.append(s.get_O(t=t, sub_obs='collision')[0])
                    avg_speeds.append(float(sum(speeds[-1000:])) / len(speeds[-1000:]))
                    avg_crashes.append(float(sum(crashes[-1000:])) / (len(crashes[-1000:]) * params['probcoll']['dt']))
                    cumulative_crash_energy += crashes[-1] * (speeds[-1] ** 2)
                    cumulative_crash_energies.append(cumulative_crash_energy)

        avg_speeds = np.array(avg_speeds)
        avg_crashes = np.array(avg_crashes)
        cumulative_crash_energies = np.array(cumulative_crash_energies)
        
        time_range = np.arange(len(avg_speeds)) * params['probcoll']['dt'] / 60.
        
        axes[0].set_title('Moving Average speed')
        axes[0].set_ylabel('Speed (m / s)')
        axes[1].set_title('Moving Average crashes per time')
        axes[1].set_ylabel('Crashes / s')
        axes[2].set_title('Cumulative energy from crashing over time')
        axes[2].set_ylabel('Energy')
        axes[2].set_xlabel('Time (min)')
        axes[0].plot(time_range, avg_speeds)
        axes[1].plot(time_range, avg_crashes)
        axes[2].plot(time_range, cumulative_crash_energies)
        stats_file = self._plot_stats_file
        
        if not os.path.exists(os.path.dirname(stats_file)):
            os.makedirs(os.path.dirname(stats_file))
        f.savefig(stats_file)

    def _plot_testing_speed_crash_statistics(self, samples_itrs, times, total_times):
        f, axes = plt.subplots(4, 1, sharex=True, figsize=(15,15))
        avg_speeds = []
        avg_crashes = []
        avg_dists = []
        cumulative_crash_energies = []
        cumulative_crash_energy = 0
        time_range = []
        for (itr, samples), time_list in zip(samples_itrs, times):
            tot_speed = 0
            tot_crashes = 0
            tot_dists = 0
            for s in samples:
                speeds = s.get_U(sub_control='cmd_vel')
                dists = speeds * params['probcoll']['dt'] 
                crashes = s.get_O(sub_obs='collision')
                tot_speed += np.sum(speeds)
                tot_crashes += np.sum(crashes)
                tot_dists += np.sum(dists)
                cumulative_crash_energy += np.sum(crashes * (speeds ** 2))
            time_range.append(total_times[itr])
            tot_time = np.sum(time_list)
            avg_speeds.append(tot_speed / tot_time)
            avg_crashes.append(tot_crashes / len(samples)) 
            avg_dists.append(tot_dists / len(samples))
            cumulative_crash_energies.append(cumulative_crash_energy)
        avg_speeds = np.array(avg_speeds)
        avg_crashes = np.array(avg_crashes)
        avg_dists = np.array(avg_dists)
        cumulative_crash_energies = np.array(cumulative_crash_energies)
        time_range = np.array(time_range) * params['probcoll']['dt'] / 60. 
        
        axes[0].set_title('Average speed')
        axes[0].set_ylabel('Speed (m / s)')
        axes[1].set_title('Percent Trajectories ending in Collision')
        axes[1].set_ylabel('Crash percent')
        axes[2].set_title('Average Distance')
        axes[2].set_ylabel('Distance m')
        axes[3].set_title('Cumulative energy from crashing over time')
        axes[3].set_ylabel('Energy')
        axes[2].set_xlabel('Time (min)')
        axes[0].scatter(time_range, avg_speeds)
        axes[1].scatter(time_range, avg_crashes)
        axes[2].scatter(time_range, avg_dists)
        axes[3].scatter(time_range, cumulative_crash_energies)
        stats_file = self._plot_testing_stats_file
        
        if not os.path.exists(os.path.dirname(stats_file)):
            os.makedirs(os.path.dirname(stats_file))
        f.savefig(stats_file)

    def _generate_value_heat_map(self):
        probcoll = ProbcollSimRCcar()
        num_itrs = probcoll.probcoll_model.get_train_itr()
        for itr in range(num_itrs + 1):
            stats_file = self._plot_get_stats_file(prefix='value_heatmap_{0}'.format(itr)) 
            if not os.path.exists(stats_file):
                probcoll.probcoll_model.load_itr(itr)
                policy = probcoll.policy
                agent = probcoll.agent
                # TODO different orientations
                ori = [0., 0., 3.14]
                plt.figure()
                if params['sim']['sim_env'] == 'cylinder' or params['sim']['sim_env'] == 'cylinder_small':
                    if params['sim']['sim_env'] == 'cylinder':
                        radius = 3.5
                    elif params['sim']['sim_env'] == 'cylinder_small':
                        radius = 2.0
                    plt.xlim([0, 18])
                    plt.ylim([0, 30])
                    values = np.zeros((30, 18))
                    positions = [] 
                    indices = []
                    for i in range(1, 18):
                        for j in range(2, 29):
                            pos = ((i - 9.) / 2., (j - 15.) / 2., 0.25)
                            if pos[0] ** 2 + pos[1] ** 2 >= radius ** 2:
                                positions.append(pos)
                                indices.append((j, i))
                for pos, index in zip(positions, indices):# TODO
                    val = agent.get_value(policy, pos=pos, ori=ori)
                    values[index] = val
                plt.imshow(values, cmap='hot')
                plt.title('Value function')
                plt.xlabel('Y position')
                plt.ylabel('X position')
                if not os.path.exists(os.path.dirname(stats_file)):
                    os.makedirs(os.path.dirname(stats_file))
                plt.savefig(stats_file) 
                plt.close()


    ###########
    ### Run ###
    ###########

    def run(self):
        try:
            times = self._plot_statistics()
        except:
            self._logger.info('No training trajectories were loaded to analyze')
        try:
            self._plot_statistics(times=times, testing=True)
        except:
            self._logger.info('No testing trajectoreis were loaded to analyze')
        if self._value_heat_map:
            try:
                self._generate_value_heat_map()
            except:
                self._logger.info('Unable to generate value heat maps')

    def run_testing(self):
        try:
            self._plot_statistics(testing=True)
        except:
            self._logger.info('No testing trajectoreis were loaded to analyze')
