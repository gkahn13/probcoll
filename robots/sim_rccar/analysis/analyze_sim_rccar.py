import os, sys, pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib

from general.analysis.analyze import Analyze

from config import params

class AnalyzeSimRCcar(Analyze):

    def __init__(self):
        Analyze.__init__(self, parent_exp_dir=None)

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
        return os.path.join(self._save_dir, self._image_folder, '{0}_trajectories_{1}.png'.format(prefix, itr))


    ################
    ### Plotting ###
    ################

    def _plot_statistics(self, testing=False):
        ### get samples
        if testing:
            samples_itrs = self._load_testing_samples()
        else:
            samples_itrs = self._load_samples()
        pkl_dict = {}
        if testing:
            if not os.path.exists(os.path.dirname(self._plot_testing_stats_file)):
                os.makedirs(os.path.dirname(self._plot_testing_stats_file))
        else:
            if not os.path.exists(os.path.dirname(self._plot_stats_file)):
                os.makedirs(os.path.dirname(self._plot_stats_file))
        
        positions_itrs = self._plot_position(samples_itrs, testing=testing)
        pkl_dict['positions'] = positions_itrs

        if testing:
            stats_f = self._plot_testing_stats_file_pkl
        else:
            stats_f = self._plot_stats_file_pkl
        
        with open(stats_f, 'w') as f:
            pickle.dump(pkl_dict, f)

    def _plot_position(self, samples_itrs, testing=False):
        positions_itrs = [] 
        blue_line = matplotlib.lines.Line2D([], [], color='b', label='collision')
        red_line = matplotlib.lines.Line2D([], [], color='r', label='no collision')
        for itr, samples in samples_itrs:
            positions_x, positions_y, collision = [], [], []
            plt.figure()
            if testing:
                plt.title('Trajectories for testing itr {0}\n{1}'.format(
                    itr,
                    params['exp_name']))
            else:
                plt.title('Trajectories for itr {0}\n{1}'.format(itr, params['exp_name']))
            plt.xlabel('X position')
            plt.ylabel('Y position')
            # TODO not make this hard coded
            if params['sim']['sim_env'] == 'square':
                plt.ylim([-22.5, 22.5])
                plt.xlim([-22.5, 22.5])
                plt.legend(handles=[blue_line, red_line], loc='center')
            elif params['sim']['sim_env'] == 'cylinder':
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
                positions_x.append(pos_x)
                positions_y.append(pos_y)
            positions_itrs.append([positions_x, positions_y, collision])
            plt.savefig(self._plot_trajectories_file(itr, testing)) 
            plt.close()
        return positions_itrs
    
    ###########
    ### Run ###
    ###########

    def run(self):
        try:
            self._plot_statistics()
        except:
            self._logger.info('No training trajectories were loaded to analyze')
        try:
            self._plot_statistics(testing=True)
        except:
            self._logger.info('No testing trajectoreis were loaded to analyze')
