import os, sys, pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib

from general.analysis.analyze import Analyze
from robots.bebop2d.algorithm.probcoll_bebop2d import ProbcollBebop2d

from config import params

class AnalyzeBebop2d(Analyze):

    def __init__(self):
        Analyze.__init__(self, parent_exp_dir=None)
        self.count = 0

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

    def _plot_statistics(self, testing=False, times=None):
        ### get samples
        if testing:
            samples_itrs, sample_times = self._load_testing_samples()
        else:
            samples_itrs, sample_times = self._load_samples()
        
        if times is None:
            times = sample_times

        pkl_dict = {}
        if testing:
            if not os.path.exists(os.path.dirname(self._plot_testing_stats_file)):
                os.makedirs(os.path.dirname(self._plot_testing_stats_file))
        else:
            if not os.path.exists(os.path.dirname(self._plot_stats_file)):
                os.makedirs(os.path.dirname(self._plot_stats_file))
        
        pkl_dict['positions'] = self._plot_position(samples_itrs, times, testing=testing)

        if testing:
            stats_f = self._plot_testing_stats_file_pkl
        else:
            stats_f = self._plot_stats_file_pkl
        
        with open(stats_f, 'w') as f:
            pickle.dump(pkl_dict, f)

        return times

    def _plot_position(self, samples_itrs, times, testing=False):
        positions_itrs = []
        blue_line = matplotlib.lines.Line2D([], [], color='b', label='collision')
        red_line = matplotlib.lines.Line2D([], [], color='r', label='no collision')
        total_time = 0
        for itr, samples in samples_itrs:
            total_time = sum([times[i] for i in xrange(itr + 1)])
            positions_x, positions_y, collision = [], [], []
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
                    total_time * params['probcoll']['dt'] / 60.))
            plt.xlabel('X position')
            plt.ylabel('Y position')
            if params['sim']['sim_env'] == 'square' or params['sim']['sim_env'] == 'square_banked':
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

    def _display_selection(self):
        prediction = ProbcollBebop2d()
        for samples_start_itr in xrange(prediction._max_iter - 1, -1, -1):
            sample_file = prediction._itr_samples_file(samples_start_itr, create=False)
            if os.path.exists(sample_file):
                samples_start_itr += 1
                break
        ### load initial dataset
        init_data_folder = params['probcoll'].get('init_data', None)
        if init_data_folder is not None:
            prediction._logger.info('Adding initial data')
            ext = os.path.splitext(prediction._itr_samples_file(0))[-1]
            fnames = [fname for fname in os.listdir(init_data_folder) if ext in fname]
            for fname in fnames:
                prediction._logger.info('\t{0}'.format(fname))
                prediction.probcoll_model.add_data([os.path.join(init_data_folder, fname) for fname in fnames])
        ### if any data and haven't trained on it already, train on it
        if (samples_start_itr > 0 or init_data_folder is not None) and (samples_start_itr != prediction._max_iter):
            prediction._run_training(samples_start_itr)
        samples_itrs, sample_times = self._load_samples()
        # import IPython; IPython.embed()
        num_O = params['model']['num_O']
        # visualized = params['planning']['visualize']
        if_plot = params['analyze']['plot']
        self._no_coll_ns = []
        self._coll_ns = []

        def stop_algorithm(ns):
            return ns[10] < 0.98

        for itr, samples in samples_itrs:
            for n, s in enumerate(samples):
                if len(s.get_O()) >= num_O:
                    for i in xrange(len(s.get_O()) - num_O + 1):
                        u_t, u_t_no_noise = prediction._mpc_policy.act(
                            s.get_O()[i:i+num_O],
                            0,
                            0,
                            only_noise=False,
                            only_no_noise=False,
                            visualize=True)
                        # print 'u_t: {0}; u_t_no_noise: {1}'.format(u_t, u_t_no_noise)
                        if s.get_O()[i:i + num_O][-1][-1]:
                            self._coll_ns.append(prediction._mpc_policy.ns)
                        else:
                            self._no_coll_ns.append(prediction._mpc_policy.ns)
                        if if_plot:
                            f = plt.figure()
                            for j in xrange(num_O):
                                # import IPython; IPython.embed()
                                # if s.get_O()[i:i+num_O][-1][-1] == 1 and j==num_O - 1:
                                #     import IPython; IPython.embed()
                                img = np.reshape(s.get_O()[i + j][:-1], [params['O']['camera']['width'], params['O']['camera']['height']])
                                a = f.add_subplot(2, np.ceil(num_O/2.0), j + 1)
                                plt.imshow(img, cmap='gray', vmin=0, vmax=255)
                                a.set_title(j)
                            # if s.get_O()[i:i+num_O][-1][-1] == 1:
                            #     import IPython; IPython.embed()
                            f.suptitle(str(u_t) + str(u_t_no_noise) + str(s.get_O()[i:i+num_O][-1][-1]))
                            # plt.savefig(params['exp_dir'] + '/'+
                            #             params['exp_name'] + '/' + 'plots/' + 'itr_'+ str(itr)+'_sample_'+str(n)+'_act_'+str(i) + '.jpg')
                            plt.savefig(params['exp_dir'] + '/'+
                                        params['exp_name'] + '/' + 'plots/' + str(self.count) + '_im' + '.jpg')
                            self.count += 1
                    if params['analyze']['plot_hist_sequence']:
                        total_length = params['analyze']['plot_last']
                        f, axarr = plt.subplots(2, total_length, figsize=(20, 10))
                        Os = s.get_O()
                        for i in xrange(total_length):
                            if num_O <= len(Os) - i:
                                u_t, u_t_no_noise = prediction._mpc_policy.act(
                                    Os[len(Os)-i-num_O: len(Os)-i],
                                    0,
                                    0,
                                    only_noise=False,
                                    only_no_noise=False,
                                    visualize=True)
                                img = np.reshape(Os[len(Os)-i-1][:-1],
                                                 [params['O']['camera']['width'], params['O']['camera']['height']])
                                axarr[0, i].imshow(img, cmap='gray', vmin=0, vmax=255)
                                axarr[0, i].set_title('last_'+str(i))
                                axarr[1, i].plot(np.linspace(0, 1, 20), prediction._mpc_policy.ns,'-o')
                                if stop_algorithm(prediction._mpc_policy.ns):
                                    axarr[1, i].set_title('hist_last_' + str(i), color='r')
                                else:
                                    axarr[1, i].set_title('hist_last_' + str(i), color='g')
                        plt.savefig(params['exp_dir'] + '/'+
                             params['exp_name'] + '/' + 'plots/' + 'itr_' + str(itr) + '_hist_sample_'+ str(n) + '.jpg')
        if params['analyze']['prob_hist']:
            no_coll_ns_array = np.array(self._no_coll_ns)
            coll_ns_array = np.array(self._coll_ns)
            no_coll_ns_mean = np.mean(no_coll_ns_array, axis=0)
            coll_ns_mean = np.mean(coll_ns_array,axis=0)
            f = plt.figure()
            sub = f.add_subplot(2, 1, 1)
            plt.plot(np.linspace(0, 1, 20), no_coll_ns_mean)
            sub.set_title('no coll hist')
            sub = f.add_subplot(2, 1, 2)
            plt.plot(np.linspace(0, 1, 20), coll_ns_mean)
            sub.set_title('coll hist')
            plt.savefig('probcoll_hist.jpg')
            plt.show()
    ###########
    ### Run ###
    ###########

    def run(self):
        # try:
        #     times = self._plot_statistics()
        # except:
        #     self._logger.info('No training trajectories were loaded to analyze')
        # try:
        #     self._plot_statistics(times=times, testing=True)
        # except:
        #     self._logger.info('No testing trajectoreis were loaded to analyze')
        # try:
        self._display_selection()
        # except:
        #     self._logger.info('No samples were loaded to analyze')
