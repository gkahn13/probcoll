import pickle
import numpy as np
import argparse
import os
import copy
import yaml
import matplotlib.pyplot as plt
from general.state_info.sample import Sample
from robots.rccar.algorithm.analyze_rccar import AnalyzeRCcar

class AnalyzeAggPerRCcar(AnalyzeRCcar):

    def __init__(self):
# Round 1
#        self.p_list = [
#                ['model', 'pct_coll', [0.25, 0.5, 0.75]],
#                ['model', 'T', [4, 8, 12, 16]],
#                ['model', 'num_O', [1, 4]],
#                ['model', 'coll_weight_pct', [0.25, 0.5, 0.75]]
#            ]
# Round 2
#        self.p_list = [
#                ['model', 'action_graph', 'cell_type', ['mulint_rnn', 'mulint_lstm', 'lstm']],
#                ['model', 'action_graph', 'dropout', [0.5, 0.8, 1.0]],
#                ['model', 'observation_graph', 'output_dim', [64, 128, 256]],
#                ['model', 'prob_coll_strictly_increasing', [True, False]]
#            ]
# Round 3
#        self.p_list = [
#                ['model', 'grad_clip', [5.0, 10.0]],
#                ['model', 'learning_rate', [0.001, 0.0005]],
#                ['model', 'reg', [1.0e-05, 1.0e-06]]
#            ]
# Round 4
#        self.p_list = [
#                ['model', 'action_graph', 'num_cells', [1, 2]],
#                ['model', 'image_graph', 'output_activation', ['tanh', 'relu', 'spatial_softmax']],
#                ['model', 'image_graph', 'use_batch_norm', [True, False]],
#                ['model', 'observation_graph', 'output_activation', ['tanh', 'relu']]
#            ]
# Round 5
#        self.p_list = [
#               ['model', 'grad_clip', [5., 10.]],
#               ['model', 'image_graph', 'use_batch_norm', [True, False]],
#               ['model', 'observation_graph', 'output_activation', ['tanh', 'None']]
#            ]
# Round 6
        self.p_list = [
                ['model', 'T', [4, 8, 12, 16]],
                ['model', 'grad_clip', [5., 10.]],
                ['model', 'pct_coll', [0.25, 0.5, 0.75]],
                ['model', 'coll_weight_pct', [0.25, 0.5, 0.75]],
                ['model', 'action_graph', 'dropout', [0.5, 0.75, 1.0]]
            ]
#        AnalyzeRCcar.__init__(self)

    #######################
    ### Data processing ###
    #######################

    def _itr_dir_from_dir(self, dir_path, itr):
        assert(type(itr) is int)
        dir = os.path.join(dir_path, 'itr{0}'.format(itr))
        return dir

    def _itr_load_testing_samples_from_dir(self, dir_path, itr):
        fname = os.path.join(self._itr_dir_from_dir(dir_path, itr), 'testing_samples_itr_{0}.npz'.format(itr))
        if os.path.exists(fname):
            return Sample.load(fname)
        elif not os.path.exists(self._itr_dir_from_dir(dir_path, itr)):
            raise Exception()
        else:
            return None

    def _load_last_testing_samples_from_dir(self, dir_path):
        test_samples = []
        itr = 2
#        itr = params['world']['testing']['itr_freq']
        while True:
            try:
                samples = self._itr_load_testing_samples_from_dir(dir_path, itr)
                if samples is not None:
                    test_samples = samples
                itr += 1
            except:
                break
        return test_samples
    
    def _get_elem(self, d, ps):
        next_d = d
        for p in ps:
            next_d = next_d[p]
        return next_d

    def _get_percentages(self, params_list, data):
        cnt = [[] for _ in xrange(len(params_list))]
        for (d, samples) in data:
            success = self._success_percentage(samples)
            for i, ps in enumerate(params_list):
                if self._get_elem(d, ps[:-1]) == ps[-1]:
                    cnt[i].append(success)
        return cnt

    def _get_data_params(self, dir_name):
        data = []
        for sub_dir in os.listdir(dir_name):
            sub_dir_path = os.path.join(dir_name, sub_dir)
            if os.path.isdir(sub_dir_path):
                yaml_path = self._get_yaml(sub_dir_path)
                with open(yaml_path) as yaml_f:
                    p = yaml.load(yaml_f)
                samples = self._load_last_testing_samples_from_dir(sub_dir_path)
                if len(samples) > 0:
                    data.append((p, samples))
        return data

    def _get_exlist_from_list(self, p_list):
        params_list = []
        for ps in p_list:
            for p in ps[-1]:
                params_list.append(copy.copy(ps[:-1]) + [p])
        return params_list

    ################
    ### Plotting ###
    ################

    def percent_plot(self, data_dir, save_path):
        data = self._get_data_params(data_dir)
#        yaml_path = self._get_yaml(data_dir)
#        ps = yaml.load(yaml_path)
        params_list = self._get_exlist_from_list(self.p_list)
        percents = self._get_percentages(params_list, data)
        index = 0
        f, axs = plt.subplots(len(self.p_list))
        f.set_size_inches(len(self.p_list) * 2, len(self.p_list) * 2)
        if len(self.p_list) == 1:
            axs = [axs, None]
        for i, p in enumerate(self.p_list):
            next_index = index + len(p[-1])
            axs[i].set_aspect(1.0)
            axs[i].set_title('{0} {1}'.format(p[-3], p[-2]))
            axs[i].boxplot(percents[index:next_index], whis='range', labels=p[-1], patch_artist=True)
            index = next_index
        plt.tight_layout()
        f.suptitle('Percent Success comparisons for params') 
        f.subplots_adjust(top=0.85)
        if not os.path.exists(os.path.dirname(save_path)):
            os.makedirs(os.path.dirname(save_path))
        f.savefig(save_path)

