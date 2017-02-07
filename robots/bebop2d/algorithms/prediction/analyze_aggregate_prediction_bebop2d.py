import os

import matplotlib.pyplot as plt
import matplotlib.cm as cm

from general.algorithms.prediction.analyze_aggregate_prediction import load_pickles, flatten_list, plot_cum_crash_speeds, plot_boxplot_xvels, plot_safety_vs_performance, plot_crash_timeline, plot_crash_speed_video

EXP_FOLDER = '/media/gkahn/ExtraDrive1/data/bebop2d/'
SAVE_FOLDER = '/home/gkahn/Dropbox/Apps/ShareLaTeX/2017_RSS_ProbColl/figures/exps/'
# SAVE_FOLDER = '/home/gkahn/tmp'

### lambda_std = 0
std0 = []
for i, exp_num in enumerate(range(20, 25)):
    std0.append({
        'folder': os.path.join(EXP_FOLDER, 'exp{0}'.format(exp_num)),
        'label': '$\lambda_{\\textsc{std}} = 0$ (without uncertainty)',
        'color': cm.Reds(0.8),
        'marker': '|',
        'ms': 15,
        'mew': 2,
        'legend_loc': 'middle left',
        'bbox_to_anchor': (-0.1, 0.5),
        # 'color': cm.Reds(0.3),
        # 'marker': 'd',
    })

### lambda_std = 2
std2 = []
for i, exp_num in enumerate(range(25, 30)):
    std2.append({
        'folder': os.path.join(EXP_FOLDER, 'exp{0}'.format(exp_num)),
        'label': '$\lambda_{\\textsc{std}} = 2$ (ours)',
        'color': cm.Blues(0.9),
        'marker': 'o',
        'ms': 8,
        'mew': 2,
        'legend_loc': 'middle right',
        'bbox_to_anchor': (1.22, 0.5),
        # 'color': cm.Reds(0.7),
        # 'marker': 'd',
    })

if __name__ == '__main__':

    exp_groups_and_names = [([std0, std2], 'all')]

    print('Loading pickles')
    load_pickles(flatten_list(zip(*exp_groups_and_names)[0]))
    print('Pickles loaded')

    for exp_groups, name in exp_groups_and_names:
        print('Plotting {0}'.format(name))
        f, axes = plt.subplots(1, 2, figsize=(12, 3))
        plot_cum_crash_speeds(exp_groups, max_speed=2.0, ax=axes[0], is_legend=False, save_path=None, std=1.)
        plot_boxplot_xvels(exp_groups, des_xvel=1.6, max_speed=2., ax=axes[1], is_legend=True, start_itr=0, ylim=None,
                           save_path=os.path.join(SAVE_FOLDER, 'bebop_{0}.png'.format(name)))

    # plot_safety_vs_performance([[std0, std2]],
    #                            des_xvel=0.8,
    #                            nrows=1, start_itr=2, xtext_offset=-0.15, ytext_offset=-0.03, leg_offset=0.45,
    #                            save_path=os.path.join(SAVE_FOLDER,
    #                                                   'bebop_safety_vs_performance_{0}.png'.format('all')))

    # plot_crash_timeline([[std2[3]], [std0[3]]], max_speed=2.0, num_bins=8, max_counts=20, end_itr=None,
    #                     save_folder='/home/gkahn/kdenlive/probcoll_realworld/rl/crash_slides')

    # plot_crash_speed_video([std0[3], std2[3]], framerate=2, max_speed=2.,
    #                        save_folder='/home/gkahn/kdenlive/probcoll_realworld/rl/speed_slides')

    # plt.show(block=False)
    # import IPython; IPython.embed()
