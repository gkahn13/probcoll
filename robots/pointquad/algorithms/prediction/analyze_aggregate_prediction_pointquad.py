import os

import matplotlib.pyplot as plt
import matplotlib.cm as cm
# import matplotlib
# matplotlib.rcParams.update({'font.size': 22})



from general.algorithms.prediction.analyze_aggregate_prediction import load_pickles, flatten_list, plot_cum_crash_speeds, plot_boxplot_xvels, plot_safety_vs_performance, plot_crash_speed_video

EXP_FOLDER_SSD = '/home/gkahn/code/gps_quadrotor/experiments/pointquad'
EXP_FOLDER_HDD = '/media/gkahn/ExtraDrive1/data/pointquad/'
SAVE_FOLDER = '/home/gkahn/Dropbox/Apps/ShareLaTeX/2017_ICRA_ProbColl/figures/exps/'

### coll = 1e2, std = 0
coll1e2_std0 = []
for i, exp_num in enumerate(range(600, 605)):
    coll1e2_std0.append({
        'folder': os.path.join(EXP_FOLDER_HDD, 'exp{0}'.format(exp_num)),
        'label': '$\lambda_{\\textsc{coll}} = 1e2, \lambda_{\\textsc{std}} = 0$' if i == 0 else None,
        'color': cm.Reds(0.3),
        'marker': 'd',
    })

### coll = 1e2, std = 1
coll1e2_std1 = []
for i, exp_num in enumerate(range(605, 610)):
    coll1e2_std1.append({
        'folder': os.path.join(EXP_FOLDER_HDD, 'exp{0}'.format(exp_num)),
        'label': '$\lambda_{\\textsc{coll}} = 1e2, \lambda_{\\textsc{std}} = 1$ (ours)' if i == 0 else None,
        'color': cm.Reds(0.5),
        'marker': 'd',
    })

### coll = 1e2, std = 2
coll1e2_std2 = []
for i, exp_num in enumerate(range(610, 615)):
    coll1e2_std2.append({
        'folder': os.path.join(EXP_FOLDER_HDD, 'exp{0}'.format(exp_num)),
        'label': '$\lambda_{\\textsc{coll}} = 1e2, \lambda_{\\textsc{std}} = 2$ (ours)' if i == 0 else None,
        'color': cm.Reds(0.7),
        'marker': 'd',
    })

### coll = 1e2, std = 3
coll1e2_std3 = []
for i, exp_num in enumerate(range(615, 620)):
    coll1e2_std3.append({
        'folder': os.path.join(EXP_FOLDER_HDD, 'exp{0}'.format(exp_num)),
        'label': '$\lambda_{\\textsc{coll}} = 1e2, \lambda_{\\textsc{std}} = 3$ (ours)' if i == 0 else None,
        'color': cm.Reds(0.9),
        'marker': 'd',
    })


### coll = 1e3, std = 0
coll1e3_std0 = []
for i, exp_num in enumerate(range(620, 625)):
    coll1e3_std0.append({
        'folder': os.path.join(EXP_FOLDER_HDD, 'exp{0}'.format(exp_num)),
        'label': '$\lambda_{\\textsc{coll}} = 1e3, \lambda_{\\textsc{std}} = 0$' if i == 0 else None,
        'color': cm.Greens(0.3),
        'marker': 'o',
    })

### coll = 1e3, std = 1
coll1e3_std1 = []
for i, exp_num in enumerate(range(625, 630)):
    coll1e3_std1.append({
        'folder': os.path.join(EXP_FOLDER_HDD, 'exp{0}'.format(exp_num)),
        'label': '$\lambda_{\\textsc{coll}} = 1e3, \lambda_{\\textsc{std}} = 1$' if i == 0 else None,
        'color': cm.Greens(0.5),
        'marker': 'o',
    })

### coll = 1e3, std = 2
coll1e3_std2 = []
for i, exp_num in enumerate(range(630, 635)):
    coll1e3_std2.append({
        'folder': os.path.join(EXP_FOLDER_HDD, 'exp{0}'.format(exp_num)),
        'label': '$\lambda_{\\textsc{coll}} = 1e3, \lambda_{\\textsc{std}} = 2$' if i == 0 else None,
        'color': cm.Greens(0.7),
        'marker': 'o',
    })

### coll = 1e3, std = 3
coll1e3_std3 = []
for i, exp_num in enumerate(range(635, 640)):
    coll1e3_std3.append({
        'folder': os.path.join(EXP_FOLDER_HDD, 'exp{0}'.format(exp_num)),
        'label': '$\lambda_{\\textsc{coll}} = 1e3, \lambda_{\\textsc{std}} = 3$' if i == 0 else None,
        'color': cm.Greens(0.9),
        'marker': 'o',
    })


### coll = 1e4, std = 0
coll1e4_std0 = []
for i, exp_num in enumerate(range(640, 645)):
    coll1e4_std0.append({
        'folder': os.path.join(EXP_FOLDER_HDD, 'exp{0}'.format(exp_num)),
        'label': '$\lambda_{\\textsc{coll}} = 1e4, \lambda_{\\textsc{std}} = 0$' if i == 0 else None,
        'color': cm.Blues(0.3),
        'marker': '^',
    })

### coll = 1e4, std = 1
coll1e4_std1 = []
for i, exp_num in enumerate(range(645, 650)):
    coll1e4_std1.append({
        'folder': os.path.join(EXP_FOLDER_HDD, 'exp{0}'.format(exp_num)),
        'label': '$\lambda_{\\textsc{coll}} = 1e4, \lambda_{\\textsc{std}} = 1$' if i == 0 else None,
        'color': cm.Blues(0.5),
        'marker': '^',
    })

### coll = 1e4, std = 2
coll1e4_std2 = []
for i, exp_num in enumerate(range(650, 655)):
    coll1e4_std2.append({
        'folder': os.path.join(EXP_FOLDER_HDD, 'exp{0}'.format(exp_num)),
        'label': '$\lambda_{\\textsc{coll}} = 1e4, \lambda_{\\textsc{std}} = 2$' if i == 0 else None,
        'color': cm.Blues(0.7),
        'marker': '^',
    })

### coll = 1e4, std = 3
coll1e4_std3 = []
for i, exp_num in enumerate(range(655, 660)):
    coll1e4_std3.append({
        'folder': os.path.join(EXP_FOLDER_HDD, 'exp{0}'.format(exp_num)),
        'label': '$\lambda_{\\textsc{coll}} = 1e4, \lambda_{\\textsc{std}} = 3$' if i == 0 else None,
        'color': cm.Blues(0.9),
        'marker': '^',
    })



### coll = 1e5, std = 0
coll1e5_std0 = []
for i, exp_num in enumerate(range(660, 665)):
    coll1e5_std0.append({
        'folder': os.path.join(EXP_FOLDER_HDD, 'exp{0}'.format(exp_num)),
        'label': '$\lambda_{\\textsc{coll}} = 1e5, \lambda_{\\textsc{std}} = 0$' if i == 0 else None,
        'color': cm.Purples(0.3),
        'marker': 's',
    })

### coll = 1e5, std = 1
coll1e5_std1 = []
for i, exp_num in enumerate(range(665, 670)):
    coll1e5_std1.append({
        'folder': os.path.join(EXP_FOLDER_HDD, 'exp{0}'.format(exp_num)),
        'label': '$\lambda_{\\textsc{coll}} = 1e5, \lambda_{\\textsc{std}} = 1$' if i == 0 else None,
        'color': cm.Purples(0.5),
        'marker': 's',
    })

### coll = 1e5, std = 2
coll1e5_std2 = []
for i, exp_num in enumerate(range(670, 675)):
    coll1e5_std2.append({
        'folder': os.path.join(EXP_FOLDER_HDD, 'exp{0}'.format(exp_num)),
        'label': '$\lambda_{\\textsc{coll}} = 1e5, \lambda_{\\textsc{std}} = 2$' if i == 0 else None,
        'color': cm.Purples(0.7),
        'marker': 's',
    })

### coll = 1e5, std = 3
coll1e5_std3 = []
for i, exp_num in enumerate(range(675, 680)):
    coll1e5_std3.append({
        'folder': os.path.join(EXP_FOLDER_HDD, 'exp{0}'.format(exp_num)),
        'label': '$\lambda_{\\textsc{coll}} = 1e5, \lambda_{\\textsc{std}} = 3$' if i == 0 else None,
        'color': cm.Purples(0.9),
        'marker': 's',
    })


### coll = 1e2, const = 1e-1
coll1e2_const1neg1 = []
for i, exp_num in enumerate(range(680, 685)):
    coll1e2_const1neg1.append({
        'folder': os.path.join(EXP_FOLDER_HDD, 'exp{0}'.format(exp_num)),
        'label': '$\lambda_{\\textsc{coll}} = 1e2, \lambda_{\\textsc{const}} = 0.1$' if i == 0 else None,
        'color': cm.Greens(0.3),
        'marker': '*',
    })

### coll = 1e2, const = 1e0
coll1e2_const1e0 = []
for i, exp_num in enumerate(range(685, 690)):
    coll1e2_const1e0.append({
        'folder': os.path.join(EXP_FOLDER_HDD, 'exp{0}'.format(exp_num)),
        'label': '$\lambda_{\\textsc{coll}} = 1e2, \lambda_{\\textsc{const}} = 1$' if i == 0 else None,
        'color': cm.Greens(0.5),
        'marker': '*',
    })

### coll = 1e2, const = 1e1
coll1e2_const1e1 = []
for i, exp_num in enumerate(range(690, 695)):
    coll1e2_const1e1.append({
        'folder': os.path.join(EXP_FOLDER_HDD, 'exp{0}'.format(exp_num)),
        'label': '$\lambda_{\\textsc{coll}} = 1e2, \lambda_{\\textsc{const}} = 10$' if i == 0 else None,
        'color': cm.Greens(0.7),
        'marker': '*',
    })

### coll = 1e2, const = 1e2
coll1e2_const1e2 = []
for i, exp_num in enumerate(range(695, 700)):
    coll1e2_const1e2.append({
        'folder': os.path.join(EXP_FOLDER_HDD, 'exp{0}'.format(exp_num)),
        'label': '$\lambda_{\\textsc{coll}} = 1e2, \lambda_{\\textsc{const}} = 100$' if i == 0 else None,
        'color': cm.Greens(0.9),
        'marker': '*',
    })

### old sim experiments for visualiztion
old_std0 = {
    'folder': os.path.join(EXP_FOLDER_SSD, 'exp134'),
    'label': '$\lambda_{\\textsc{std}} = 0$',
    'color': cm.Greys(1.0),
}
old_std1 = {
    'folder': os.path.join(EXP_FOLDER_SSD, 'exp133'),
    'label': '$\lambda_{\\textsc{std}} = 0$',
    'color': cm.Greys(1.0),
}

if __name__ == '__main__':
    exp_groups_all = [coll1e2_std0, coll1e2_std1, coll1e2_std2, coll1e2_std3,
                      coll1e3_std0, coll1e3_std1, coll1e3_std2, coll1e3_std3,
                      coll1e4_std0, coll1e4_std1, coll1e4_std2, coll1e4_std3,
                      coll1e5_std0, coll1e5_std1, coll1e5_std2, coll1e5_std3,
                      coll1e2_const1neg1, coll1e2_const1e0, coll1e2_const1e1, coll1e2_const1e2]
    exp_groups_coll1e2 = [coll1e2_std0, coll1e2_std1, coll1e2_std2, coll1e2_std3]
    exp_groups_coll1e3 = [coll1e3_std0, coll1e3_std1, coll1e3_std2, coll1e3_std3]
    exp_groups_coll1e4 = [coll1e4_std0, coll1e4_std1, coll1e4_std2, coll1e4_std3]
    exp_groups_coll1e5 = [coll1e5_std0, coll1e5_std1, coll1e5_std2, coll1e5_std3]
    exp_groups_std0 = [coll1e2_std0, coll1e3_std0, coll1e4_std0, coll1e5_std0]
    exp_groups_std1 = [coll1e2_std1, coll1e3_std1, coll1e4_std1, coll1e5_std1]
    exp_groups_std2 = [coll1e2_std2, coll1e3_std2, coll1e4_std2, coll1e5_std2]
    exp_groups_std3 = [coll1e2_std3, coll1e3_std3, coll1e4_std3, coll1e5_std3]
    exp_groups_const = [coll1e2_const1neg1, coll1e2_const1e0, coll1e2_const1e1, coll1e2_const1e2]

    # exp_groups_and_names = [(exp_groups_coll1e2, 'coll1e2')]
    exp_groups_and_names = [(exp_groups_coll1e2, 'coll1e2'),
                            (exp_groups_coll1e3, 'coll1e3'),
                            (exp_groups_coll1e4, 'coll1e4'),
                            (exp_groups_coll1e5, 'coll1e5'),
                            (exp_groups_std0, 'std0'),
                            (exp_groups_std1, 'std1'),
                            (exp_groups_std2, 'std2'),
                            (exp_groups_std3, 'std3'),
                            (exp_groups_const, 'const'),
                            (exp_groups_coll1e2 + exp_groups_const, 'std_vs_const')]
    print('Loading pickles')
    load_pickles(exp_groups_all)
    load_pickles([[old_std0, old_std1]], is_replay=True)
    print('Pickles loaded')

    # for exp_groups, name in exp_groups_and_names:
    #     print('Plotting {0}'.format(name))
    #     # f, axes = plt.subplots(1, len(exp_groups), figsize=(6 * len(exp_groups), 5), sharex=True, sharey=True)
    #     # for i, exp_group in enumerate(exp_groups):
    #     #     save_path = os.path.join(SAVE_FOLDER, 'pointquad_cum_crash_speeds_{0}.png'.format(name)) if i == len(exp_groups) - 1 else None
    #     #     plot_cum_crash_speeds([exp_group], ax=axes[i], max_speed=1, start_itr=2,
    #     #                           save_path=save_path)
    #     # plt.close(f)
    #     plot_boxplot_xvels(exp_groups, des_xvel=0.5, max_speed=1., start_itr=2, ylim=(-0.4, 1.0),
    #                        save_path=os.path.join(SAVE_FOLDER, 'pointquad_boxplot_xvels_{0}.png'.format(name)))

    # plot_safety_vs_performance([exp_groups_coll1e2, exp_groups_coll1e3, exp_groups_coll1e4, exp_groups_coll1e5,
    #                             exp_groups_std0, exp_groups_std1, exp_groups_std2, exp_groups_std3],
    #                            des_xvel=0.5,
    #                            nrows=2, start_itr=2, xtext_offset=-0.05, ytext_offset=-0.01, leg_offset=0.,
    #                            save_path=os.path.join(SAVE_FOLDER, 'pointquad_safety_vs_performance_{0}.png'.format('all')))

    # plot_safety_vs_performance([exp_groups_coll1e2, exp_groups_const],
    #                            des_xvel=0.5,
    #                            nrows=1, start_itr=2, xtext_offset=-0.15, ytext_offset=-0.06, leg_offset=0.45,
    #                            save_path=os.path.join(SAVE_FOLDER, 'pointquad_safety_vs_performance_{0}.png'.format('const')))

    plot_crash_speed_video([old_std0, old_std1], framerate=5, max_speed=1., include_collision=False,
                           save_folder='/home/gkahn/kdenlive/probcoll_sim')

    # plt.show(block=False)
    # import IPython; IPython.embed()
