import os

import numpy as np

import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.patches import Rectangle
import matplotlib.ticker as ticker
from matplotlib import gridspec

from general.algorithm.analyze_aggregate_prediction import load_pickles, flatten_list, plot_crash_speed_video, DataAverageInterpolation

from general.state_info.sample import Sample

EXP_FOLDER = '/home/gkahn/code/gps_quadrotor/experiments/rccar/'
# SAVE_FOLDER = '/home/gkahn/Dropbox/Apps/ShareLaTeX/2017_RSS_ProbColl/figures/exps/'
SAVE_FOLDER = '/home/gkahn/tmp'

### lambda_std = 0
std0 = []
for i, exp_num in enumerate([50, 51]):
    std0.append({
        'folder': os.path.join(EXP_FOLDER, 'exp{0}'.format(exp_num)),
        'label': '$\lambda_{\\textsc{std}} = 0$',
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
std1 = []
for i, exp_num in enumerate([55, 56]):
    std1.append({
        'folder': os.path.join(EXP_FOLDER, 'exp{0}'.format(exp_num)),
        'label': '$\lambda_{\\textsc{std}} = 1$ (ours)',
        'color': cm.Blues(0.9),
        'marker': 'o',
        'ms': 8,
        'mew': 2,
        'legend_loc': 'middle right',
        'bbox_to_anchor': (1.22, 0.5),
        # 'color': cm.Reds(0.7),
        # 'marker': 'd',
    })

def plot_safety_vs_performance(exp_groups_list, des_xvel, min_xvel,
                               xtext_offset=-0.05, ytext_offset=-0.01, leg_offset=0.,
                               start_itr=0, std_weight=0.674490, nrows=1, fs=20., save_path=None):

    ### plot final flight speed (x axis) vs crash speed (y axis)
    ncols = len(exp_groups_list) // nrows
    f, axes = plt.subplots(nrows, ncols, figsize=(ncols * 5, nrows * 4),
                           sharex=True, sharey=True)
    if nrows == 1:
        if ncols == 1:
            axes = np.array([[axes]])
        else:
            axes = np.expand_dims(axes, 0)
    f.tight_layout()
    legend_handles = []
    legend_labels = []

    T = 0
    for exp_groups in exp_groups_list:
        for exp_group in exp_groups:
            for exp in exp_group:
                for U_itr in exp['U']:
                    for U in U_itr:
                        T = max(T, len(U))

    for i, (ax, exp_groups) in enumerate(zip(axes.ravel(), exp_groups_list)):
        end_itr = min([len(exp['crash_speeds']) for exp in flatten_list(exp_groups)])

        for exp_group in exp_groups:
            crash_speeds = []
            for exp in exp_group:
                crash_speeds += [item - min_xvel for sublist in exp['crash_speeds'][start_itr:end_itr] for item in sublist]

            final_speeds = []
            for exp in exp_group:
                # final_speeds += flatten_list([list(U[:, 0]) for U in exp['U'][end_itr - 1]])
                final_speeds += flatten_list([list(U[:, 1] - min_xvel) + [0]*(T - len(U)) for U in exp['U'][end_itr-1]])

            if len(crash_speeds) == 0:
                crash_speeds = [0]

            color = exp_group[0]['color']
            marker = exp_group[0]['marker']
            ax.errorbar([np.mean(final_speeds)], [np.mean(crash_speeds)],
            # ax.errorbar([np.median(final_speeds)], [np.median(crash_speeds)],
                        fmt=marker, mfc=color, mec=color, ms=10.,
                        xerr=std_weight * np.std(final_speeds), yerr=std_weight * np.std(crash_speeds),
                        # xerr=abs(np.array([[np.percentile(final_speeds, 25), np.percentile(final_speeds, 75)]]).T - np.median(final_speeds)),
                        # yerr=abs(np.array([[np.percentile(crash_speeds, 25), np.percentile(crash_speeds, 75)]]).T - np.median(crash_speeds)),
                        color=color)
            if i // ncols == 0:
                legend_handles += ax.plot([], [], marker, ms=10., color=color, mfc=color, mec=color)
                legend_labels.append(exp_group[0]['label'])

        ax.set_xlabel('({0})'.format(chr(ord('a') + i)))
        ax.set_ylim((-0.05, ax.get_ylim()[1]))

    for i, ax in enumerate(axes.ravel()):
        line = ax.plot((des_xvel, des_xvel), ax.get_ylim(), 'k:', lw=3.)[0]
        if i == 0:
            legend_handles.insert(0, line)
            legend_labels.insert(0, 'Desired speed')

        ax.xaxis.label.set_size(fs)
        ax.yaxis.label.set_size(fs)
        for tick in ax.xaxis.get_major_ticks() + ax.yaxis.get_major_ticks():
            tick.label.set_fontsize(fs)

    for ax in axes[-1, :]:
        ax.xaxis.get_major_ticks()[0].label1.set_visible(False)

    extra_artists = []
    extra_artists.append(axes[0, -1].legend(legend_handles, legend_labels, loc='center left', bbox_to_anchor=(1.05, leg_offset), fontsize=fs))

    extra_artists.append(f.text(0.5, xtext_offset, 'Speeds on final iteration (m/s)', ha='center', fontsize=fs))
    extra_artists.append(f.text(ytext_offset, 0.5, 'Crash speeds (m/s)\nfrom all training iterations', va='center', rotation='vertical', fontsize=fs))

    if save_path is not None:
        f.savefig(save_path, bbox_extra_artists=extra_artists, bbox_inches='tight')

    plt.close(f)

def plot_crash_timeline(exp_groups, max_speed, min_speed, fs=20., end_itr=None, save_path=None):
    ### crash timeline
    f, ax = plt.subplots(1, 1, figsize=(20, 5))

    if end_itr is None:
        end_itr = min([len(exp['U']) for exp in flatten_list(exp_groups)])
    reps = len(exp_groups[0][0]['U'][0])
    T = 0
    for exp_group in exp_groups:
        for exp in exp_group:
            for U_itr in exp['U']:
                for U in U_itr:
                    T = max(T, len(U))
    assert(np.equal([[[len(U) for U in exp['U']] for exp in exp_group] for exp_group in exp_groups], reps).all())

    exps_crash_speeds = []
    for exp in flatten_list(exp_groups):
        crash_speeds = []
        for U_itr in exp['U']:
            for U in U_itr:
                if len(U) == T:
                    crash_speeds.append(0.)
                else:
                    crash_speeds.append(max_speed * (U[-1,1] - min_speed))
        exps_crash_speeds.append(crash_speeds)

    for r in xrange(end_itr * reps):
        ax.cla()

        for exp, crash_speeds in zip(flatten_list(exp_groups), exps_crash_speeds):
            ax.plot(np.linspace(0, r / float(reps), len(crash_speeds[:r])), crash_speeds[:r],
                    marker=exp['marker'], color=exp['color'], mec=exp['color'], mfc=exp['color'], ms=exp['ms'], mew=exp['mew'],
                    linestyle='')

        legends = []
        for exp_group in exp_groups:
            color = exp_group[0]['color']
            leg = ax.legend(
                ax.plot([], [], marker=exp_group[0]['marker'], color=color, mfc=color, mec=color, ms=exp_group[0]['ms'], mew=exp_group[0]['mew'],
                        linestyle=''),
                [exp_group[0]['label']],
                loc=exp_group[0]['legend_loc'],
                bbox_to_anchor=exp_group[0]['bbox_to_anchor'],
                fontsize=fs)
            legends.append(leg)
            ax.add_artist(leg)

        ax.set_xlim((-0.1, end_itr + 0.1))
        ax.set_ylim((-0.1, max(flatten_list(exps_crash_speeds)) + 0.2))
        ax.set_xticks([i for i in xrange(end_itr)])
        for tick in ax.xaxis.get_major_ticks() + ax.yaxis.get_major_ticks():
            tick.label.set_fontsize(fs)

        ax.set_title(r'Crash speeds per rollout when not accounting for uncertainty (left) versus with accounting for uncertainty (right)')
        ax.set_xlabel(r'Iteration', fontsize=fs)
        ax.set_ylabel(r'Speed (m/s)', fontsize=fs)

        # f.canvas.draw()
        # plt.show(block=False)

        if save_path:
            from matplotlib.transforms import Bbox
            f.savefig(save_path.replace('.png', '_{0:03d}.png'.format(r)),
                      bbox_extra_artists=legends, bbox_inches=Bbox([[-1.5, -0.5], [21.5, 5]]))

        # import IPython; IPython.embed()

def plot_num_successful(exp_groups, ax=None, fs=20., save_path=None, is_legend=False):
    if ax is None:
        f, ax = plt.subplots(1, 1, figsize=(4, 4))
    else:
        f = ax.get_figure()

    end_itr = min([len(exp['U']) for exp in flatten_list(exp_groups)])
    reps = len(exp_groups[0][0]['U'][0])
    T = 10

    for itr in xrange(0, end_itr):
        ax.axvspan(itr, itr + 1, facecolor=cm.Greys(0.05 + (itr % 2) * 0.1), edgecolor='none')

    for exp_group_num, exp_group in enumerate(exp_groups):
        frac_successful_itrs_group = []
        for exp in exp_group:
            num_successful_itrs = np.array([0.] * end_itr)
            for itr in xrange(end_itr):
                for rep in xrange(reps):
                    if exp['stop_rollout'][itr][rep] or len(exp['U'][itr][rep]) == T:
                        num_successful_itrs[itr] += 1.

            frac_successful_itrs = num_successful_itrs / (reps)
            frac_successful_itrs_group.append(frac_successful_itrs)

        frac_successful_itrs_group = 100 * np.array(frac_successful_itrs_group)

        width = 0.8 / len(exp_groups)
        ax.bar(np.arange(end_itr) + exp_group_num * width + width/4., frac_successful_itrs_group.mean(axis=0),
                yerr=frac_successful_itrs_group.std(axis=0),
                width=width*0.8,
                error_kw=dict(lw=2, capsize=3, capthick=1),
                color=exp_group[0]['color'], ecolor=cm.Greys(0.5), label=exp_group[0]['label'])


    ax.set_xlabel(r'Iteration', fontsize=fs)
    ax.set_ylabel(r'\begin{center}Successful\\rollouts (\%)\end{center}', fontsize=fs)

    ax.set_ylim((0, 100))

    N = end_itr
    ax.set_xticks(np.arange(N) + 0.5)
    ax.set_xticklabels([str(x) for x in range(N)])
    # Hide major tick labels and customize minor tick labels
    ax.xaxis.set_minor_formatter(ticker.NullFormatter())
    ax.xaxis.set_major_locator(ticker.FixedLocator(np.arange(N) + 0.5))
    ax.xaxis.set_major_formatter(ticker.FixedFormatter([r'{0}'.format(str(x)) for x in range(N)]))
    for tick in ax.xaxis.get_major_ticks() + ax.yaxis.get_major_ticks():
        tick.label.set_fontsize(fs)

    if is_legend:
        leg = ax.legend(loc='upper left', ncol=2, bbox_to_anchor=(-0.5, 1.5), fontsize=fs)
    else:
        leg = None

    if save_path is not None:
        f.tight_layout()
        f.savefig(save_path, bbox_extra_artists=(leg,), bbox_inches='tight', dpi=200)


def plot_cum_crash_speeds(exp_groups, crash_topic, max_mps, min_speed, ax=None, is_legend=True, start_itr=0, std=0.674490, save_path=None, fs=20.):
    ### plot crash speeds (x axis) vs num crashes <= that speed (y axis)
    if ax is None:
        f, ax = plt.subplots(1, 1, figsize=(4, 2))
    else:
        f = ax.get_figure()

    end_itr = min([len(exp[crash_topic]) for exp in flatten_list(exp_groups)])
    max_crash_speed = -np.inf
    for exp in flatten_list(exp_groups):
        crash_speeds = [item - min_speed for sublist in exp[crash_topic][start_itr:end_itr] for item in sublist]
        if len(crash_speeds) > 0:
            max_crash_speed = max(max_crash_speed, max(crash_speeds))


    for exp_group in exp_groups:
        if not np.isinf(max_crash_speed):
            data_interp = DataAverageInterpolation()
            for exp in exp_group:
                crash_speeds = sorted([(max_mps / max_crash_speed) * (item - min_speed) for sublist in exp[crash_topic][start_itr:end_itr] for item in sublist], reverse=True)
                # crash_speeds = sorted([item for sublist in exp[crash_topic][start_itr:end_itr] for item in sublist], reverse=True)
                counts = list(np.r_[1:1 + len(crash_speeds):1])
                if len(crash_speeds) == 0 or np.abs(max(crash_speeds) - max_crash_speed) > 1e-3:
                    crash_speeds.insert(0, max_crash_speed)
                    counts.insert(0, 0)
                crash_speeds.append(0.)
                counts.append(counts[-1])
                data_interp.add_data(crash_speeds, counts)

            crash_speeds = np.r_[min(flatten_list(data_interp.xs)):max(flatten_list(data_interp.xs)):0.01][1:-1]
            counts_mean, counts_std = data_interp.eval(crash_speeds)
            counts_std[...] = 10.

            ax.fill_between(crash_speeds, counts_mean - std * counts_std, counts_mean + std * counts_std,
                            color=exp_group[0]['color'], alpha=0.8, edgecolor='k')
            ax.plot(crash_speeds, counts_mean, color='white', lw=2.)

        ax.plot([], [], color=exp_group[0]['color'], label=exp_group[0]['label'])

    ax.set_xlim((0, max_mps))
    # ax.set_xlim((0, 5.))
    ax.set_ylim((0, ax.get_ylim()[1]))

    if is_legend:
        leg = ax.legend(fontsize=fs*0.8)
        for legobj in leg.legendHandles:
            legobj.set_linewidth(4.0)
    ax.set_xlabel(r'Speed (m/s)', fontsize=fs*0.8)
    ax.set_ylabel(r'\begin{center}Number of crashes\\at this speed or above\end{center}', fontsize=fs*0.8)

    for tick in ax.xaxis.get_major_ticks() + ax.yaxis.get_major_ticks():
        tick.label.set_fontsize(fs)
    ax.yaxis.get_major_ticks()[0].label1.set_visible(False)

    if save_path is not None:
        f.savefig(save_path, bbox_extra_artists=(leg,), bbox_inches='tight', dpi=200)

    # plt.close(f)

def plot_boxplot_speeds(exp_groups, des_mps, des_speed, min_speed, ax=None, is_legend=True, start_itr=0, save_path=None, ylim=None, fs=20.):
    end_itr = min([len(exp['U']) for exp in flatten_list(exp_groups)])
    itrs_offset = np.linspace(0.35, 0.65, len(exp_groups))
    if ax is None:
        f, ax = plt.subplots(1, 1, figsize=(1.5 * (end_itr - start_itr), 5))
    else:
        f = ax.get_figure()

    T = 0
    for exp_group in exp_groups:
        for exp in exp_group:
            for U_itr in exp['U']:
                for U in U_itr:
                    T = max(T, len(U))

    for exp_group_num, exp_group in enumerate(exp_groups):
        xvels_itrs = []
        for itr in xrange(start_itr, end_itr):
            xvels = []
            for exp in exp_group:
                xvels += flatten_list([list((des_mps / des_speed) * (U[:, 1] - min_speed)) + [0] * (T - len(U)) for U in exp['U'][itr]])
            xvels_itrs.append(xvels)

            color = exp_group[0]['color']
            ax.errorbar([itr + itrs_offset[exp_group_num]], [np.mean(xvels)],
                        [[np.mean(xvels) - np.min(xvels)], [np.max(xvels) - np.mean(xvels)]],
                        fmt='.k', ecolor='gray', lw=1.5, capsize=5., capthick=2.)
            ax.errorbar(itr + itrs_offset[exp_group_num], np.mean(xvels), np.std(xvels),
                        marker='o', color=color, ecolor=color, mec=color, lw=3.)

    des_xvel_handle = ax.plot((start_itr, end_itr), (des_mps, des_mps), color='k', ls=':', lw=2.)[0]

    ax.set_xlim((start_itr, end_itr))
    ax.set_ylim((0, des_mps*1.1))

    if ylim is not None:
        ax.set_ylim(ylim)
    # background color
    for itr in xrange(start_itr, end_itr):
        ax.axvspan(itr, itr + 1, facecolor=cm.Greys(0.05 + (itr % 2) * 0.1), edgecolor='none')
    ax.set_xticks(np.arange(start_itr, end_itr) + 0.5)
    ax.set_xticklabels([str(x) for x in range(start_itr, end_itr)])
    # Hide major tick labels and customize minor tick labels
    ax.xaxis.set_major_formatter(ticker.NullFormatter())
    ax.xaxis.set_minor_locator(ticker.FixedLocator(np.arange(start_itr, end_itr) + 0.5))
    ax.xaxis.set_minor_formatter(ticker.FixedFormatter([r'{0}'.format(str(x)) for x in range(start_itr, end_itr)]))
    ax.set_xlabel('Iteration', fontsize=fs)
    ax.set_ylabel('Speed (m/s)', fontsize=fs)
    for tick in ax.xaxis.get_minor_ticks() + ax.yaxis.get_major_ticks():
        tick.label.set_fontsize(fs)

    if is_legend:
        # Create legend
        extra = Rectangle((0, 0), 1, 1, fc="w", fill=False, edgecolor='none', linewidth=0)
        legend_handles = []
        legend_labels = []

        legend_handles.append(des_xvel_handle)
        legend_labels.append('')
        legend_handles.append(extra)
        legend_labels.append('Desired speed')
        for exp_group in exp_groups:
            legend_handles.append(Rectangle((0, 0), 1, 1, color=exp_group[0]['color'], fill=True, linewidth=1))
            legend_labels.append('')
            legend_handles.append(extra)
            legend_labels.append(exp_group[0]['label'])
        lgd = f.legend(legend_handles, legend_labels, loc='upper center', bbox_to_anchor=(0.45, 1.15),
                        ncol=len(legend_handles), handletextpad=-2, handlelength=3., fontsize=fs)

    if save_path is not None:
        f.tight_layout()
        f.savefig(save_path, bbox_extra_artists=(lgd,), bbox_inches='tight')

    # plt.close(f)

def plot_distance_travelled(exp_groups, des_mps, des_speed, min_speed, ax=None, fs=20., save_path=None):
    if ax is None:
        f, ax = plt.subplots(1, 1, figsize=(20, 5))
    else:
        f = ax.get_figure()
    itrs_offset = np.linspace(0.3, 0.7, len(exp_groups))

    T = 10
    dt = 0.5
    end_itr = min([len(exp['U']) for exp in flatten_list(exp_groups)])

    for exp_group_num, exp_group in enumerate(exp_groups):
        distance_itrs = []
        for itr in xrange(end_itr):
            distances = []
            for exp in exp_group:
                running_distance = 0.
                for U in exp['U'][itr]:
                    distances.append(np.sum(dt * (des_mps / des_speed) * (U[:, 1] - min_speed)))
            distance_itrs.append(distances)

            # ax.plot([itr + itrs_offset[exp_group_num]] * len(distances), distances,
            #         color=exp_group[0]['color'],
            #         marker='_',
            #         ms=10., mew=2.,
            #         linestyle='')

        bp = ax.boxplot(distance_itrs,
                        positions=np.arange(len(distance_itrs)) + itrs_offset[exp_group_num],
                        widths=0.2 * np.ones(len(distance_itrs)))
        color = exp_group[0]['color']
        for key in ('boxes', 'medians', 'whiskers', 'fliers', 'caps'):
            plt.setp(bp[key], color=color)
        plt.setp(bp['fliers'], marker='_')
        plt.setp(bp['fliers'], markeredgecolor=color)
        plt.setp(bp['fliers'], mew=2.)
        plt.setp(bp['boxes'][0], label=exp_group[0]['label'])


    ax.set_xlabel(r'Iteration', fontsize=fs)
    ax.set_ylabel(r'\begin{center}Distance travelled\\until collision (m)\end{center}', fontsize=fs)

    ax.set_xlim((0, end_itr))

    ax.set_xticks(np.arange(0, end_itr) + 0.5)
    ax.set_xticklabels([str(x) for x in range(end_itr)])
    # Hide major tick labels and customize minor tick labels
    ax.xaxis.set_major_formatter(ticker.NullFormatter())
    ax.xaxis.set_minor_locator(ticker.FixedLocator(np.arange(end_itr) + 0.5))
    ax.xaxis.set_minor_formatter(ticker.FixedFormatter([r'{0}'.format(str(x)) for x in range(end_itr)]))
    for tick in ax.xaxis.get_minor_ticks() + ax.yaxis.get_major_ticks():
        tick.label.set_fontsize(fs)

    if save_path is not None:
        f.savefig(save_path, bbox_inches='tight')

def plot_final_policy(exp_groups, des_mps, des_speed, min_speed, ax=None, fs=20., save_path=None):
    dt = 0.5

    if ax is None:
        f, ax = plt.subplots(1, 1, figsize=(10, 3.5))
    else:
        f = ax.get_figure()

    exp_groups.insert(0, [{'folder': os.path.join(EXP_FOLDER, 'random'),
                           'label': 'Random policy',
                           'color': cm.Greys(0.5)}])

    for exp_num, exp in enumerate(flatten_list(exp_groups)):
        samples_folder = os.path.join(exp['folder'], 'prediction/final_policy/')
        samples_fnames = [os.path.join(samples_folder, fname) for fname in os.listdir(samples_folder)
                          if fname[-4:] == '.npz']
        assert(len(samples_fnames) == 1)
        samples_fname = samples_fnames[0]

        samples = Sample.load(samples_fname)
        distance_travelled = []
        for sample in samples:
            distance = dt * (des_mps / des_speed) * (sample.get_U()[:, 1] - min_speed).sum()
            distance_travelled.append(distance)

        # ax.plot(range(len(distance_travelled)), distance_travelled,
        #         linestyle='',
        #         marker=exp['marker'],
        #         color=exp['color'],
        #         mew=exp['mew'])

        ax.bar(np.arange(len(distance_travelled)) + 0.15 * exp_num + 0.1,
               distance_travelled,
               width=0.15,
               color=exp['color'])

    for exp_group in exp_groups:
        ax.plot([], [], lw=5., label=exp_group[0]['label'], color=exp_group[0]['color'])
    lgd = ax.legend(loc='upper right', fontsize=fs*0.8)

    ax.set_xlabel(r'Starting position', fontsize=fs)
    ax.set_ylabel(r'Distance travelled (m)', fontsize=fs)

    N = len(distance_travelled)
    ax.set_xticks(np.arange(N) + 0.5)
    ax.set_xticklabels([str(x) for x in range(N)])
    # Hide major tick labels and customize minor tick labels
    ax.xaxis.set_minor_formatter(ticker.NullFormatter())
    ax.xaxis.set_major_locator(ticker.FixedLocator(np.arange(N) + 0.5))
    ax.xaxis.set_major_formatter(ticker.FixedFormatter([r'{0}'.format(str(x)) for x in range(N)]))
    for tick in ax.xaxis.get_major_ticks() + ax.yaxis.get_major_ticks():
        tick.label.set_fontsize(fs)

    if save_path is not None:
        f.savefig(save_path, bbox_extra_artists=(lgd,), bbox_inches='tight')

def plot_final_policy_combined(exp_groups, des_mps, des_speed, min_speed, ax=None, fs=20., save_path=None):
    dt = 0.5

    if ax is None:
        f, ax = plt.subplots(1, 1, figsize=(5, 3.5))
    else:
        f = ax.get_figure()

    exp_groups.insert(0, [{'folder': os.path.join(EXP_FOLDER, 'random'),
                           'label': 'Random policy',
                           'color': cm.Greys(1.0)}])

    for exp_group_num, exp_group in enumerate(exp_groups):
        distance_travelled = []
        for exp in exp_group:
            samples_folder = os.path.join(exp['folder'], 'prediction/final_policy/')
            samples_fnames = [os.path.join(samples_folder, fname) for fname in os.listdir(samples_folder)
                              if fname[-4:] == '.npz']
            assert(len(samples_fnames) == 1)
            samples_fname = samples_fnames[0]

            samples = Sample.load(samples_fname)
            for sample in samples:
                distance = dt * (des_mps / des_speed) * (sample.get_U()[:, 1] - min_speed).sum()
                distance_travelled.append(distance)

        width = 0.8 / len(exp_groups)
        ax.bar([exp_group_num * width], [np.mean(distance_travelled)], yerr=[np.std(distance_travelled)],
                width=width*0.8,
                error_kw=dict(lw=4, capsize=5, capthick=2),
                color=exp_group[0]['color'], ecolor=cm.Greys(0.5), label=exp_group[0]['label'])

    # for exp_group in exp_groups:
    #     ax.plot([], [], lw=5., label=exp_group[0]['label'], color=exp_group[0]['color'])
    lgd = ax.legend(loc='upper center', fontsize=fs*0.8, bbox_to_anchor=(1.35, 0.7))

    ax.set_xlim((-0.05, width * len(exp_groups)))
    ax.set_ylim((-0.3, ax.get_ylim()[1]))
    ax.set_ylabel(r'Distance travelled (m)', fontsize=fs)

    for tick in ax.xaxis.get_major_ticks() + ax.xaxis.get_minor_ticks():
        tick.label1.set_visible(False)
    for tick in ax.yaxis.get_major_ticks():
        tick.label.set_fontsize(fs)

    if save_path is not None:
        f.savefig(save_path, bbox_extra_artists=(lgd,), bbox_inches='tight')

def plot_crash_timeline(exp_groups, des_mps, des_speed, min_speed, num_bins=11, max_counts=130, fs=50., save_folder=None):
    ### crash timeline
    f, ax = plt.subplots(1, 1, figsize=(15, 10))

    end_itr = min([len(exp['U']) for exp in flatten_list(exp_groups)])
    reps = len(exp_groups[0][0]['U'][0])

    T = 20

    for exp in flatten_list(exp_groups):
        crash_speeds = []
        for U_itr in exp['U']:
            for U in U_itr:
                if len(U) == T:
                    crash_speeds.append(None)
                else:
                    crash_speeds.append((des_mps / des_speed) * np.linalg.norm(U[-1, 1] - min_speed))

        for r in xrange(end_itr * reps):
            ax.cla()
            crash_speeds_r = [cs for cs in crash_speeds[:r] if cs is not None]
            ax.hist(crash_speeds_r, np.linspace(0, des_mps*1.3, num_bins), histtype='bar',
                    weights=np.ones(len(crash_speeds_r)),
                    color=exp['color'],
                    label=exp['label'])
            ax.legend(loc='upper right', fontsize=fs)
            ax.set_xlim((0, des_mps))
            ax.set_ylim((0, max_counts))
            ax.set_xlabel(r'\textbf{Speed (m/s)}', fontsize=fs)
            ax.set_ylabel(r'\textbf{Number of crashes}', fontsize=fs)
            ax.set_title(r'Iteration {0}'.format(r // reps), fontsize=fs)
            for tick in ax.xaxis.get_major_ticks() + ax.yaxis.get_major_ticks():
                tick.label.set_fontsize(fs)
            ax.yaxis.get_major_ticks()[0].label1.set_visible(False)
            plt.tight_layout()

            if save_folder:
                from matplotlib.transforms import Bbox
                outfile = os.path.join(save_folder, 'crash_{0}_{1:03d}.png'.format(
                    os.path.basename(exp['folder']).split('.')[0], r))
                f.savefig(outfile, bbox_extra_artists=None)#, bbox_inches=Bbox([[-1.5, -0.5], [21.5, 5]]))

            # import IPython; IPython.embed()

if __name__ == '__main__':

    exp_groups_and_names = [([std0, std1], 'all')]

    print('Loading pickles')
    load_pickles(flatten_list(zip(*exp_groups_and_names)[0]))
    print('Pickles loaded')

    # for exp_groups, name in exp_groups_and_names:
    #     print('Plotting {0}'.format(name))
    #     fig = plt.figure(figsize=(15, 3))
    #     gs = gridspec.GridSpec(1, 3, width_ratios=[1, 2, 2])
    #     axes = (plt.subplot(gs[0]), plt.subplot(gs[1]), plt.subplot(gs[2]))
    #     # fig, axes = plt.subplots(1, 2, figsize=(10,3))
    #
    #     plot_cum_crash_speeds(exp_groups, crash_topic='crash_speeds', max_mps=1.2, min_speed=2., ax=axes[0], is_legend=False, save_path=None, std=1.)
    #     plot_num_successful(exp_groups, ax=axes[2], save_path=None, is_legend=False)#os.path.join(SAVE_FOLDER, 'rccar_cone_{0}.png'.format(name)))
    #
    #     # plot_cum_crash_speeds(exp_groups, crash_topic='encoder_crash_speeds',
    #     #                       max_mps=1., min_speed=0, ax=axes[0], is_legend=False, save_path=None, std=1.)
    #     # plot_distance_travelled(exp_groups, des_mps=1.2, des_speed=8., min_speed=2., ax=axes[2])
    #     plot_boxplot_speeds(exp_groups, des_mps=1.2, des_speed=8., min_speed=2., ax=axes[1], is_legend=True, start_itr=0, ylim=None,
    #                        save_path=os.path.join(SAVE_FOLDER, 'rccar_cone_{0}.png'.format(name)))

    # USING
    # plot_cum_crash_speeds([std0, std1], crash_topic='crash_speeds', max_mps=1.2, min_speed=2.0,
    #                       ax=None, is_legend=True, std=1., fs=16.,
    #                       save_path=os.path.join(SAVE_FOLDER, 'rccar_cone_cum_crash.png'))

    # USING
    # plot_final_policy([std0, std1], des_mps=1.5, des_speed=14., min_speed=6., fs=20.,
    #                   save_path=os.path.join(SAVE_FOLDER, 'rccar_final_policy.png'))
    #
    # # USING
    # plot_final_policy_combined([std0, std1], des_mps=1.5, des_speed=14., min_speed=6., fs=20.,
    #               save_path=os.path.join(SAVE_FOLDER, 'rccar_final_policy_combined.png'))


    # plot_safety_vs_performance([[std0, std1]],
    #                            des_xvel=8., min_xvel=6.,
    #                            nrows=1, start_itr=2, xtext_offset=-0.15, ytext_offset=-0.03, leg_offset=0.45,
    #                            save_path=os.path.join(SAVE_FOLDER,
    #                                                   'rccar_safety_vs_performance_{0}.png'.format('all')))

    plot_crash_timeline([[std0[1], std1[1]]], des_mps=1.2, des_speed=8., min_speed=2.,
                        save_folder='/home/gkahn/tmp/rccar_timeline/')


    # plt.show(block=False)
    # import IPython; IPython.embed()
