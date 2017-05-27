import os, pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
#matplotlib.use('Agg')
class MLPlotter:
    """
    Plot/save machine learning data
    """
    def __init__(self, title, subplot_dicts, shape=None, figsize=None):
        """
        :param title: title of plot
        :param subplot_dicts: dictionary with dictionaries of form
                              name: {subplot, title, color, ylabel}
        """
        ### setup plot figure
        num_subplots = max(d['subplot'] for d in subplot_dicts.values()) + 1
        if shape is None:
            shape = (1, num_subplots)
        if figsize is None:
            figsize = (30, 7)
        self.f, self.axes = plt.subplots(shape[0], shape[1], figsize=figsize)
#        mng = plt.get_current_fig_manager()
#        mng.window.showMinimized()
        plt.suptitle(title)
#        plt.show(block=False)
#        plt.pause(0.01)

        self.train_lines = {}
        self.val_lines = {}

        axes = self.axes.ravel().tolist()
        for name, d in subplot_dicts.items():
            ax = axes[d['subplot']]
            ax.set_xlabel('Training samples')
            if 'title' in d: ax.set_title(d['title'])
            if 'ylabel' in d: ax.set_ylabel(d['ylabel'])

            self.train_lines[name] = ax.plot([], [], color=d['color'], linestyle='-', label=name)[0]
            self.val_lines[name] = ax.plot([], [], color=d['color'], linestyle='--')[0]

            ax.legend()

#        self.f.canvas.draw()
#        plt.pause(0.01)

    def _update_line(self, line, new_x, new_y):
        xdata, ydata = line.get_xdata(), line.get_ydata()

        xdata = np.concatenate((xdata, [new_x]))
        ydata = np.concatenate((ydata, [new_y]))

        line.set_xdata(xdata)
        line.set_ydata(ydata)

        ax = line.axes
        ax.relim()
        ax.autoscale_view()
        ax.set_ylim([0, 1])

    def add_train(self, name, training_samples, value):
        self._update_line(self.train_lines[name], training_samples, value)

    def add_val(self, name, value):
        xdata = self.train_lines[name].get_xdata()
        self._update_line(self.val_lines[name], xdata[-1] if len(xdata) > 0 else 0, value)

    def plot(self):
        self.f.canvas.draw()
        plt.pause(0.01)

    def save(self, save_dir, suffix=""):
        fig_name = "training_{0}.png".format(suffix)
        pkl_name = "plotter_{0}.pkl".format(suffix)
        self.f.savefig(os.path.join(save_dir, fig_name))
        with open(os.path.join(save_dir, pkl_name), 'w') as f:
            pickle.dump(dict([(k, (v.get_xdata(), v.get_ydata())) for k, v in self.train_lines.items()] +
                             [(k, (v.get_xdata(), v.get_ydata())) for k, v in self.val_lines.items()]),
                        f)

    def close(self):
        plt.close(self.f)

