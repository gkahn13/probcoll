import abc
import os
import time
import tensorflow as tf
import numpy as np
from collections import defaultdict

from general.algorithm.mlplotter import MLPlotter
from general.algorithm.probcoll_model_replay_buffer import ProbcollModelReplayBuffer 

class ProbcollModelRCcar(ProbcollModelReplayBuffer):

    def __init__(self, save_dir=None, data_dir=None):
        ProbcollModelReplayBuffer.__init__(self, save_dir=save_dir, data_dir=data_dir)
        self._setup_plotter()

    #############
    ### train ###
    #############

    def train(self, reset=False, **kwargs):
        if not self.train_replay_buffer.can_sample():
            self._logger.debug('Training skipped due to lack of data')
        else:
            self.graph.as_default()
            global_step = self.sess.run(self.global_step)
            self._logger.info('Training starting at step {0}'.format(global_step))
            reset = reset or (global_step >= self.reset_after_step) or global_step == 0
            if reset:
                self._reset()

            ### train
            train_values = defaultdict(list)
            train_nums = defaultdict(float)

            train_start = time.time()
            itr_steps = self.display_steps
            for _ in range(itr_steps):

                ### train
                feed_dict = self._get_bootstrap_batch_feed_dict(
                    self.train_replay_buffer,
                    self.d_train['U_inputs'],
                    self.d_train['O_im_inputs'],
                    self.d_train['O_vec_inputs'],
                    self.d_train['outputs'],
                    self.d_train['len'])
                _, global_step, train_cost, train_cross_entropy, \
                train_err, train_err_coll, train_err_nocoll, \
                train_coll, train_nocoll = self.sess.run(
                    [
                        self.d_train['optimizer'],
                        self.global_step,
                        self.d_train['cost'],
                        self.d_train['cross_entropy'],
                        self.d_train['err'],
                        self.d_train['err_coll'],
                        self.d_train['err_nocoll'],
                        self.d_train['num_coll'],
                        self.d_train['num_nocoll']
                    ],
                    feed_dict=feed_dict)

                train_values['cost'].append(train_cost)
                train_values['cross_entropy'].append(train_cross_entropy)
                train_values['err'].append(train_err)
                if not np.isnan(train_err_coll): train_values['err_coll'].append(train_err_coll)
                if not np.isnan(train_err_nocoll): train_values['err_nocoll'].append(train_err_nocoll)
                train_nums['coll'] += train_coll
                train_nums['nocoll'] += train_nocoll

            self._plotter.add_train('err', global_step * self.batch_size, np.mean(train_values['err']))
            if len(train_values['err_coll']) > 0:
                self._plotter.add_train('err_coll', global_step * self.batch_size, np.mean(train_values['err_coll']))
            if len(train_values['err_nocoll']) > 0:
                self._plotter.add_train('err_nocoll', global_step * self.batch_size, np.mean(train_values['err_nocoll']))
            self._plotter.add_train('cost', global_step * self.batch_size, np.mean(train_values['cost']))
            self._plotter.add_train('cross_entropy', global_step * self.batch_size, np.mean(train_values['cross_entropy']))

            self._logger.debug('error: {0:5.2f}%, error coll: {1:5.2f}%, error nocoll: {2:5.2f}%, pct coll: {3:4.1f}%, cost: {4:4.2f}, ce: {5:4.2f}'.format(
                100 * np.mean(train_values['err']),
                100 * np.mean(train_values['err_coll']),
                100 * np.mean(train_values['err_nocoll']),
                100 * train_nums['coll'] / (train_nums['coll'] + train_nums['nocoll']),
                np.mean(train_values['cost']),
                np.mean(train_values['cross_entropy'])))

            self._logger.debug('Training Speed: {0:.2f} samples / s'.format(
                float((itr_steps) * self.batch_size) /  (time.time() - train_start)))
            
            ### validation
            val_values = defaultdict(list)
            val_nums = defaultdict(float)
            val_steps = 0
            self._logger.debug('Computing validation...')
            while val_steps < self.val_steps:
                feed_dict = self._get_bootstrap_batch_feed_dict(
                    self.val_replay_buffer,
                    self.d_val['U_inputs'],
                    self.d_val['O_im_inputs'],
                    self.d_val['O_vec_inputs'],
                    self.d_val['outputs'],
                    self.d_val['len'])
                val_cost, val_cross_entropy, \
                        val_err, val_err_coll, val_err_nocoll, \
                        val_coll, val_nocoll = \
                    self.sess.run(
                        [
                            self.d_val['cost'],
                            self.d_val['cross_entropy'],
                            self.d_val['err'],
                            self.d_val['err_coll'],
                            self.d_val['err_nocoll'],
                            self.d_val['num_coll'],
                            self.d_val['num_nocoll']
                        ],
                        feed_dict)

                val_values['cost'].append(val_cost)
                val_values['cross_entropy'].append(val_cross_entropy)
                val_values['err'].append(val_err)
                if not np.isnan(val_err_coll): val_values['err_coll'].append(val_err_coll)
                if not np.isnan(val_err_nocoll): val_values['err_nocoll'].append(val_err_nocoll)
                val_nums['coll'] += val_coll
                val_nums['nocoll'] += val_nocoll

                val_steps += 1

            self._plotter.add_val('err', np.mean(val_values['err']))
            self._plotter.add_val('err_coll', np.mean(val_values['err_coll']))
            self._plotter.add_val('err_nocoll', np.mean(val_values['err_nocoll']))
            self._plotter.add_val('cost', np.mean(val_values['cost']))
            self._plotter.add_val('cross_entropy', np.mean(val_values['cross_entropy']))

            self._logger.debug(
                'error: {0:5.2f}%, error coll: {1:5.2f}%, error nocoll: {2:5.2f}%, pct coll: {3:4.1f}%, cost: {4:4.2f}, ce: {5:4.2f}'.format(
                    100 * np.mean(val_values['err']),
                    100 * np.mean(val_values['err_coll']),
                    100 * np.mean(val_values['err_nocoll']),
                    100 * val_nums['coll'] / (val_nums['coll'] + val_nums['nocoll']),
                    np.mean(val_values['cost']),
                    np.mean(val_values['cross_entropy'])))


            global_step = self.sess.run(self.global_step)
            ### save model
            if global_step >= self.ckpt_after_step:
                new_model_file, model_num  = self._next_model_file()
                self.save(new_model_file)
                self._plotter.save(self._plots_dir, suffix=str(model_num))

    def _setup_plotter(self):
        if hasattr(self, '_plotter'):
            self._plotter.close()
        self._plotter = MLPlotter(
            self.save_dir,
            {
                'err': {
                    'title': 'Error',
                    'subplot': 0,
                    'color': 'k',
                    'ylabel': 'Percentage'
                },
                'err_coll': {
                    'title': 'Error Collision',
                    'subplot': 1,
                    'color': 'r',
                    'ylabel': 'Percentage'
                },
                'err_nocoll': {
                    'title': 'Error no collision',
                    'subplot': 2,
                    'color': 'g',
                    'ylabel': 'Percentage'
                },
                'cost': {
                    'title': 'Cost',
                    'subplot': 3,
                    'color': 'k',
                    'ylabel': 'cost'
                },
                'cross_entropy': {
                    'subplot': 3,
                    'color': 'm'
                }
            })

    def _reset(self):
        self._logger.info('Resetting model')
        self.graph_init_vars()
        ### create plotter
        self._setup_plotter()

    #############################
    ### Load/save/reset/close ###
    #############################

    def get_latest_checkpoint(self):
        ckpt_file = os.path.join(self.checkpoints_dir, "0.ckpt")
        check_file = os.path.join(self.checkpoints_dir, "0.ckpt.index")
        if os.path.exists(check_file):
            return ckpt_file
        else:
            return None

    def recover(self):
        latest_file = self.get_latest_checkpoint()
        if latest_file is not None:
            while True:
                try:
                    self.load(latest_file)
                    self._logger.info("Found checkpoint file")
                    break
                except Exception as e:
                    self._logger.warning("Could not find checkpoint file")
                    self._logger.warning(e)
        else:
            self._logger.warning("Could not find checkpoint file")


    def close(self):
        """ Release tf session """
        if hasattr(self, 'coord'):
            assert(hasattr(self, 'threads'))
            self.coord.request_stop()
            self.coord.join(self.threads)
        self.sess.close()
        self.sess = None
        if hasattr(self, '_probcoll'):
            self._plotter.close()
