import os
import random
import time
import itertools
import shutil
from collections import defaultdict
import hashlib
import numpy as np
import tensorflow as tf
import sys
import copy
import string

from general.tf import tf_utils
from general.tf.nn.fc_nn import fcnn
from general.tf.nn.conv_nn import convnn
from general.tf.nn.separable_conv_nn import separable_convnn
from general.tf.nn.rnn import rnn
from general.utility.logger import get_logger
from general.state_info.sample import Sample
from general.algorithm.mlplotter import MLPlotter
from general.algorithm.probcoll_model import ProbcollModel
from general.algorithm.replay_buffer import SplitReplayBuffer
from general.algorithm.replay_buffer import ReplayBuffer
from config import params


class ProbcollModelReplayBuffer(ProbcollModel):
    ####################
    ### Initializing ###
    ####################

    def __init__(self, save_dir=None, data_dir=None):
        if save_dir is None:
            self.save_dir = os.path.join(params['exp_dir'], params['exp_name'])
        else:
            self.save_dir = save_dir
        if data_dir is None:
            self.data_dir = self.save_dir
        else:
            self.data_dir = data_dir
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        self._logger = get_logger(
            self.__class__.__name__,
            params['model']['logger'],
            os.path.join(self.save_dir, 'debug.txt'))
        tf.logging.set_verbosity(tf.logging.ERROR)
        self.random_seed = params['random_seed']
        for k, v in params['model'].items():
            setattr(self, k, v)

        self.dU = len(self.U_idxs())
        self.num_O = params['model'].get('num_O', 1)
        self.dO_im = len(self.O_im_idxs()) * self.num_O
        self.dO_vec = len(self.O_vec_idxs()) * self.num_O
        self.doutput = len(self.output_idxs())
        self.dtype = tf_utils.str_to_dtype(params["model"]["dtype"])
        self.dropout = params["model"]["action_graph"].get("dropout", None)
        
        if not hasattr(self, 'pct_coll'):
            self.train_replay_buffer = ReplayBuffer(
                int(5e5),
                self.T,
                self.dU,
                self.dO_im,
                self.dO_vec,
                self.doutput)

            self.val_replay_buffer = ReplayBuffer(
                int(5e5),
                self.T,
                self.dU,
                self.dO_im,
                self.dO_vec,
                self.doutput)
        else:
            self.train_replay_buffer = SplitReplayBuffer(
                int(5e5),
                self.T,
                self.dU,
                self.dO_im,
                self.dO_vec,
                self.doutput,
                self.pct_coll)

            self.val_replay_buffer = SplitReplayBuffer(
                int(5e5),
                self.T,
                self.dU,
                self.dO_im,
                self.dO_vec,
                self.doutput,
                self.pct_coll)
        
        self.threads = []
        self.graph = tf.Graph()
        os.environ["CUDA_VISIBLE_DEVICES"] = str(self.device)
        if params['probcoll']['asynchronous_training']:
            self.gpu_fraction = self.gpu_fraction / 2.0
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=self.gpu_fraction)
        config = tf.ConfigProto(
            gpu_options=gpu_options,
            log_device_placement=False,
            allow_soft_placement=True)
        with self.graph.as_default():
            tf.set_random_seed(self.random_seed)
            self.sess = tf.Session(config=config)
        code_file_exists = os.path.exists(self._code_file)
        if code_file_exists:
            self._logger.info('Creating OLD graph')
        else:
            self._logger.info('Creating NEW graph')
            shutil.copyfile(self._this_file, self._code_file)
        self.tf_debug = {}
        self._graph_setup()


    ############
    ### Data ###
    ############

    def _modify_sample(self, sample):
        """
        In case you want to pre-process the sample before adding it
        :return: Sample
        """
        return [sample]

    def _load_samples(self, npz_fnames):
        """
        :param npz_fnames: pkl files names containing Samples
        :return: no_coll_data, coll_data 
        """

        no_coll_data = {"U": [], "O_im": [],"O_vec": [], "output": [], "extended": []}
        coll_data =    {"U": [], "O_im": [], "O_vec": [],"output": [], "extended": []}

        random.shuffle(npz_fnames)
        for npz_fname in npz_fnames:
            ### load samples
            self._logger.debug('\tOpening {0}'.format(npz_fname))

            samples = Sample.load(npz_fname)
            # Shuffle the data so validation and training data is shuffled
            random.shuffle(samples)
            ### add to data
            for og_sample in samples:
                for sample in self._modify_sample(og_sample):
                    s_params = sample._meta_data
                    U = sample.get_U()[:, self.U_idxs(p=s_params)]
                    O_im = sample.get_O()[:, self.O_im_idxs(p=s_params)].astype(np.uint8)
                    O_vec = sample.get_O()[:, self.O_vec_idxs(p=s_params)]
                    output = sample.get_O()[:, self.output_idxs(p=s_params)].astype(np.uint8)
                    buffer_len = 0
                    if len(U) < 1 + buffer_len: # used to be self.T, but now we are extending
                        continue

                    for arr in (U, O_im, O_vec, output):
                        assert(np.all(np.isfinite(arr)))
                        assert(not np.any(np.isnan(arr)))

                    if output[-1, 0] == 1:
                        # For collision data extend collision by T-1 (and buffer)
                        extend_u = np.zeros((self.T - 1 - buffer_len, U.shape[1]))
                        U_coll = np.vstack((U[-self.T:], extend_u))
                        O_im_coll = np.vstack((O_im[-self.T:], np.tile([O_im[-1]], (self.T - 1 - buffer_len, 1))))
                        O_vec_coll = np.vstack((O_vec[-self.T:], np.tile([O_vec[-1]], (self.T - 1 - buffer_len, 1))))
                        output_coll = np.vstack((output[-self.T:], np.tile([output[-1]], (self.T - 1 - buffer_len, 1))))
                        coll_data["U"].append(U_coll)
                        coll_data["O_im"].append(O_im_coll)
                        coll_data["O_vec"].append(O_vec_coll)
                        coll_data["output"].append(output_coll)
                        coll_data["extended"].append(True)
                        # For noncollision data remove the collision
                        if len(U) - 1 >= self.T:
                            no_coll_data["U"].append(U[:-1])
                            no_coll_data["O_im"].append(O_im[:-1])
                            no_coll_data["O_vec"].append(O_vec[:-1])
                            no_coll_data["output"].append(output[:-1])
                            no_coll_data["extended"].append(False)
                    else:
                        extend_u = np.zeros((self.T - 1 - buffer_len, U.shape[1]))
                        U_no_coll = np.vstack((U, extend_u))
                        O_im_no_coll = np.vstack((O_im, np.tile([O_im[-1]], (self.T - 1 - buffer_len, 1))))
                        O_vec_no_coll = np.vstack((O_vec, np.tile([O_vec[-1]], (self.T - 1 - buffer_len, 1))))
                        output_no_coll = np.vstack((output, np.tile([output[-1]], (self.T - 1 - buffer_len, 1))))
                        no_coll_data["U"].append(U_no_coll)
                        no_coll_data["O_im"].append(O_im_no_coll)
                        no_coll_data["O_vec"].append(O_vec_no_coll)
                        no_coll_data["output"].append(output_no_coll)
                        no_coll_data["extended"].append(True)

        return no_coll_data, coll_data

    def _add_to_buffer(
            self,
            replay_buffer,
            partition,
            U_by_sample,
            O_im_by_sample,
            O_vec_by_sample,
            output_by_sample,
            extended_by_sample):

        if len(U_by_sample) > 0:
            for i, (U, O_im, O_vec, output, extended) in enumerate(zip(U_by_sample, O_im_by_sample, O_vec_by_sample, output_by_sample, extended_by_sample)):
                assert(len(U) >= self.T)
                for j in range(len(U) - self.T + 1):
                    feature_u = U[j:j+self.T]
                    if j < self.num_O - 1:
                        obs_im = O_im[:j].ravel()
                        feature_obs_im = np.concatenate([np.zeros(self.dO_im - obs_im.shape[0], dtype=np.uint8), obs_im])
                        obs_vec = O_vec[:j].ravel()
                        feature_obs_vec = np.concatenate([np.zeros(self.dO_vec - obs_vec.shape[0]), obs_vec])
                    else:
                        feature_obs_im = (O_im[j-self.num_O+1:j+1]).ravel()
                        feature_obs_vec = (O_vec[j-self.num_O+1:j+1]).ravel()
                    feature_output = output[j:j+self.T]
                    output_list = feature_output.ravel()
                    if extended:
                        length = min(self.T, len(U) - self.T + 1 - j)
                    else:
                        length = self.T
                    assert(length != 0)
                    feature_len = length
                    replay_buffer.add_data_point(feature_u, feature_obs_im, feature_obs_vec, feature_output, feature_len, partition) 

    def add_data(self, npz_fnames):
        no_coll_data, coll_data = self._load_samples(npz_fnames)
        no_coll_len = len(no_coll_data["U"])
        coll_len = len(coll_data["U"])
        tot_coll = sum([np.argmax(coll_data["output"][i]) + 1 for i in range(coll_len)])
        tot_no = sum([len(no_coll_data["output"][i]) - self.T + 1 for i in range(no_coll_len)])
        self._logger.info("Size of no collision data: {0}".format(tot_no))
        self._logger.info("Size of collision data: {0}".format(tot_coll))
        coll_val_index = int(coll_len * self.val_pct)
        if coll_len > 1:
            coll_val_index = max(coll_val_index, 1)
        no_coll_val_index = int(no_coll_len * self.val_pct)
        if no_coll_len > 1:
            no_coll_val_index = max(no_coll_val_index, 1)

        self._add_to_buffer(
            self.train_replay_buffer,
            1,
            no_coll_data["U"][no_coll_val_index:],
            no_coll_data["O_im"][no_coll_val_index:],
            no_coll_data["O_vec"][no_coll_val_index:],
            no_coll_data["output"][no_coll_val_index:],
            no_coll_data["extended"][no_coll_val_index:])
        
        self._add_to_buffer(
            self.train_replay_buffer,
            0,
            coll_data["U"][coll_val_index:],
            coll_data["O_im"][coll_val_index:],
            coll_data["O_vec"][coll_val_index:],
            coll_data["output"][coll_val_index:],
            coll_data["extended"][coll_val_index:])
        
        self._add_to_buffer(
            self.val_replay_buffer,
            1,
            no_coll_data["U"][:no_coll_val_index],
            no_coll_data["O_im"][:no_coll_val_index],
            no_coll_data["O_vec"][:no_coll_val_index],
            no_coll_data["output"][:no_coll_val_index],
            no_coll_data["extended"][:no_coll_val_index])
        
        self._add_to_buffer(
            self.val_replay_buffer,
            0,
            coll_data["U"][:coll_val_index],
            coll_data["O_im"][:coll_val_index],
            coll_data["O_vec"][:coll_val_index],
            coll_data["output"][:coll_val_index],
            coll_data["extended"][:coll_val_index])
    
    #############
    ### Graph ###
    #############
   
    def _graph_inputs_outputs_from_placeholders(self):
        with tf.variable_scope('feed_input'):
            b = self.num_bootstrap
            u_ph = tf.placeholder(self.dtype, [b, self.batch_size, self.T, self.dU], name='U_placeholder')
            o_im_ph = tf.placeholder(tf.uint8, [b, self.batch_size, self.dO_im], name='O_im_placeholder')
            o_vec_ph = tf.placeholder(self.dtype, [b, self.batch_size, self.dO_vec], name='O_vec_placeholder')
            output_ph = tf.placeholder(tf.uint8, [b, self.batch_size, self.T, self.doutput], name='Output_placeholder')
            length_ph = tf.placeholder(tf.int32, [b, self.batch_size], name='length_placeholder')
        return u_ph, o_im_ph, o_vec_ph, output_ph, length_ph 

    def _get_bootstrap_batch_feed_dict(
            self,
            replay_buffer,
            u_ph,
            o_im_ph,
            o_vec_ph,
            output_ph,
            length_ph):
        # batch_size = self.batch_size if tf.train.global_step(self.sess, self.global_step) > 0 else 5096 # TODO: can't do b/c batch size fixed
        batch_size = self.batch_size
        list_data = [replay_buffer.sample(batch_size) for _ in range(self.num_bootstrap)]
        u_b, o_im_b, o_vec_b, output_b, length_b = zip(*list_data) 
        feed_dict = {
                u_ph: u_b,
                o_im_ph: o_im_b,
                o_vec_ph: o_vec_b,
                output_ph: output_b,
                length_ph: length_b
            }
        return feed_dict

    def _graph_setup(self):
        """ Only call once """
        with self.graph.as_default():
            self.d_train = dict()
            self.d_val = dict()
            self.d_eval = dict()
            self.global_step = tf.Variable(0, trainable=False, name='global_step')

            ### prepare for training
            for i, (name, d) in enumerate((('train', self.d_train), ('val', self.d_val))):
                d['U_inputs'], d['O_im_inputs'], d['O_vec_inputs'], d['outputs'], d['len'], \
                    = self._graph_inputs_outputs_from_placeholders()
                d['output_mats'] = self._graph_inference(
                    name,
                    d['U_inputs'],
                    d['O_im_inputs'],
                    d['O_vec_inputs'],
                    reuse=i>0,
                    tf_debug=self.tf_debug)
                d['bootstraps_cost'], d['reg_cost'], d['cost'], d['cross_entropy'], d['err'], d['err_coll'], d['err_nocoll'], d['num_coll'], d['num_nocoll'] = \
                    self._graph_cost(name, d['output_mats'], d['outputs'], d['len'], reg=self.reg)
            ### optimizer
            self.d_train['optimizer'], self.d_train['grads'], self.d_train['optimizer_vars'] = \
                self._graph_optimize(self.d_train['bootstraps_cost'], self.d_train['reg_cost'])

            ### prepare for eval
            self.d_eval['U_inputs'], self.d_eval['O_im_input'], self.d_eval['O_vec_input'] = self._graph_inputs_from_placeholders()
            self.d_eval['output_pred_mean'], self.d_eval['output_pred_std'], self.d_eval['output_mat_mean'], \
            self.d_eval['output_mat_std'] = \
                self.graph_eval_inference(
                    self.d_eval['U_inputs'],
                    self.d_eval['O_im_input'],
                    self.d_eval['O_vec_input'],
                    reuse=True)

            ### initialize
            self._initializer = [tf.local_variables_initializer(), tf.global_variables_initializer()]
            self.graph_init_vars()

            # Set logs writer into folder /tmp/tensorflow_logs
            merged = tf.summary.merge_all()
            writer = tf.summary.FileWriter(
                os.path.join('/tmp', params['exp_name']),
                graph=self.sess.graph)
            self.saver = tf.train.Saver(max_to_keep=None)

    ################
    ### Training ###
    ################
    
    def train(self, reset=False, **kwargs):
        if not self.train_replay_buffer.can_sample():
            self._logger.info('Training skipped due to lack of data')
        else:
            self.graph.as_default()
            new_model_file, model_num  = self._next_model_file()
            step = self.sess.run(self.global_step)
            reset = reset or (step >= self.reset_after_step)
            if reset:
                self._logger.info('Resetting model')
                self.graph_init_vars()

            ### create plotter
            plotter = MLPlotter(
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

            ### train
            train_values = defaultdict(list)
            train_nums = defaultdict(float)

            step = 0
            epoch_start = save_start = train_start = time.time()
            if reset:
                itr_steps = self.reset_steps
            else:
                itr_steps = self.steps
            # TODO add check for early stopping
            while step < itr_steps:

                ### validation
                if (step != 0 and self.val_replay_buffer.can_sample() and \
                        (step % int(self.val_freq * itr_steps)) == 0):
                    val_values = defaultdict(list)
                    val_nums = defaultdict(float)
                    val_steps = 0
                    self._logger.info('Computing validation...')
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

                    plotter.add_val('err', np.mean(val_values['err']))
                    plotter.add_val('err_coll', np.mean(val_values['err_coll']))
                    plotter.add_val('err_nocoll', np.mean(val_values['err_nocoll']))
                    plotter.add_val('cost', np.mean(val_values['cost']))
                    plotter.add_val('cross_entropy', np.mean(val_values['cross_entropy']))

                    self._logger.info(
                        'error: {0:5.2f}%, error coll: {1:5.2f}%, error nocoll: {2:5.2f}%, pct coll: {3:4.1f}%, cost: {4:4.2f}, ce: {5:4.2f} ({6:.2f} samples / s)'.format(
                            100 * np.mean(val_values['err']),
                            100 * np.mean(val_values['err_coll']),
                            100 * np.mean(val_values['err_nocoll']),
                            100 * val_nums['coll'] / (val_nums['coll'] + val_nums['nocoll']),
                            np.mean(val_values['cost']),
                            np.mean(val_values['cross_entropy']),
                            float((self.val_freq * itr_steps + self.val_steps ) * self.batch_size) /  (time.time() - epoch_start)))

                    epoch_start = time.time()

                    ### save model
                    if not reset:
                        self.save(new_model_file)

                ### train
                feed_dict = self._get_bootstrap_batch_feed_dict(
                    self.train_replay_buffer,
                    self.d_train['U_inputs'],
                    self.d_train['O_im_inputs'],
                    self.d_train['O_vec_inputs'],
                    self.d_train['outputs'],
                    self.d_train['len'])
                _, train_cost, train_cross_entropy, \
                train_err, train_err_coll, train_err_nocoll, \
                train_coll, train_nocoll = self.sess.run(
                    [
                        self.d_train['optimizer'],
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

                # Print an overview fairly often.
                if step % self.display_steps == 0 and step > 0:
                    plotter.add_train('err', step * self.batch_size, np.mean(train_values['err']))
                    if len(train_values['err_coll']) > 0:
                        plotter.add_train('err_coll', step * self.batch_size, np.mean(train_values['err_coll']))
                    if len(train_values['err_nocoll']) > 0:
                        plotter.add_train('err_nocoll', step * self.batch_size, np.mean(train_values['err_nocoll']))
                    plotter.add_train('cost', step * self.batch_size, np.mean(train_values['cost']))
                    plotter.add_train('cross_entropy', step * self.batch_size, np.mean(train_values['cross_entropy']))

                    self._logger.info('Training step pct: {0:.1f}'.format(100 * step / float(itr_steps)))
                    self._logger.info('error: {0:5.2f}%, error coll: {1:5.2f}%, error nocoll: {2:5.2f}%, pct coll: {3:4.1f}%, cost: {4:4.2f}, ce: {5:4.2f}'.format(
                        100 * np.mean(train_values['err']),
                        100 * np.mean(train_values['err_coll']),
                        100 * np.mean(train_values['err_nocoll']),
                        100 * train_nums['coll'] / (train_nums['coll'] + train_nums['nocoll']),
                        np.mean(train_values['cost']),
                        np.mean(train_values['cross_entropy'])))

                    train_values = defaultdict(list)
                    train_nums = defaultdict(float)

                if time.time() - save_start > 60.:
                    plotter.save(self._plots_dir, suffix=str(model_num))
                    save_start = time.time()

                step += 1

            self._logger.info('Training Speed: {0:.2f} samples / s'.format(
                float((itr_steps + self.val_steps / self.val_freq) * self.batch_size) /  (time.time() - train_start)))
            
            # Logs the number of times files were accessed
            self.save(new_model_file)
            plotter.save(self._plots_dir, suffix=str(model_num))
            plotter.close()
