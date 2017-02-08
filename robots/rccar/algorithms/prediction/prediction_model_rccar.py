import os, sys
import random

import numpy as np
import tensorflow as tf

from general.algorithm.probcoll_model import ProbcollModel

from config import params

class ProbcollModelRCcar(ProbcollModel):

    ####################
    ### Initializing ###
    ####################

    def __init__(self, read_only=False, finalize=True):
        dist_eps = params['O']['collision']['buffer']
        ProbcollModel.__init__(self, dist_eps, read_only=read_only, finalize=finalize)

    #############
    ### Files ###
    #############

    @property
    def _this_file(self):
        return os.path.abspath(__file__.replace('.pyc', '.py'))

    ############
    ### Data ###
    ############

    def _modify_sample(self, sample):
        """
        In case you want to pre-process the sample before adding it
        :return: Sample
        """
        # return ProbcollModel._modify_sample(self, sample)

        ### move collision observation one time step earlier
        if sample.get_O(t=-1, sub_obs='collision'):
            try:
                T_sample = len(sample.get_X())
                t_backward = 0
                new_sample = sample.match(slice(0, T_sample - t_backward))
                new_sample.set_O([1.], t=-1, sub_obs='collision')
                # t_forward = 2
                # new_sample = sample.match(slice(t_forward, T_sample))
                # new_sample.set_U(sample.get_U(t=slice(0, T_sample-t_forward)), t=slice(0, T_sample-t_forward))
                # new_sample.set_X(sample.get_X(t=slice(0, T_sample-t_forward)), t=slice(0, T_sample-t_forward))
                sample = new_sample
            except:
                self.logger.debug('Modify sample exception')

        return [sample]

    def _balance_data(self, start_idxs_by_sample, X_by_sample, U_by_sample, O_by_sample, output_by_sample):
        bd_param = params['prediction']['model']['balance']

        ### split idxs into those with collisions and those without
        samples_coll, samples_no_coll = [], []
        idxs_coll = []
        idxs_no_coll = []
        for i, (start_idxs, output) in enumerate(zip(start_idxs_by_sample, output_by_sample)):
            is_coll = False
            for start_idx in start_idxs:
                idxs = range(start_idx, start_idx + self.T)
                if np.sum(output[idxs]) >= 1:
                    # was a collision
                    is_coll = True
                    idxs_coll.append(idxs)
                else:
                    idxs_no_coll.append(idxs)

            if is_coll:
                samples_coll.append(i)
            else:
                samples_no_coll.append(i)

        self.logger.info('Percentage samples ending in collision: {0:.2f}'.format(
            len(samples_coll) / float(len(samples_coll) + len(samples_no_coll))))
        self.logger.info('Percentage with P(coll < {0:.2f}) = {1:f}'.format(self.dist_eps,
                                                                            len(idxs_coll) / float(
                                                                                len(idxs_coll) + len(idxs_no_coll))))

        assert(len(start_idxs_by_sample) >= 4)
        num_val = max(2, int(len(start_idxs_by_sample) * self.val_pct))

        if bd_param['type'] == 'collision':# and len(samples_coll) >= 2 and len(samples_no_coll) >= 2:
            resample_param = bd_param['collision']
            pct_coll = resample_param['pct_coll']
            self.logger.info('Rebalancing data to be {0:.2f}% collision'.format(100 * pct_coll))

            ### split by traj
            ### ensure both training and test have at least one collision and no collision
            # val_has_coll = min(samples_coll) < num_val
            # val_has_nocoll = min(samples_no_coll) < num_val
            # train_has_coll = max(samples_coll) >= num_val
            # train_has_nocoll = max(samples_no_coll) >= num_val
            # assert(val_has_coll or val_has_nocoll)
            # assert(train_has_coll or train_has_nocoll)
            # swap_val_i = swap_val_j = 0
            # if not val_has_coll: swap_val_j = min(samples_coll)
            # if not val_has_nocoll: swap_val_j = min(samples_no_coll)
            # swap_train_i = swap_train_j = len(start_idxs_by_sample) - 1
            # if not train_has_coll: swap_train_j = max([j for j in samples_coll if j != swap_val_j])
            # if not train_has_nocoll: swap_train_j = max([j for j in samples_no_coll if j != swap_val_j])
            # assert(swap_val_j != swap_train_j)
            # for by_sample in (start_idxs_by_sample, X_by_sample, U_by_sample, O_by_sample, output_by_sample):
            #     by_sample[swap_val_i], by_sample[swap_val_j] = by_sample[swap_val_j], by_sample[swap_val_i]
            #     by_sample[swap_train_i], by_sample[swap_train_j] = by_sample[swap_train_j], by_sample[swap_train_i]

            start_idxs_by_val_sample = start_idxs_by_sample[:num_val]
            output_by_val_sample = output_by_sample[:num_val]
            start_idxs_by_train_sample = start_idxs_by_sample[num_val:]
            output_by_train_sample = output_by_sample[num_val:]

            ### separate by collision / no collision
            def separate_collision(start_idxs_by_sample, output_by_sample):
                coll_start_idxs_by_sample = []
                nocoll_start_idxs_by_sample = []
                for start_idxs, output in zip(start_idxs_by_sample, output_by_sample):
                    coll_start_idxs = []
                    nocoll_start_idxs = []
                    for start_idx in start_idxs:
                        idxs = range(start_idx, start_idx + self.T)
                        if np.sum(output[idxs]) >= 1:
                            coll_start_idxs.append(start_idx)
                        else:
                            nocoll_start_idxs.append(start_idx)
                    coll_start_idxs_by_sample.append(coll_start_idxs)
                    nocoll_start_idxs_by_sample.append(nocoll_start_idxs)

                return coll_start_idxs_by_sample, nocoll_start_idxs_by_sample

            coll_start_idxs_by_train_sample, nocoll_start_idxs_by_train_sample = \
                separate_collision(start_idxs_by_train_sample, output_by_train_sample)
            coll_start_idxs_by_val_sample, nocoll_start_idxs_by_val_sample = \
                separate_collision(start_idxs_by_val_sample, output_by_val_sample)

            assert(len(coll_start_idxs_by_train_sample) > 0)
            assert(len(nocoll_start_idxs_by_train_sample) > 0)
            assert(len(coll_start_idxs_by_val_sample) > 0)
            assert(len(nocoll_start_idxs_by_val_sample) > 0)

            ### idxs to resample from (sample, start_idx)
            def resample_idxs(start_idxs_by_sample):
                return [(i, j) for i in xrange(len(start_idxs_by_sample))
                        for j in start_idxs_by_sample[i]]

            coll_train_resample_idxs = resample_idxs(coll_start_idxs_by_train_sample)
            nocoll_train_resample_idxs = resample_idxs(nocoll_start_idxs_by_train_sample)
            coll_val_resample_idxs = resample_idxs(coll_start_idxs_by_val_sample)
            nocoll_val_resample_idxs = resample_idxs(nocoll_start_idxs_by_val_sample)

            ### do resampling
            def resample(coll_resample_idxs, nocoll_resample_idxs, num_samples):
                num_coll = len(coll_resample_idxs)
                num_nocoll = len(nocoll_resample_idxs)

                N = num_coll + num_nocoll

                new_num_coll = int(pct_coll * N)
                new_num_nocoll = N - new_num_coll

                ### [# train/val samples, # bootstrap, start idxs]
                bootstrap_start_idxs_by_sample = [[[] for _ in xrange(self.num_bootstrap)] for _ in xrange(num_samples)]
                for b in xrange(self.num_bootstrap):
                    for _ in xrange(new_num_coll):
                        sample_idx, start_idx = random.choice(coll_resample_idxs)
                        bootstrap_start_idxs_by_sample[sample_idx][b].append(start_idx)
                    for _ in xrange(new_num_nocoll):
                        sample_idx, start_idx = random.choice(nocoll_resample_idxs)
                        bootstrap_start_idxs_by_sample[sample_idx][b].append(start_idx)

                return bootstrap_start_idxs_by_sample

            bootstrap_start_idxs_by_train_sample = resample(coll_train_resample_idxs, nocoll_train_resample_idxs,
                                                            len(start_idxs_by_train_sample))
            bootstrap_start_idxs_by_val_sample = resample(coll_val_resample_idxs, nocoll_val_resample_idxs,
                                                          len(start_idxs_by_val_sample))

            return bootstrap_start_idxs_by_train_sample, X_by_sample[num_val:], U_by_sample[num_val:], O_by_sample[num_val:], output_by_sample[num_val:], \
                   bootstrap_start_idxs_by_val_sample, X_by_sample[:num_val], U_by_sample[:num_val], O_by_sample[:num_val], output_by_sample[:num_val]

        else:
            ### default no balancing
            self.logger.info('Not rebalancing data')
            return ProbcollModel._balance_data(self, start_idxs_by_sample, X_by_sample, U_by_sample, O_by_sample, output_by_sample)

    #############
    ### Graph ###
    #############

    def _get_old_graph_inference(self, graph_type='fc'):
        self.logger.info('Graph type: {0}'.format(graph_type))
        sys.path.append(os.path.dirname(self._code_file))
        exec('from {0} import {1} as OldProbcollModel'.format(
            os.path.basename(self._code_file).split('.')[0], 'ProbcollModelRCcar'))

        if graph_type == 'fc':
            return OldProbcollModel._graph_inference_fc
        else:
            raise Exception('graph_type {0} is not valid'.format(graph_type))

    @staticmethod
    def _graph_inference_fc(name, T, bootstrap_X_inputs, bootstrap_U_inputs, bootstrap_O_inputs,
                            X_mean, X_orth, U_mean, U_orth, O_mean, O_orth, dropout, meta_data,
                            reuse=False, random_seed=None, finalize=True, tf_debug={}):
        assert(name == 'train' or name == 'val' or name == 'eval')
        num_bootstrap = len(bootstrap_X_inputs)

        bootstrap_output_mats = []
        bootstrap_output_preds = []
        dropout_placeholders = [] if name == 'eval' else None

        with tf.name_scope(name + '_inference'):
            tf.set_random_seed(random_seed)

            for b in xrange(num_bootstrap):
                ### inputs
                x_input_b = bootstrap_X_inputs[b]
                u_input_b = bootstrap_U_inputs[b]
                o_input_b = bootstrap_O_inputs[b]

                dX = x_input_b.get_shape()[2].value
                dU = u_input_b.get_shape()[2].value
                dO = o_input_b.get_shape()[1].value
                T = x_input_b.get_shape()[1].value
                # batch_size = x_input_b.get_shape()[0].value
                batch_size = tf.shape(x_input_b)[0]
                n_output = 1

                ### concatenate inputs
                with tf.name_scope('inputs_b{0}'.format(b)):
                    concat_list = []
                    # import IPython; IPython.embed()
                    if dO > 0:
                        concat_list.append(tf.matmul(o_input_b - O_mean, O_orth))
                    if dX > 0:
                        X_orth_batch = tf.tile(tf.expand_dims(X_orth, 0),
                                               [batch_size, 1, 1])
                        x_input_b = tf.batch_matmaul(x_input_b - X_mean, X_orth_batch)
                        x_input_flat_b = tf.reshape(x_input_b, [1, T * dX])
                        concat_list.append(x_input_flat_b)
                    if dU > 0:
                        U_orth_batch= tf.tile(tf.expand_dims(U_orth, 0),
                                              [batch_size, 1, 1])
                        u_input_b = tf.batch_matmul(u_input_b - U_mean, U_orth_batch)
                        u_input_flat_b = tf.reshape(u_input_b, [-1, T * dU])
                        concat_list.append(u_input_flat_b)
                    input_layer = tf.concat(1, concat_list)

                n_input = input_layer.get_shape()[-1].value

                ### weights
                with tf.variable_scope('inference_vars_{0}'.format(b), reuse=reuse):
                    weights_b = [
                        tf.get_variable('w_hidden_0_b{0}'.format(b), [n_input, 40], initializer=tf.contrib.layers.xavier_initializer()),
                        tf.get_variable('w_hidden_1_b{0}'.format(b), [40, 40], initializer=tf.contrib.layers.xavier_initializer()),
                        tf.get_variable('w_output_b{0}'.format(b), [40, n_output], initializer=tf.contrib.layers.xavier_initializer()),
                    ]
                    biases_b = [
                        tf.get_variable('b_hidden_0_b{0}'.format(b), [40], initializer=tf.constant_initializer(0.)),
                        tf.get_variable('b_hidden_1_b{0}'.format(b), [40], initializer=tf.constant_initializer(0.)),
                        tf.get_variable('b_output_b{0}'.format(b), [n_output], initializer=tf.constant_initializer(0.)),
                    ]

                ### weight decays
                for v in weights_b + biases_b:
                    tf.add_to_collection('weight_decays', 0.5 * tf.reduce_mean(v ** 2))

                ### fully connected relus
                layer = input_layer
                for i, (weight, bias) in enumerate(zip(weights_b[:-1], biases_b[:-1])):
                    with tf.name_scope('hidden_{0}_b{1}'.format(i, b)):
                        layer = tf.nn.relu(tf.add(tf.matmul(layer, weight), bias))
                        if dropout is not None:
                            assert(type(dropout) is float and 0 <= dropout and dropout <= 1.0)
                            if name == 'eval':
                                dp = tf.placeholder('float', [None, layer.get_shape()[1].value])
                                layer = tf.mul(layer, dp)
                                dropout_placeholders.append(dp)
                            else:
                                layer = tf.nn.dropout(layer, dropout)

                with tf.name_scope('scope_output_pred_b{0}'.format(b)):
                    ### sigmoid
                    # layer = tf.nn.l2_normalize(layer, 1) # TODO
                    output_mat_b = tf.add(tf.matmul(layer, weights_b[-1]), biases_b[-1])
                    output_pred_b = tf.sigmoid(output_mat_b, name='output_pred_b{0}'.format(b))

                bootstrap_output_mats.append(output_mat_b)
                bootstrap_output_preds.append(output_pred_b)

            ### combination of all the bootstraps
            with tf.name_scope('combine_bootstraps'):
                output_pred_mean = (1 / float(num_bootstrap)) * tf.add_n(bootstrap_output_preds, name='output_pred_mean')
                std_normalize = (1 / float(num_bootstrap - 1)) if num_bootstrap > 1 else 1
                output_pred_std = tf.sqrt(std_normalize * tf.add_n(
                    [tf.square(tf.sub(output_pred_b, output_pred_mean)) for output_pred_b in bootstrap_output_preds]))

                output_mat_mean = (1 / float(num_bootstrap)) * tf.add_n(bootstrap_output_mats, name='output_mat_mean')
                output_mat_std = tf.sqrt(std_normalize * tf.add_n(
                    [tf.square(tf.sub(output_mat_b, output_mat_mean)) for output_mat_b in bootstrap_output_mats]))

        return output_pred_mean, output_pred_std, output_mat_mean, output_mat_std, bootstrap_output_mats, dropout_placeholders

    ################
    ### Training ###
    ################

    def _create_input(self, X, U, O):
        return ProbcollModel._create_input(self, X, U, O)

    def _create_output(self, output):
        return ProbcollModel._create_output(self, output).astype(int)

