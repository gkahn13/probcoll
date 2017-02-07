import os, sys
import random

import numpy as np
import tensorflow as tf

from general.algorithms.prediction.prediction_model import PredictionModel

from general.policy.tf.rnn import dynamic_rnn as dynamic_rnn_GK
from general.policy.tf.rnn_cell import BasicRNNCell as BasicRNNCellDropout
from general.policy.tf.rnn_cell import BasicLSTMCell as BasicLSTMCellDropout

from config import params

class PredictionModelPointquad(PredictionModel):

    ####################
    ### Initializing ###
    ####################

    def __init__(self, read_only=False, finalize=True):
        dist_eps = params['O']['collision']['buffer']
        PredictionModel.__init__(self, dist_eps, read_only=read_only, finalize=finalize)

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
        # return PredictionModel._modify_sample(self, sample)

        ### move collision observation one time step earlier
        if sample.get_O(t=-1, sub_obs='collision'):
            try:
                T_sample = len(sample.get_X())
                new_sample = sample.match(slice(0, T_sample-1))
                new_sample.set_O([1.], t=-1, sub_obs='collision')
                sample = new_sample
            except:
                self.logger.debug('Modify sample exception')

        return [sample]

        ### subsample
        # T_sample = len(sample.get_O())
        # T_back = min(T_sample-2, self.propagate_collision)
        # for t in xrange(T_sample-2, T_sample-2-T_back, -1):
        #     sample.set_O(sample.get_O(t=T_sample-1, sub_obs='collision'), t=t, sub_obs='collision')

        ### subsample by X and generate X new 1/X samples
        # new_samples = []
        # for t_back in xrange(self.subsample):
        #     try:
        #         X_sub = np.flipud(sample.get_X()[-1-t_back::-self.subsample])
        #         U_sub = np.flipud(sample.get_U()[-1-t_back::-self.subsample])
        #         O_sub = np.flipud(sample.get_O()[-1-t_back::-self.subsample])
        #
        #         T_sub = len(X_sub)
        #         sample_sub = Sample(T=T_sub, meta_data=sample._meta_data)
        #         sample_sub.set_X(X_sub, t=slice(0, T_sub))
        #         sample_sub.set_U(U_sub, t=slice(0, T_sub))
        #         sample_sub.set_O(O_sub, t=slice(0, T_sub))
        #         sample_sub.set_O(sample.get_O(t=-1, sub_obs='collision'), t=-1, sub_obs='collision')
        #
        #         new_samples.append(sample_sub)
        #     except:
        #         pass
        #
        # return new_samples

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

        if bd_param['type'] == 'collision' and len(samples_coll) >= 2 and len(samples_no_coll) >= 2:
            resample_param = bd_param['collision']
            pct_coll = resample_param['pct_coll']
            self.logger.info('Rebalancing data to be {0:.2f}% collision'.format(100 * pct_coll))

            ### split by traj
            ### ensure both training and test have at least one collision and no collision
            val_has_coll = min(samples_coll) < num_val
            val_has_nocoll = min(samples_no_coll) < num_val
            train_has_coll = max(samples_coll) >= num_val
            train_has_nocoll = max(samples_no_coll) >= num_val
            assert(val_has_coll or val_has_nocoll)
            assert(train_has_coll or train_has_nocoll)
            swap_val_i = swap_val_j = 0
            if not val_has_coll: swap_val_j = min(samples_coll)
            if not val_has_nocoll: swap_val_j = min(samples_no_coll)
            swap_train_i = swap_train_j = len(start_idxs_by_sample) - 1
            if not train_has_coll: swap_train_j = max([j for j in samples_coll if j != swap_val_j])
            if not train_has_nocoll: swap_train_j = max([j for j in samples_no_coll if j != swap_val_j])
            assert(swap_val_j != swap_train_j)
            for by_sample in (start_idxs_by_sample, X_by_sample, U_by_sample, O_by_sample, output_by_sample):
                by_sample[swap_val_i], by_sample[swap_val_j] = by_sample[swap_val_j], by_sample[swap_val_i]
                by_sample[swap_train_i], by_sample[swap_train_j] = by_sample[swap_train_j], by_sample[swap_train_i]

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
            return PredictionModel._balance_data(self, start_idxs_by_sample, X_by_sample, U_by_sample, O_by_sample, output_by_sample)

    #############
    ### Graph ###
    #############

    def _get_old_graph_inference(self, graph_type='fc'):
        self.logger.info('Graph type: {0}'.format(graph_type))
        sys.path.append(os.path.dirname(self._code_file))
        exec('from {0} import {1} as OldPredictionModel'.format(
            os.path.basename(self._code_file).split('.')[0], 'PredictionModelPointquad'))

        if graph_type == 'fc':
            return OldPredictionModel._graph_inference_fc
        if graph_type == 'fc_sparse':
            return OldPredictionModel._graph_inference_fc_sparse
        elif graph_type == 'cnn':
            return OldPredictionModel._graph_inference_cnn
        elif graph_type == 'rnn':
            return OldPredictionModel._graph_inference_rnn
        else:
            raise Exception('graph_type {0} is not valid'.format(graph_type))

    @staticmethod
    def _graph_inference_fc_sparse(name, T, bootstrap_X_inputs, bootstrap_U_inputs, bootstrap_O_inputs,
                                   X_mean, U_mean, O_mean, dropout, meta_data,
                                   reuse=False, random_seed=None, finalize=True, tf_debug={}):
        assert (name == 'train' or name == 'val' or name == 'eval')
        num_bootstrap = len(bootstrap_X_inputs)

        dropout_placeholders = [] if name == 'eval' else None

        with tf.name_scope(name + '_inference'):
            tf.set_random_seed(random_seed)

            input_layers = []
            weights = []
            biases = []
            with tf.name_scope('create_inputs_and_variables'):
                for b in xrange(num_bootstrap):
                    ### inputs
                    x_input_b = bootstrap_X_inputs[b]
                    u_input_b = bootstrap_U_inputs[b]
                    o_input_b = bootstrap_O_inputs[b]

                    dX = x_input_b.get_shape()[2].value
                    dU = u_input_b.get_shape()[2].value
                    dO = o_input_b.get_shape()[1].value
                    T = x_input_b.get_shape()[1].value
                    n_output = 1

                    ### concatenate inputs
                    with tf.name_scope('inputs_b{0}'.format(b)):
                        concat_list = []
                        if dO > 0:
                            with tf.variable_scope('means', reuse=reuse or b > 0):
                                tf_O_mean = tf.get_variable('O_mean', shape=[1, len(O_mean)], trainable=False,
                                                            dtype=tf.float32,
                                                            initializer=tf.constant_initializer(
                                                                O_mean.reshape((1, len(O_mean)))))

                            concat_list.append(o_input_b - tf_O_mean)
                        if dX > 0:
                            with tf.variable_scope('means', reuse=reuse or b > 0):
                                tf_X_mean = tf.get_variable('X_mean', shape=[1, len(X_mean)], trainable=False,
                                                            dtype=tf.float32,
                                                            initializer=tf.constant_initializer(
                                                                X_mean.reshape((1, len(X_mean)))))

                            x_input_flat_b = tf.reshape(x_input_b, [1, T * dX])
                            x_input_flat_b -= tf.tile(tf_X_mean, [1, T])  # subtract mean
                            concat_list.append(x_input_flat_b)
                        if dU > 0:
                            with tf.variable_scope('means', reuse=reuse or b > 0):
                                tf_U_mean = tf.get_variable('U_mean', shape=[1, len(U_mean)], trainable=False,
                                                            dtype=tf.float32,
                                                            initializer=tf.constant_initializer(
                                                                U_mean.reshape((1, len(U_mean)))))

                            u_input_flat_b = tf.reshape(u_input_b, [-1, T * dU])
                            u_input_flat_b -= tf.tile(tf_U_mean, [1, T])  # subtract mean
                            concat_list.append(u_input_flat_b)
                        input_layer = tf.concat(1, concat_list)

                    n_input = input_layer.get_shape()[-1].value

                    ### weights
                    with tf.variable_scope('inference_vars_{0}'.format(b), reuse=reuse):
                        weights_b = [
                            tf.get_variable('w_hidden_0_b{0}'.format(b), [n_input, 40],
                                            initializer=tf.contrib.layers.xavier_initializer()),
                            tf.get_variable('w_hidden_1_b{0}'.format(b), [40, 40],
                                            initializer=tf.contrib.layers.xavier_initializer()),
                            tf.get_variable('w_output_b{0}'.format(b), [40, n_output],
                                            initializer=tf.contrib.layers.xavier_initializer()),
                        ]
                        biases_b = [
                            tf.get_variable('b_hidden_0_b{0}'.format(b), [40], initializer=tf.constant_initializer(0.)),
                            tf.get_variable('b_hidden_1_b{0}'.format(b), [40], initializer=tf.constant_initializer(0.)),
                            tf.get_variable('b_output_b{0}'.format(b), [n_output], initializer=tf.constant_initializer(0.)),
                        ]

                    ### weight decays
                    for v in weights_b + biases_b:
                        tf.add_to_collection('weight_decays', 0.5 * tf.reduce_mean(v ** 2))

                    input_layers.append(input_layer)
                    weights.append(weights_b)
                    biases.append(biases_b)

            ### combine input_layers/weights/biases
            combined_input_layers = tf.concat(1, input_layers)
            combined_biases = [tf.concat(0, biases_i) for biases_i in np.array(biases).T.tolist()]
            combined_weights = []
            for i in xrange(len(weights[0])):
                weights_i = [weights_b[i] for weights_b in weights]
                shapes_i = [w_i.get_shape().as_list() for w_i in weights_i]
                sparse_shape = (sum(zip(*shapes_i)[0]), sum(zip(*shapes_i)[1]))

                start_dim0, start_dim1 = 0, 0
                sparse_indices = []
                for shape in shapes_i:
                    sparse_indices += (np.indices(shape).reshape((2, np.prod(shape))).T + (start_dim0, start_dim1)).tolist()
                    start_dim0 += shape[0]
                    start_dim1 += shape[1]

                sparse_values = tf.concat(0, [tf.reshape(w_i, [-1]) for w_i in weights_i])

                combined_weights.append(tf.SparseTensor(indices=sparse_indices, values=sparse_values, shape=sparse_shape))

            ### fully connected relus
            def sparse_matmul(layer, weight):
                weight_shape = weight.get_shape().as_list()
                weight_T = tf.sparse_transpose(weight)
                weight_T = tf.SparseTensor(indices=weight_T.indices, values=weight_T.values,
                                           shape=(weight_shape[1], weight_shape[0]))
                return tf.transpose(tf.sparse_tensor_dense_matmul(weight_T, tf.transpose(layer)))

            with tf.name_scope('network'):
                layer = combined_input_layers
                for i, (weight, bias) in enumerate(zip(combined_weights[:-1], combined_biases[:-1])):
                    with tf.name_scope('hidden_{0}'.format(i)):
                        layer = tf.nn.relu(tf.add(sparse_matmul(layer, weight), bias))
                        if dropout is not None:
                            assert (type(dropout) is float and 0 <= dropout and dropout <= 1.0)
                            if name == 'eval':
                                dp = tf.placeholder('float', [None, layer.get_shape()[1].value])
                                layer = tf.mul(layer, dp)
                                dropout_placeholders.append(dp)
                            else:
                                layer = tf.nn.dropout(layer, dropout)

            with tf.name_scope('scope_output_pred'):
                ### sigmoid
                output_mat = tf.add(sparse_matmul(layer, combined_weights[-1]), combined_biases[-1])
                output_pred = tf.sigmoid(output_mat, name='output_pred')

            bootstrap_output_mats = tf.split(1, num_bootstrap, output_mat)
            bootstrap_output_preds = tf.split(1, num_bootstrap, output_pred)

            ### combination of all the bootstraps
            with tf.name_scope('combine_bootstraps'):
                output_pred_mean = (1 / float(num_bootstrap)) * tf.add_n(bootstrap_output_preds,
                                                                         name='output_pred_mean')
                std_normalize = (1 / float(num_bootstrap - 1)) if num_bootstrap > 1 else 1
                output_pred_std = tf.sqrt(std_normalize * tf.add_n(
                    [tf.square(tf.sub(output_pred_b, output_pred_mean)) for output_pred_b in bootstrap_output_preds]))

                output_mat_mean = (1 / float(num_bootstrap)) * tf.add_n(bootstrap_output_mats, name='output_mat_mean')
                output_mat_std = tf.sqrt(std_normalize * tf.add_n(
                    [tf.square(tf.sub(output_mat_b, output_mat_mean)) for output_mat_b in bootstrap_output_mats]))

        return output_pred_mean, output_pred_std, output_mat_mean, output_mat_std, bootstrap_output_mats, dropout_placeholders

    @staticmethod
    def _graph_inference_fc(name, T, bootstrap_X_inputs, bootstrap_U_inputs, bootstrap_O_inputs,
                            X_mean, U_mean, O_mean, dropout, meta_data,
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
                n_output = 1

                ### concatenate inputs
                with tf.name_scope('inputs_b{0}'.format(b)):
                    concat_list = []
                    if dO > 0:
                        concat_list.append(o_input_b - O_mean)
                    if dX > 0:
                        x_input_flat_b = tf.reshape(x_input_b, [1, T * dX])
                        x_input_flat_b -= tf.tile(X_mean, [1, T]) # subtract mean
                        concat_list.append(x_input_flat_b)
                    if dU > 0:
                        u_input_flat_b = tf.reshape(u_input_b, [-1, T * dU])
                        u_input_flat_b -= tf.tile(U_mean, [1, T]) # subtract mean
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

    @staticmethod
    def _graph_inference_cnn(name, T, bootstrap_X_inputs, bootstrap_U_inputs, bootstrap_O_inputs,
                             X_mean, U_mean, O_mean, dropout, meta_data,
                             reuse=False, random_seed=None, finalize=True, tf_debug={}):
        num_bootstrap = len(bootstrap_X_inputs)

        bootstrap_output_mats = []
        bootstrap_output_preds = []

        ### hard-code camera
        assert(len(meta_data['prediction']['model']['O_order']) == 1 and
               meta_data['prediction']['model']['O_order'][0] == 'camera')
        cam_width = meta_data['O']['camera']['width']
        cam_height = meta_data['O']['camera']['height']

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
                n_output = 1

                ### concatenate inputs
                with tf.name_scope('inputs_b{0}'.format(b)):
                    concat_list = []
                    if dO > 0:
                        ### subtract mean
                        with tf.variable_scope('means', reuse=reuse or b > 0):
                            tf_O_mean = tf.get_variable('O_mean', shape=[1, len(O_mean)], trainable=False, dtype=tf.float32,
                                                        initializer=tf.constant_initializer(O_mean.reshape((1, len(O_mean)))))
                        o_input_b -= tf_O_mean

                        ### conv layers
                        with tf.name_scope('conv_b{0}'.format(b)):
                            with tf.variable_scope('conv_vars_{0}'.format(b), reuse=reuse):
                                conv_kernels_b = [
                                    tf.get_variable('conv_k_0_b{0}'.format(b), [4, 4, 1, 32],
                                                    initializer=tf.contrib.layers.xavier_initializer_conv2d()),
                                    tf.get_variable('conv_k_1_b{0}'.format(b), [4, 4, 32, 16],
                                                    initializer=tf.contrib.layers.xavier_initializer_conv2d()),
                                ]

                                conv_biases_b = [
                                    tf.get_variable('conv_b_0_b{0}'.format(b), [32],
                                                    initializer=tf.constant_initializer(0.)),
                                    tf.get_variable('conv_b_1_b{0}'.format(b), [16],
                                                    initializer=tf.constant_initializer(0.)),
                                ]

                            conv_strides_b = [
                                [1, 2, 2, 1],
                                [1, 2, 2, 1],
                            ]

                            max_pool_ksizes_b = [
                                [1, 2, 2, 1],
                                [1, 2, 2, 1]
                            ]

                            for var in conv_kernels_b + conv_biases_b:
                                tf_debug[var.name] = var

                            ### apply convolutions
                            conv_layer_b = tf.reshape(o_input_b, [-1, cam_width, cam_height, 1])
                            for i, (kernel, stride, bias, max_pool_ksize) in \
                                    enumerate(zip(conv_kernels_b, conv_strides_b, conv_biases_b, max_pool_ksizes_b)):
                                conv_layer_b = tf.nn.conv2d(conv_layer_b, kernel, strides=[1, 1, 1, 1], padding='VALID')
                                conv_layer_b = tf.nn.bias_add(conv_layer_b, bias)
                                conv_layer_b = tf.nn.relu(conv_layer_b)
                                conv_layer_b = tf.nn.max_pool(conv_layer_b, ksize=max_pool_ksize, strides=stride, padding='VALID')
                            conv_output_b = conv_layer_b

                            ### flatten
                            conv_output_shape_b = [s.value for s in conv_output_b.get_shape()]
                            flat_shape = np.prod(conv_output_shape_b[1:])
                            conv_output_flat_b = tf.reshape(conv_output_b, [-1, flat_shape])

                        concat_list.append(conv_output_flat_b)

                    if dX > 0:
                        with tf.variable_scope('means', reuse=reuse or b > 0):
                            tf_X_mean = tf.get_variable('X_mean', shape=[1, len(X_mean)], trainable=False, dtype=tf.float32,
                                                        initializer=tf.constant_initializer(X_mean.reshape((1, len(X_mean)))))

                        x_input_flat_b = tf.transpose(x_input_flat_b, [-1, T * dX])
                        x_input_flat_b -= tf.tile(tf_X_mean, [1, T]) # subtract mean
                        concat_list.append(x_input_flat_b)
                    if dU > 0:
                        with tf.variable_scope('means', reuse=reuse or b > 0):
                            tf_U_mean = tf.get_variable('U_mean', shape=[1, len(U_mean)], trainable=False, dtype=tf.float32,
                                                        initializer=tf.constant_initializer(U_mean.reshape((1, len(U_mean)))))

                        u_input_flat_b = tf.reshape(u_input_b, [-1, T * dU])
                        u_input_flat_b -= tf.tile(tf_U_mean, [1, T]) # subtract mean
                        concat_list.append(u_input_flat_b)
                    input_layer = tf.concat(1, concat_list)

                n_input = input_layer.get_shape()[-1].value

                ### weights
                with tf.variable_scope('fc_vars_{0}'.format(b), reuse=reuse):
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
                            layer = tf.nn.dropout(layer, dropout)
                        tf_debug['grad_fc_{0}_b{1}'.format(i, b)] = tf.gradients(layer, weight)[0]

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

        return output_pred_mean, output_pred_std, output_mat_mean, output_mat_std, bootstrap_output_mats

    @staticmethod
    def _graph_inference_rnn(name, T, bootstrap_X_inputs, bootstrap_U_inputs, bootstrap_O_inputs,
                             X_mean, U_mean, O_mean, dropout, meta_data,
                             reuse=False, random_seed=None, finalize=True, tf_debug={}):
        num_bootstrap = len(bootstrap_X_inputs)

        bootstrap_output_mats = []
        bootstrap_output_preds = []

        ### hard-code camera
        assert(len(meta_data['prediction']['model']['O_order']) == 1 and
               meta_data['prediction']['model']['O_order'][0] == 'camera')
        cam_width = meta_data['O']['camera']['width']
        cam_height = meta_data['O']['camera']['height']

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
                n_output = 1

                assert(dO > 0)
                assert(dU + dX > 0)

                ### subtract means
                with tf.name_scope('subtract_means_b{0}'.format(b)):
                    if dO > 0:
                        with tf.variable_scope('means', reuse=reuse or b > 0):
                            tf_O_mean = tf.get_variable('O_mean', shape=[1, len(O_mean)], trainable=False,
                                                        dtype=tf.float32,
                                                        initializer=tf.constant_initializer(
                                                            O_mean.reshape((1, len(O_mean)))))

                        o_input_b -= tf_O_mean
                    if dX > 0:
                        with tf.variable_scope('means', reuse=reuse or b > 0):
                            tf_X_mean = tf.get_variable('X_mean', shape=[1, len(X_mean)], trainable=False,
                                                        dtype=tf.float32,
                                                        initializer=tf.constant_initializer(
                                                            X_mean.reshape((1, len(X_mean)))))

                        x_input_b -= tf.tile(tf_X_mean, [T, 1])
                    if dU > 0:
                        with tf.variable_scope('means', reuse=reuse or b > 0):
                            tf_U_mean = tf.get_variable('U_mean', shape=[1, len(U_mean)], trainable=False,
                                                        dtype=tf.float32,
                                                        initializer=tf.constant_initializer(
                                                            U_mean.reshape((1, len(U_mean)))))

                        u_input_b -= tf.tile(tf_U_mean, [T, 1])

                ### create hidden/internal state
                ### initial hidden state learned from observation

                ### fc version
                # n_rnn = 40
                # with tf.variable_scope('obs_to_istate_vars_b{0}'.format(b), reuse=reuse):
                #     weights_obs_to_istate_b = [
                #         tf.get_variable('w_obs_to_istate_0_b{0}'.format(b), [dO, 40], initializer=tf.contrib.layers.xavier_initializer()),
                #         tf.get_variable('w_obs_to_istate_1_b{0}'.format(b), [40, n_rnn], initializer=tf.contrib.layers.xavier_initializer()),
                #     ]
                #     biases_obs_to_istate_b = [
                #         tf.get_variable('b_obs_to_istate_0_b{0}'.format(b), [40], initializer=tf.constant_initializer(0.)),
                #         tf.get_variable('b_obs_to_istate_1_b{0}'.format(b), [n_rnn], initializer=tf.constant_initializer(0.)),
                #     ]
                # for var in weights_obs_to_istate_b + biases_obs_to_istate_b:
                #     tf.add_to_collection('weight_decays', 0.5 * tf.reduce_mean(v ** 2))
                #
                # with tf.name_scope('obs_to_istate_b{0}'.format(b)):
                #     layer = o_input_b
                #     for weight, bias in zip(weights_obs_to_istate_b, biases_obs_to_istate_b):
                #         layer = tf.nn.relu(tf.add(tf.matmul(layer, weight), bias)) # TODO: dropout?
                #     istate_b = layer

                ### conv version
                with tf.variable_scope('obs_to_istate_vars_b{0}'.format(b), reuse=reuse):
                    conv_kernels_b = [
                        tf.get_variable('conv_k_0_b{0}'.format(b), [4, 4, 1, 32],
                                        initializer=tf.contrib.layers.xavier_initializer_conv2d()),
                        tf.get_variable('conv_k_1_b{0}'.format(b), [4, 4, 32, 16],
                                        initializer=tf.contrib.layers.xavier_initializer_conv2d()),
                    ]
                    conv_biases_b = [
                        tf.get_variable('conv_b_0_b{0}'.format(b), [32],
                                        initializer=tf.constant_initializer(0.)),
                        tf.get_variable('conv_b_1_b{0}'.format(b), [16],
                                        initializer=tf.constant_initializer(0.)),
                    ]

                conv_strides_b = [
                    [1, 2, 2, 1],
                    [1, 2, 2, 1],
                ]

                max_pool_ksizes_b = [
                    [1, 2, 2, 1],
                    [1, 2, 2, 1]
                ]

                conv_layer_b = tf.reshape(o_input_b, [-1, cam_width, cam_height, 1])
                for i, (kernel, stride, bias, max_pool_ksize) in \
                        enumerate(zip(conv_kernels_b, conv_strides_b, conv_biases_b, max_pool_ksizes_b)):
                    conv_layer_b = tf.nn.conv2d(conv_layer_b, kernel, strides=[1, 1, 1, 1], padding='VALID')
                    conv_layer_b = tf.nn.bias_add(conv_layer_b, bias)
                    conv_layer_b = tf.nn.relu(conv_layer_b)
                    conv_layer_b = tf.nn.max_pool(conv_layer_b, ksize=max_pool_ksize, strides=stride, padding='VALID')
                conv_output_b = conv_layer_b

                conv_output_shape_b = [s.value for s in conv_output_b.get_shape()]
                n_rnn = np.prod(conv_output_shape_b[1:])
                istate_b = tf.reshape(conv_output_b, [-1, n_rnn])

                ### create input at each timestep
                with tf.variable_scope('rnn_input_vars_b{0}'.format(b), reuse=reuse):
                    weights_rnn_input_b = [
                        tf.get_variable('w_rnn_input_0_b{0}'.format(b), [dU + dX, n_rnn], initializer=tf.contrib.layers.xavier_initializer()),
                    ]
                    biases_rnn_input_b = [
                        tf.get_variable('b_rnn_input_0_b{0}'.format(b), [n_rnn], initializer=tf.constant_initializer(0.)),
                    ]
                for v in weights_rnn_input_b + biases_rnn_input_b:
                    tf.add_to_collection('weight_decays', 0.5 * tf.reduce_mean(v ** 2))

                with tf.name_scope('rnn_input_b{0}'.format(b)):
                    rnn_input_b = []
                    for t in xrange(T):
                        layer = tf.concat(1, [u_input_b[:,t,:], x_input_b[:,t,:]])
                        for weight, bias in zip(weights_rnn_input_b, biases_rnn_input_b):
                            layer = tf.nn.relu(tf.add(tf.matmul(layer, weight), bias))
                        rnn_input_b.append(layer)

                    rnn_input_b = tf.pack(rnn_input_b, 1) # TODO: set shape?

                ### create rnn
                with tf.name_scope('rnn_b{0}'.format(b)):
                    with tf.variable_scope('rnn_vars_b{0}'.format(b), reuse=reuse):
                        # rnn_cell_b = BasicRNNCellDropout(rnn_input_b.get_shape()[2].value, activation=tf.nn.relu) # TODO: dropout?
                        rnn_cell_b = tf.nn.rnn_cell.BasicRNNCell(n_rnn, activation=tf.nn.relu)

                        # rnn_outputs_b, rnn_states_b = dynamic_rnn_GK(rnn_cell_b, rnn_input_b,
                        #                                              initial_state=istate_b,
                        #                                              dropout=dropout) # TODO dropout?
                        rnn_outputs_b, rnn_states_b = tf.nn.dynamic_rnn(rnn_cell_b, rnn_input_b, initial_state=istate_b)

                        rnn_final_output_b = rnn_outputs_b[:,T-1,:]

                ### final layers to get output
                with tf.variable_scope('rnn_output_vars_b{0}'.format(b), reuse=reuse):
                    weights_rnn_output_b = [
                        tf.get_variable('w_rnn_output_0_b{0}'.format(b), [n_rnn, n_output],
                                        initializer=tf.contrib.layers.xavier_initializer()),
                    ]
                    biases_rnn_output_b = [
                        tf.get_variable('b_rnn_output_0_b{0}'.format(b), [n_output],
                                        initializer=tf.constant_initializer(0.)),
                    ]
                for v in weights_rnn_output_b + biases_rnn_output_b:
                    tf.add_to_collection('weight_decays', 0.5 * tf.reduce_mean(v ** 2))

                with tf.name_scope('rnn_output_b{0}'.format(b)):
                    layer = rnn_final_output_b
                    for i, (weight, bias) in enumerate(zip(weights_rnn_output_b, biases_rnn_output_b)):
                        layer = tf.add(tf.matmul(layer, weight), bias) # TODO: dropout?
                        if i < len(weights_rnn_output_b) - 1:
                            layer = tf.nn.relu(layer)

                with tf.name_scope('output_pred_b{0}'.format(b)):
                    output_mat_b = layer
                    output_pred_b = tf.sigmoid(output_mat_b)

                bootstrap_output_mats.append(output_mat_b)
                bootstrap_output_preds.append(output_pred_b)

            ### combination of all the bootstraps
            with tf.name_scope('combine_bootstraps'):
                output_pred_mean = (1 / float(num_bootstrap)) * tf.add_n(bootstrap_output_preds,
                                                                         name='output_pred_mean')
                std_normalize = (1 / float(num_bootstrap - 1)) if num_bootstrap > 1 else 1
                output_pred_std = tf.sqrt(std_normalize * tf.add_n(
                    [tf.square(tf.sub(output_pred_b, output_pred_mean)) for output_pred_b in
                     bootstrap_output_preds]))

                output_mat_mean = (1 / float(num_bootstrap)) * tf.add_n(bootstrap_output_mats,
                                                                        name='output_mat_mean')
                output_mat_std = tf.sqrt(std_normalize * tf.add_n(
                    [tf.square(tf.sub(output_mat_b, output_mat_mean)) for output_mat_b in bootstrap_output_mats]))

        return output_pred_mean, output_pred_std, output_mat_mean, output_mat_std, bootstrap_output_mats

    @staticmethod
    def _create_multistep_graph(dX, dU, dO, T, num_bootstrap, dropout,
                                log_dir='/tmp', use_gpu=True, gpu_fraction=0.49, random_seed=None, reg=0):
        """
        Multistep model
        """
        dO -= 1 # TODO hard-coded to ignore collision
        assert(dU == 0)
        assert(dO > 0)

        graph = tf.Graph()
        with graph.as_default():
            tf.set_random_seed(random_seed)

            n_input = T * dX + dO
            n_output = T
            n_hidden_0 = 75
            n_hidden_1 = 50
            n_hidden_2 = 25

            ### neural network
            x_inputs, u_inputs, o_inputs, outputs = [], [], [], []
            weights, biases = [], []
            output_mats, output_preds = [], []
            costs, accuracies = [], []
            x_grads, u_grads = [], []
            ### one for each bootstrap
            for b in xrange(num_bootstrap):
                ### tf graph input/output
                x_input_b = tf.placeholder('float', [None, T, dX], name='x_input_b{0}'.format(b))
                u_input_b = tf.placeholder('float', [None, dU], name='o_input_b{0}'.format(b))
                o_input_b = tf.placeholder('float', [None, dO], name='o_input_b{0}'.format(b))
                output_b = tf.placeholder('float', [None, T], name='output_b{0}'.format(b))

                ### create variables
                shared_weights = True
                weights_b, biases_b = [], []
                def create_weights(t, b):
                    return [
                        tf.Variable(tf.random_normal([dO + dX, n_hidden_0]), name='w_hidden_0_t{0}_b{1}'.format(t,b)),
                        tf.Variable(tf.random_normal([n_hidden_0, n_hidden_1]), name='w_hidden_1_t{0}_b{1}'.format(t,b)),
                        tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2]), name='w_hidden_2_t{0}_b{1}'.format(t,b)),
                        tf.Variable(tf.random_normal([n_hidden_2, 1]), name='w_output_t{0}_b{1}'.format(t,b))
                    ]
                def create_biases(t, b):
                    return [
                        tf.Variable(tf.random_normal([n_hidden_0]), name='b_hidden_0_t{0}_b{1}'.format(t,b)),
                        tf.Variable(tf.random_normal([n_hidden_1]), name='b_hidden_1_t{0}_b{1}'.format(t,b)),
                        tf.Variable(tf.random_normal([n_hidden_2]), name='b_hidden_2_t{0}_b{1}'.format(t,b)),
                        tf.Variable(tf.random_normal([1]), name='b_output_t{0}_b{1}'.format(t,b))
                    ]
                weights_b.append(create_weights(0, b))
                biases_b.append(create_biases(0, b))
                for t in xrange(1, T):
                    if shared_weights:
                        weights_b.append(weights_b[0])
                        biases_b.append(biases_b[0])
                    else:
                        weights_b.append(create_weights(t, b))
                        biases_b.append(create_weights(t, b))

                output_mat_list_b = []
                with tf.name_scope('multistep_b{0}'.format(b)):
                    for t in xrange(T):
                        with tf.name_scope('multistep_t{0}_b{1}'.format(t, b)):
                            layer = tf.concat(1, [o_input_b, x_input_b[:,t,:]])
                            for i, (weight_b, bias_b) in enumerate(zip(weights_b[t], biases_b[t])):
                                layer = tf.add(tf.matmul(layer, weight_b), bias_b)
                                if i < len(weights_b[t]) - 1:
                                    layer = tf.nn.relu(layer)
                                    if dropout is not None:
                                        layer = tf.nn.dropout(layer, dropout)
                            output_mat_list_b.append(layer)

                with tf.name_scope('output_pred_b{0}'.format(b)):
                    output_mat_b = tf.concat(1, output_mat_list_b)
                    output_pred_b = tf.sigmoid(output_mat_b, name='output_pred_b{0}'.format(b))

                with tf.name_scope('scope_cost_b{0}'.format(b)):
                    ### sigmoid
                    with tf.name_scope('entropy_b{0}'.format(b)):
                        entropy = (1 / float(n_output)) * tf.reduce_mean(
                            tf.nn.sigmoid_cross_entropy_with_logits(output_mat_b, output_b))

                    with tf.name_scope('weight_decay_b{0}'.format(b)):
                        weights_b_list = np.array(weights_b).ravel().tolist()
                        num_weights = sum([np.prod(w.get_shape().as_list()) for w in weights_b_list])
                        l2_loss = 0 # TODO (1e-1 / float(num_weights)) * tf.add_n([tf.nn.l2_loss(w) for w in weights_b_list])

                    cost_b = tf.add_n([entropy, l2_loss], name='cost_b{0}'.format(b))
                with tf.name_scope('scope_accuracy_b{0}'.format(b)):
                    ### sigmoid
                    output_geq_b = tf.cast(tf.greater_equal(output_pred_b, 0.5), tf.float32)
                    output_correct_b = tf.reduce_mean(tf.cast(tf.equal(output_geq_b, output_b), tf.float32))
                    accuracy_b = tf.reduce_mean(output_correct_b, name='accuracy_b{0}'.format(b))

                with tf.name_scope('scope_grad_b{0}'.format(b)):
                    # output_pred_sep_b = tf.split(1, n_output, output_pred_b)
                    x_grads_b, u_grads_b = [], []
                    # if dX > 0:
                    #     x_grads_b = [tf.gradients(o, x_input_flat_b)[0] for o in output_pred_sep_b]
                    # if dU > 0:
                    #     u_grads_b = [tf.gradients(o, u_input_flat_b)[0] for o in output_pred_sep_b]

                ### append
                x_inputs.append(x_input_b)
                u_inputs.append(u_input_b)
                o_inputs.append(o_input_b)
                outputs.append(output_b)
                weights += np.array(weights_b).ravel().tolist()
                biases += np.array(biases_b).ravel().tolist()
                output_mats.append(output_mat_b)
                output_preds.append(output_pred_b)
                costs.append(cost_b)
                accuracies.append(accuracy_b)
                x_grads.append(x_grads_b)
                u_grads.append(u_grads_b)

            ### combination of all the bootstraps
            output_pred_mean = (1 / float(num_bootstrap)) * tf.add_n(output_preds, name='output_pred_mean')
            # output_mat_mean = (1 / float(num_bootstrap)) * tf.add_n(output_mats)
            # output_pred_mean = tf.sigmoid(output_mat_mean, name='output_pred_mean') # TODO

            std_normalize = (1 / float(num_bootstrap - 1)) if num_bootstrap > 1 else 1
            output_pred_std = tf.sqrt(std_normalize * tf.add_n(
                [tf.square(tf.sub(output_pred_b, output_pred_mean)) for output_pred_b in output_preds]))
            # output_pred_std = tf.sqrt(std_normalize * tf.add_n(
            #     [tf.square(tf.sigmoid(tf.sub(output_mat_b, output_mat_mean))) for output_mat_b in output_mats])) # TODO chug through nn or not?

            cost = (1 / float(num_bootstrap)) * tf.add_n(costs, name='cost')
            accuracy = (1 / float(num_bootstrap)) * tf.add_n(accuracies, name='accuracy')
            x_grad, u_grad = [0] * n_output, [0] * n_output
            # for i in xrange(n_output):
            #     x_grad[i] = (1 / float(num_bootstrap)) * tf.add_n(np.array(x_grads)[:, i].tolist())
            #     u_grad[i] = (1 / float(num_bootstrap)) * tf.add_n(np.array(u_grads)[:, i].tolist())

            optimizer = tf.train.AdamOptimizer().minimize(cost)

            ### initialize graph
            gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_fraction)
            init = tf.initialize_all_variables()
            saver = tf.train.Saver(max_to_keep=None)
            sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, device_count={'GPU': int(use_gpu)}),
                              graph=graph)
            sess.run(init)

            # Set logs writer into folder /tmp/tensorflow_logs
            merged = tf.merge_all_summaries()
            writer = tf.train.SummaryWriter(log_dir, graph_def=sess.graph_def)

            graph.finalize()

        return graph, sess, saver, init, \
               x_inputs, u_inputs, o_inputs, outputs, \
               output_pred_mean, output_pred_std, \
               cost, accuracy, optimizer, \
               x_grad, u_grad, \
               weights, biases, output_mats, output_preds

    ################
    ### Training ###
    ################

    def _create_input(self, X, U, O):
        return PredictionModel._create_input(self, X, U, O)

    def _create_output(self, output):
        return PredictionModel._create_output(self, output).astype(int)

