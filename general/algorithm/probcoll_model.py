import abc
import os, pickle
import random, time
import itertools
import shutil
from collections import defaultdict
import hashlib
import numpy as np
import tensorflow as tf

from general.utility.logger import get_logger
from general.state_info.sample import Sample
from general.algorithm.mlplotter import MLPlotter
from config import params

class ProbcollModel:
    __metaclass__ = abc.ABCMeta

    ####################
    ### Initializing ###
    ####################

    def __init__(self, dist_eps, read_only=False, finalize=True):
        self.dist_eps = dist_eps

        self.save_dir = os.path.join(params['exp_dir'], params['exp_name'])
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        self._logger = get_logger(self.__class__.__name__, 'debug', os.path.join(self.save_dir, 'debug.txt'))

        self.random_seed = params['random_seed']
        for k, v in params['model'].items():
            setattr(self, k, v)

        self.dX = len(self.X_idxs())
        self.dU = len(self.U_idxs())
        self.dO = len(self.O_idxs())
        self.doutput = len(self.output_idxs())

        self.tfrecords_train_fnames = [
                self.tfrecords_no_coll_train_fnames,
                self.tfrecords_coll_train_fnames
            ]

        self.tfrecords_val_fnames = [
                self.tfrecords_no_coll_val_fnames,
                self.tfrecords_coll_val_fnames
            ]
                
        self.preprocess_fnames = []
        
        self._control_width = np.array(self.control_range["upper"]) - \
            np.array(self.control_range["lower"])
        self.X_mean = np.zeros(self.dX)
        self.U_mean = np.zeros(self.dU)
        self.O_mean = np.zeros(self.dO)

        code_file_exists = os.path.exists(self._code_file)
        if code_file_exists:
            self._logger.info('Creating OLD graph')
        else:
            self._logger.info('Creating NEW graph')
            shutil.copyfile(self._this_file, self._code_file)
        self._graph_inference = self._get_old_graph_inference(graph_type=self.graph_type)

        self.tf_debug = {}
        self._graph_setup()

    #############
    ### Files ###
    #############
    
    @abc.abstractproperty
    @property
    def _this_file(self):
        raise NotImplementedError('Implement in subclass')

    
    @property
    def tfrecords_no_coll_train_fnames(self):
        return [
                os.path.join(self._no_coll_train_tfrecords_dir, fn)
                    for fn in os.listdir(self._no_coll_train_tfrecords_dir)
            ]

    @property
    def tfrecords_coll_train_fnames(self):
        return [
                os.path.join(self._coll_train_tfrecords_dir, fn)
                    for fn in os.listdir(self._coll_train_tfrecords_dir)
            ]
    
    @property
    def tfrecords_no_coll_val_fnames(self):
        return [
                os.path.join(self._no_coll_val_tfrecords_dir, fn)
                    for fn in os.listdir(self._no_coll_val_tfrecords_dir)
            ]

    @property
    def tfrecords_coll_val_fnames(self):
        return [
                os.path.join(self._coll_val_tfrecords_dir, fn)
                    for fn in os.listdir(self._coll_val_tfrecords_dir)
            ]

    @property
    def _code_file(self):
        return os.path.join(os.path.abspath(self.save_dir), 'probcoll_model_{0}.py'.format(params['exp_name']))

    @property
    def _hash(self):
        """ Anything that if changed, need to re-save the data """
        d = {}

        pm_params = params['model']
        for key in ('T', 'num_bootstrap', 'val_pct', 'X_order', 'U_order', 'O_order', 'output_order', 'balance',
                    'aggregate_save_data', 'save_type'):
            d[key] = pm_params[key]

        for key in ('X', 'U', 'O'):
            d[key] = params[key]

        return hashlib.md5(str(d)).hexdigest()

    ############
    ### DIR ####
    ############

    @property
    def _no_coll_train_tfrecords_dir(self):
        dir = os.path.join(self.save_dir, "no_coll_train_tfrecords") 
        if not os.path.exists(dir):
            os.makedirs(dir)
        return dir
    
    @property
    def _coll_train_tfrecords_dir(self):
        dir = os.path.join(self.save_dir, "coll_train_tfrecords")
        if not os.path.exists(dir):
            os.makedirs(dir)
        return dir
    
    @property
    def _no_coll_val_tfrecords_dir(self):
        dir = os.path.join(self.save_dir, "no_coll_val_tfrecords")
        if not os.path.exists(dir):
            os.makedirs(dir)
        return dir
    
    @property
    def _coll_val_tfrecords_dir(self):
        dir = os.path.join(self.save_dir, "coll_val_tfrecords") 
        if not os.path.exists(dir):
            os.makedirs(dir)
        return dir
    

    ############
    ### Data ###
    ############

    def X_idxs(self, p=None, without=[]):
        if p is None: p = params
        return list(itertools.chain(*[range(p['X'][ord]['idx'],
                                            p['X'][ord]['idx']+p['X'][ord]['dim'])
                                      for ord in self.X_order if ord not in without]))

    def U_idxs(self, p=None, without=[]):
        if p is None: p = params
        return list(itertools.chain(*[range(p['U'][ord]['idx'],
                                            p['U'][ord]['idx']+p['U'][ord]['dim'])
                                      for ord in self.U_order if ord not in without]))

    def O_idxs(self, p=None, without=[]):
        if p is None: p = params
        return list(itertools.chain(*[range(p['O'][ord]['idx'],
                                            p['O'][ord]['idx']+p['O'][ord]['dim'])
                                      for ord in self.O_order if ord not in without]))

    def output_idxs(self, p=None, without=[]):
        if p is None: p = params
        return list(itertools.chain(*[range(p['O'][ord]['idx'],
                                            p['O'][ord]['idx']+p['O'][ord]['dim'])
                                      for ord in self.output_order if ord not in without]))

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

        no_coll_data = {"X": [], "U": [], "O": [], "output": []}
        coll_data = {"X": [], "U": [], "O": [], "output": []}

        random.shuffle(npz_fnames)
        for npz_fname in npz_fnames:
            ### load samples
            # self._logger.debug('\tOpening {0}'.format(npz_fname))

            samples = Sample.load(npz_fname)
            # TODO possibly remove shuffle, only keeping for validation split
            random.shuffle(samples)

            ### add to data
            for og_sample in samples:
                for sample in self._modify_sample(og_sample):
                    s_params = sample._meta_data
                    X = sample.get_X()[:, self.X_idxs(p=s_params)]
                    U = sample.get_U()[:, self.U_idxs(p=s_params)]
                    O = sample.get_O()[:, self.O_idxs(p=s_params)]
                    output = sample.get_O()[:, self.output_idxs(p=s_params)].astype(np.int32)
                    buffer_len = 1
                    if len(X) < 1 + buffer_len: # used to be self.T, but now we are extending
                        continue
                    
                    for arr in (X, U, O, output):
                        assert(np.all(np.isfinite(arr)))
                        assert(not np.any(np.isnan(arr)))

                    if output[-1, 0] == 1:
                        # For collision data extend collision by T-1 (and buffer)
                        random_u = np.random.random((self.T - 1 - buffer_len, U.shape[1])) * \
                            self._control_width + np.array(self.control_range["lower"])
                        U_coll = np.vstack((U[-self.T:], random_u))

                        X_coll = np.vstack((X[-self.T:], np.tile([X[-1]], (self.T - 1 - buffer_len, 1))))
                        O_coll = np.vstack((O[-self.T:], np.tile([O[-1]], (self.T - 1 - buffer_len, 1))))
                        output_coll = np.vstack((output[-self.T:], np.tile([output[-1]], (self.T - 1 - buffer_len, 1))))
                        coll_data["X"].append(X_coll)
                        coll_data["U"].append(U_coll)
                        coll_data["O"].append(O_coll)
                        coll_data["output"].append(output_coll)
                        # For noncollision data remove the collision
                        X = X[:-1]
                        U = U[:-1]
                        O = O[:-1]
                        output = output[:-1]
                    
                    if len(X) >= self.T:
                        no_coll_data["X"].append(X)
                        no_coll_data["U"].append(U)
                        no_coll_data["O"].append(O)
                        no_coll_data["output"].append(output)

        return no_coll_data, coll_data 

    def _save_tfrecords(
            self,
            tfrecords,
            X_by_sample,
            U_by_sample,
            O_by_sample,
            output_by_sample):
        
        if self.save_type == 'fixedlen':
            save_tfrecords = self._save_tfrecords_fixedlen
        else:
            raise Exception('{0} not valid save type'.format(self.save_type))

        self.save_tfrecords(
            tfrecords,
            X_by_sample,
            U_by_sample,
            O_by_sample,
            output_by_sample)

    # TODO Not sure if this works
    def _save_tfrecords_fixedlen(
            self,
            tfrecords,
            X_by_sample,
            U_by_sample,
            O_by_sample,
            output_by_sample):

        def _floatlist_feature(value):
            return tf.train.Feature(float_list=tf.train.FloatList(value=value))

        def _int64list_feature(value):
            return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

        def _bytes_feature(value):
            return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

        writer = tf.python_io.TFRecordWriter(tfrecords)

        record_num = 0
        for i, (X, U, O, output) in enumerate(zip(X_by_sample, U_by_sample, O_by_sample, output_by_sample)):
            assert(len(X) >= self.T)
            for j in range(len(X) - self.T + 1):
                feature = {
                    'fname': _bytes_feature(os.path.splitext(os.path.basename(tfrecords))[0] + '_{0}'.format(record_num)),
                }
                feature['X'] = _floatlist_feature(np.ravel(X[j:j+self.T]).tolist())
                feature['U'] = _floatlist_feature(np.ravel(U[j:j+self.T]).tolist())
                feature['O'] = _floatlist_feature(np.ravel(O[j]).tolist())
                feature['output'] = _int64list_feature(np.ravel(output[j+self.T-1]).tolist())
                
                example = tf.train.Example(features=tf.train.Features(feature=feature))
                writer.write(example.SerializeToString())
                record_num += 1

        writer.close()

    def _compute_mean(self):
        """ Calculates mean and updates variable in graph """
        if len(self.preprocess_fnames) > 0:
            total_X_mean, total_U_mean, total_O_mean = 0, 0, 0
            total_X_cov, total_U_cov, total_O_cov = 0, 0, 0
            total_timesteps = 0
            for preprocess_fname in self.preprocess_fnames:
                d = np.load(preprocess_fname)
                total_timesteps += d['timesteps']
                total_X_mean += d['X_mean'] * d['timesteps']
                total_U_mean += d['U_mean'] * d['timesteps']
                total_O_mean += d['O_mean'] * d['timesteps']
                total_X_cov += d['X_cov'] * d['timesteps']
                total_U_cov += d['U_cov'] * d['timesteps']
                total_O_cov += d['O_cov'] * d['timesteps']

            X_mean = total_X_mean / float(total_timesteps)
            U_mean = total_U_mean / float(total_timesteps)
            O_mean = total_O_mean / float(total_timesteps)
            X_cov = total_X_cov / float(total_timesteps)
            U_cov = total_U_cov / float(total_timesteps)
            O_cov = total_O_cov / float(total_timesteps)
            if self.dX > 0:
                X_orth, X_eigs, _ = np.linalg.svd(X_cov)
                X_orth /= np.sqrt(X_eigs + 1e-5)
            else:
                X_orth = np.eye(self.dX, dtype=np.float32)
            if self.dU > 0:
                U_orth, U_eigs, _ = np.linalg.svd(U_cov)
                U_orth /= np.sqrt(U_eigs + 1e-5)
            else:
                U_orth = np.eye(self.dU, dtype=np.float32)
            if self.dO > 0 and self.use_O_orth:
                O_orth, O_eigs, _ = np.linalg.svd(O_cov)
                O_orth /= np.sqrt(O_eigs + 1e-5)
            else:
                O_orth = np.eye(self.dO, dtype=np.float32)
        else:
            X_mean = np.zeros(self.dX, dtype=np.float32)
            U_mean = np.zeros(self.dU, dtype=np.float32)
            O_mean = np.zeros(self.dO, dtype=np.float32)
            X_orth = np.eye(self.dX, dtype=np.float32)
            U_orth = np.eye(self.dU, dtype=np.float32)
            O_orth = np.eye(self.dO, dtype=np.float32)

        self.sess.run([self.d_mean['X_assign'], self.d_mean['U_assign'], self.d_mean['O_assign'],
                       self.d_orth['X_assign'], self.d_orth['U_assign'], self.d_orth['O_assign']],
                      feed_dict={self.d_mean['X_placeholder']: np.expand_dims(X_mean, 0),
                                 self.d_mean['U_placeholder']: np.expand_dims(U_mean, 0),
                                 self.d_mean['O_placeholder']: np.expand_dims(O_mean, 0),
                                 self.d_orth['X_placeholder']: X_orth,
                                 self.d_orth['U_placeholder']: U_orth,
                                 self.d_orth['O_placeholder']: O_orth})

    def _get_tfrecords_fnames(self, fname, create=True):
        def _file_creator(coll, val):
            if coll:
                if val:
                    tfdir = self._coll_val_tfrecords_dir
                else:
                    tfdir = self._coll_train_tfrecords_dir
            else:
                if val:
                    tfdir = self._no_coll_val_tfrecords_dir
                else:
                    tfdir = self._no_coll_train_tfrecords_dir
            sample_fname = fname.split("/")[-1]
            tffname = os.path.join(tfdir, sample_fname.replace(
                ".npz",
                "_{0}.tfrecord".format(self._hash)))
            return tffname
        coll_train_fname = _file_creator(True, False) 
        no_coll_train_fname = _file_creator(False, False)
        coll_val_fname = _file_creator(True, True)
        no_coll_val_fname = _file_creator(False, True)
        return [
                coll_train_fname,
                no_coll_train_fname,
                coll_val_fname,
                no_coll_val_fname
            ]

    def add_data(self, npz_fnames):
        tfrecords = self._get_tfrecords_fnames(npz_fnames[-1]) 
        tfrecords_coll_train, tfrecords_no_coll_train, \
            tfrecords_coll_val, tfrecords_no_coll_val = tfrecords
        self._logger.info('Saving tfrecords')
        no_coll_data, coll_data = self._load_samples(npz_fnames)
        no_coll_len = len(no_coll_data["X"])
        coll_len = len(coll_data["X"])
        tot_coll = sum([len(coll_data["X"][i]) - self.T + 1 for i in range(coll_len)])
        tot_no = sum([len(no_coll_data["X"][i]) - self.T + 1 for i in range(no_coll_len)])
        self._logger.info("Size of no collision data: {0}".format(tot_no))
        self._logger.info("Size of collision data: {0}".format(tot_coll))

        self._save_tfrecords_fixedlen(
            tfrecords_no_coll_train, 
            no_coll_data["X"][int(no_coll_len * self.val_pct):],
            no_coll_data["U"][int(no_coll_len * self.val_pct):],
            no_coll_data["O"][int(no_coll_len * self.val_pct):],
            no_coll_data["output"][int(no_coll_len * self.val_pct):])

        self._save_tfrecords_fixedlen(
            tfrecords_coll_train, 
            coll_data["X"][int(coll_len * self.val_pct):],
            coll_data["U"][int(coll_len * self.val_pct):],
            coll_data["O"][int(coll_len * self.val_pct):],
            coll_data["output"][int(coll_len * self.val_pct):])

        self._save_tfrecords_fixedlen(
            tfrecords_no_coll_val, 
            no_coll_data["X"][:int(no_coll_len * self.val_pct)],
            no_coll_data["U"][:int(no_coll_len * self.val_pct)],
            no_coll_data["O"][:int(no_coll_len * self.val_pct)],
            no_coll_data["output"][:int(no_coll_len * self.val_pct)])

        self._save_tfrecords_fixedlen(
            tfrecords_coll_val, 
            coll_data["X"][:int(coll_len * self.val_pct)],
            coll_data["U"][:int(coll_len * self.val_pct)],
            coll_data["O"][:int(coll_len * self.val_pct)],
            coll_data["output"][:int(coll_len * self.val_pct)])

    #############
    ### Graph ###
    #############

    def _graph_inputs_outputs_from_file(self, name):
        if self.save_type == 'fixedlen':
            graph_inputs_outputs_from_file = self._graph_inputs_outputs_from_file_fixedlen
        else:
            raise Exception('{0} is not valid save type'.format(self.save_type))

        return graph_inputs_outputs_from_file(name)

    def _graph_inputs_outputs_from_file_fixedlen(self, name):
        with tf.name_scope(name + '_file_input'):
            filename_places = (tf.placeholder(tf.string), tf.placeholder(tf.string))
            filename_vars = ( 
                    tf.get_variable(
                        name + '_no_coll_fnames',
                        initializer=filename_places[0],
                        validate_shape=False,
                        trainable=False),
                    tf.get_variable(
                        name + '_coll_fnames',
                        initializer=filename_places[1],
                        validate_shape=False,
                        trainable=False)
                )
            ### create file queues
            filename_queues = (
                    tf.train.string_input_producer(
                        filename_vars[0],
                        num_epochs=None,
                        shuffle=True),
                    tf.train.string_input_producer(
                        filename_vars[1],
                        num_epochs=None,
                        shuffle=True)
                )

            ### read and decode
            readers = [tf.TFRecordReader(), tf.TFRecordReader()]
            
            features = {
                'fname': tf.FixedLenFeature([], tf.string)
            }

            features['X'] = tf.FixedLenFeature([self.dX * self.T], tf.float32)
            features['U'] = tf.FixedLenFeature([self.dU * self.T], tf.float32)
            features['O'] = tf.FixedLenFeature([self.dO], tf.float32)
            features['output'] = tf.FixedLenFeature([1], tf.int64)
            
            # TODO make sure this works
            # Figure out how to do arbitrary split across batchsize
            inputs = [None, None]
            for i, fq in enumerate(filename_queues):
                serialized_examples = [readers[(b+i)%2].read(fq)[1] for b in xrange(self.num_bootstrap)]
                parsed_example = [
                        tf.parse_single_example(serialized_examples[b], features=features)
                        for b in xrange(self.num_bootstrap)
                    ]

                fname = parsed_example[0]['fname']
                bootstrap_X_input = [tf.reshape(parsed_example[b]['X'], (self.T, self.dX))
                                     for b in xrange(self.num_bootstrap)]
                bootstrap_U_input = [tf.reshape(parsed_example[b]['U'], (self.T, self.dU))
                                     for b in xrange(self.num_bootstrap)]
                bootstrap_O_input = [parsed_example[b]['O'] for b in xrange(self.num_bootstrap)]
                bootstrap_output = [parsed_example[b]['output'] for b in xrange(self.num_bootstrap)]
                inputs[i] = (fname,) + tuple(bootstrap_X_input + bootstrap_U_input + bootstrap_O_input + bootstrap_output)

            shuffled = tf.train.shuffle_batch_join(
                inputs,
                batch_size=self.batch_size,
                capacity=10*self.batch_size + 3 * self.batch_size,
                min_after_dequeue=10*self.batch_size)
            
            fname_batch = shuffled[0]
            bootstrap_X_inputs = shuffled[1:1+self.num_bootstrap]
            bootstrap_U_inputs = shuffled[1+self.num_bootstrap:1+2*self.num_bootstrap]
            bootstrap_O_inputs = shuffled[1+2*self.num_bootstrap:1+3*self.num_bootstrap]
            bootstrap_outputs = shuffled[1+3*self.num_bootstrap:1+4*self.num_bootstrap]

        return fname_batch, bootstrap_X_inputs, bootstrap_U_inputs, bootstrap_O_inputs, bootstrap_outputs,\
               filename_queues, filename_places, filename_vars

    def _graph_inputs_outputs_from_placeholders(self):
        with tf.variable_scope('feed_input'):
            bootstrap_X_inputs = [tf.placeholder('float', [None, self.T, self.dX]) for _ in xrange(self.num_bootstrap)]
            bootstrap_U_inputs = [tf.placeholder('float', [None, self.T, self.dU]) for _ in xrange(self.num_bootstrap)]
            bootstrap_O_inputs = [tf.placeholder('float', [None, self.dO]) for _ in xrange(self.num_bootstrap)]
            bootstrap_outputs = [tf.placeholder('float', [None]) for _ in xrange(self.num_bootstrap)]

        return bootstrap_X_inputs, bootstrap_U_inputs, bootstrap_O_inputs, bootstrap_outputs

    def _graph_cost(self, name, bootstrap_output_mats, bootstrap_outputs, reg=0.):
        with tf.name_scope(name + '_cost_and_err'):
            costs = []
            num_coll = 0
            num_errs_on_coll = 0
            num_nocoll = 0
            num_errs_on_nocoll = 0

            for b, (output_mat_b, output_b) in enumerate(zip(bootstrap_output_mats, bootstrap_outputs)):
                output_b = tf.to_float(output_b)
                output_pred_b = tf.nn.sigmoid(output_mat_b)

                ### cost
                with tf.name_scope('cost_b{0}'.format(b)):
                    cross_entropy_b = tf.nn.sigmoid_cross_entropy_with_logits(output_mat_b, output_b)
                    costs.append(cross_entropy_b)
                ### accuracy
                with tf.name_scope('err_b{0}'.format(b)):
                    output_geq_b = tf.cast(tf.greater_equal(output_pred_b, 0.5), tf.float32)
                    output_incorrect_b = tf.cast(tf.not_equal(output_geq_b, output_b), tf.float32)

                    ### coll err
                    num_coll += tf.reduce_sum(output_b)
                    num_errs_on_coll += tf.reduce_sum(output_b * output_incorrect_b)

                    ### nocoll err
                    num_nocoll += tf.reduce_sum(1 - output_b)
                    num_errs_on_nocoll += tf.reduce_sum((1 - output_b) * output_incorrect_b)

            with tf.name_scope('total'):
                cross_entropy = tf.reduce_mean(tf.concat(0, costs))
                weight_decay = reg * tf.add_n(tf.get_collection('weight_decays'))
                cost = cross_entropy + weight_decay
                err = (1. / tf.cast(num_coll + num_nocoll, tf.float32)) * (num_errs_on_coll + num_errs_on_nocoll)
                err_on_coll = tf.cond(num_coll >= 0,
                                      lambda: (1. / tf.cast(num_coll, tf.float32)) * num_errs_on_coll,
                                      lambda: tf.constant(np.nan))
                err_on_nocoll = tf.cond(num_nocoll > 0,
                                        lambda: (1. / tf.cast(num_nocoll, tf.float32)) * num_errs_on_nocoll,
                                        lambda: tf.constant(np.nan))


        return cost, cross_entropy, err, err_on_coll, err_on_nocoll

    def _graph_optimize(self, cost):
        vars_before = tf.global_variables()

        with tf.name_scope('optimizer'):
            opt = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
            optimizer = opt.minimize(cost)
            grad = opt.compute_gradients(cost)

        vars_after = tf.global_variables()
        optimizer_vars = list(set(vars_after).difference(set(vars_before)))

        return optimizer, grad, optimizer_vars

    def _graph_init_vars(self):
        self.sess.run(
            tf.global_variables_initializer(),
            feed_dict=dict([(p, []) for p in (
                    self.d_train['no_coll_queue_placeholder'],
                    self.d_train['coll_queue_placeholder'],
                    self.d_val['no_coll_queue_placeholder'],
                    self.d_val['coll_queue_placeholder']
                )]))
        self._compute_mean()

    def _graph_setup(self):
        """ Only call once """

        tf.reset_default_graph()

        self.d_mean = dict()
        self.d_orth = dict()
        self.d_train = dict()
        self.d_val = dict()
        self.d_eval = dict()

        ### data stuff
        with tf.variable_scope('means', reuse=False):
            # X
            self.d_mean['X_placeholder'] = tf.placeholder(tf.float32, shape=(1, self.dX))
            self.d_mean['X_var'] = tf.get_variable('X_mean', shape=[1, self.dX], trainable=False, dtype=tf.float32,
                                                   initializer=tf.constant_initializer(np.zeros((1, self.dX))))
            self.d_mean['X_assign'] = tf.assign(self.d_mean['X_var'], self.d_mean['X_placeholder'])
            # U
            self.d_mean['U_placeholder'] = tf.placeholder(tf.float32, shape=(1, self.dU))
            self.d_mean['U_var'] = tf.get_variable('U_mean', shape=[1, self.dU], trainable=False, dtype=tf.float32,
                                                   initializer=tf.constant_initializer(np.zeros((1, self.dU))))
            self.d_mean['U_assign'] = tf.assign(self.d_mean['U_var'], self.d_mean['U_placeholder'])
            # O
            self.d_mean['O_placeholder'] = tf.placeholder(tf.float32, shape=(1, self.dO))
            self.d_mean['O_var'] = tf.get_variable('O_mean', shape=[1, self.dO], trainable=False, dtype=tf.float32,
                                                   initializer=tf.constant_initializer(np.zeros((1, self.dO))))
            self.d_mean['O_assign'] = tf.assign(self.d_mean['O_var'], self.d_mean['O_placeholder'])
        with tf.variable_scope('orths', reuse=False):
            # X
            self.d_orth['X_placeholder'] = tf.placeholder(tf.float32, shape=(self.dX, self.dX))
            self.d_orth['X_var'] = tf.get_variable('X_orth', shape=[self.dX, self.dX], trainable=False, dtype=tf.float32,
                                                   initializer=tf.constant_initializer(np.zeros((self.dX, self.dX))))
            self.d_orth['X_assign'] = tf.assign(self.d_orth['X_var'], self.d_orth['X_placeholder'])
            # U
            self.d_orth['U_placeholder'] = tf.placeholder(tf.float32, shape=(self.dU, self.dU))
            self.d_orth['U_var'] = tf.get_variable('U_orth', shape=[self.dU, self.dU], trainable=False, dtype=tf.float32,
                                                   initializer=tf.constant_initializer(np.zeros((self.dU, self.dU))))
            self.d_orth['U_assign'] = tf.assign(self.d_orth['U_var'], self.d_orth['U_placeholder'])
            # O
            self.d_orth['O_placeholder'] = tf.placeholder(tf.float32, shape=(self.dO, self.dO))
            self.d_orth['O_var'] = tf.get_variable('O_orth', shape=[self.dO, self.dO], trainable=False, dtype=tf.float32,
                                                   initializer=tf.constant_initializer(np.zeros((self.dO, self.dO))))
            self.d_orth['O_assign'] = tf.assign(self.d_orth['O_var'], self.d_orth['O_placeholder'])

        ### prepare for training
        for i, (name, d) in enumerate((('train', self.d_train), ('val', self.d_val))):
            d['fnames'], d['X_inputs'], d['U_inputs'], d['O_inputs'], d['outputs'], \
            queues, queue_places, queue_vars = self._graph_inputs_outputs_from_file(name)
            d['no_coll_queue'], d['coll_queue'] = queues
            d['no_coll_queue_placeholder'], d['coll_queue_placeholder']  = queue_places
            d['no_coll_queue_var'], d['coll_queue_var'] = queue_vars
            _, _, _, _, d['output_mats'], _ = self._graph_inference(name, self.T,
                                                                    d['X_inputs'], d['U_inputs'], d['O_inputs'],
                                                                    self.d_mean['X_var'], self.d_orth['X_var'],
                                                                    self.d_mean['U_var'], self.d_orth['U_var'],
                                                                    self.d_mean['O_var'], self.d_orth['O_var'],
                                                                    self.dropout, params,
                                                                    reuse=i>0, random_seed=self.random_seed,
                                                                    tf_debug=self.tf_debug)
            d['cost'], d['cross_entropy'], d['err'], d['err_coll'], d['err_nocoll'] = \
                self._graph_cost(name, d['output_mats'], d['outputs'], reg=self.reg)

        ### optimizer
        self.d_train['optimizer'], self.d_train['grads'], self.d_train['optimizer_vars'] = \
            self._graph_optimize(self.d_train['cost'])

        ### prepare for eval
        self.d_eval['X_inputs'], self.d_eval['U_inputs'], self.d_eval['O_inputs'], self.d_eval['outputs'] = \
            self._graph_inputs_outputs_from_placeholders()
        self.d_eval['output_pred_mean'], self.d_eval['output_pred_std'], self.d_eval['output_mat_mean'], \
        self.d_eval['output_mat_std'], _, self.d_eval['dropout_placeholders'] = \
            self._graph_inference('eval', self.T,
                                  self.d_eval['X_inputs'], self.d_eval['U_inputs'], self.d_eval['O_inputs'],
                                  self.d_mean['X_var'], self.d_orth['X_var'],
                                  self.d_mean['U_var'], self.d_orth['U_var'],
                                  self.d_mean['O_var'], self.d_orth['O_var'],
                                  self.dropout, params,
                                  reuse=True, random_seed=self.random_seed)

        ### initialize
        os.environ["CUDA_VISIBLE_DEVICES"] = str(self.device)
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=self.gpu_fraction)
        config = tf.ConfigProto(gpu_options=gpu_options,
                                log_device_placement=False,
                                allow_soft_placement=True)
        # config.intra_op_parallelism_threads = 1
        # config.inter_op_parallelism_threads = 1
        self.sess = tf.Session(config=config)
        self._graph_init_vars()

        # Set logs writer into folder /tmp/tensorflow_logs
        merged = tf.merge_all_summaries()
        writer = tf.train.SummaryWriter('/tmp', graph_def=self.sess.graph_def)

        self.saver = tf.train.Saver(max_to_keep=None)

    @abc.abstractmethod
    def _get_old_graph_inference(self):
        raise NotImplementedError('Implement in subclass')

    # @staticmethod
    # @abc.abstractmethod

    # def _graph_inference(name, T, bootstrap_X_inputs, bootstrap_U_inputs, bootstrap_O_inputs,
    #                      X_mean, U_mean, O_mean, dropout, meta_data,
    #                      reuse=False, random_seed=None, finalize=True, tf_debug={}):
    #     raise NotImplementedError('Implement in subclass')

    ################
    ### Training ###
    ################

    def _create_input(self, X, U, O):
        return X, U, O

    def _create_output(self, output):
        return output

    def _flush_queue(self):
        for tfrecords_fnames, queue in (
                    (self.tfrecords_no_coll_train_fnames, self.d_train['no_coll_queue']),
                    (self.tfrecords_coll_train_fnames, self.d_train['coll_queue']),
                    (self.tfrecords_no_coll_val_fnames, self.d_val['no_coll_queue']),
                    (self.tfrecords_coll_val_fnames, self.d_val['coll_queue'])
                ):
            while not np.all([(fname in tfrecords_fnames) for fname in
                              self.sess.run(queue.dequeue_many(10*self.batch_size))]):
                pass

    def train(self, prev_model_file=None, new_model_file=None, **kwargs):

        if prev_model_file is not None and not self.reset_every_train:
            self.load(prev_model_file)
        else:
            self._graph_init_vars()
#        self._compute_mean()

        self.sess.run(
            [
                tf.assign(self.d_train['no_coll_queue_var'], self.tfrecords_no_coll_train_fnames, validate_shape=False),
                tf.assign(self.d_train['coll_queue_var'], self.tfrecords_coll_train_fnames, validate_shape=False),
                tf.assign(self.d_val['no_coll_queue_var'], self.tfrecords_no_coll_val_fnames, validate_shape=False),
                tf.assign(self.d_val['coll_queue_var'], self.tfrecords_coll_val_fnames, validate_shape=False)
            ])

        if not hasattr(self, 'coord'):
            assert(not hasattr(self, 'threads'))
            self.coord = tf.train.Coordinator()
            self.threads = tf.train.start_queue_runners(sess=self.sess, coord=self.coord)

        self._logger.debug('Flushing queue')
        self._flush_queue()

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

        train_fnames_dict = defaultdict(int)
        val_fnames_dict = defaultdict(int)

        step = 0
        epoch_start = save_start = time.time()
        while step < self.steps:
            if step == 0:
                for _ in xrange(10): print('')

            ### validation
            if (step != 0 and (step % self.val_freq) == 0):
                val_values = defaultdict(list)
                val_nums = defaultdict(float)
                val_steps = 0
                self._logger.debug('\tComputing validation...')
                while val_steps < self.val_steps:
                    val_cost, val_cross_entropy, \
                    val_err, val_err_coll, val_err_nocoll, \
                    val_fnames, val_outputs = \
                        self.sess.run([self.d_val['cost'], self.d_val['cross_entropy'],
                                       self.d_val['err'], self.d_val['err_coll'], self.d_val['err_nocoll'],
                                       self.d_val['fnames'], self.d_val['outputs']])

                    val_values['cost'].append(val_cost)
                    val_values['cross_entropy'].append(val_cross_entropy)
                    val_values['err'].append(val_err)
                    if not np.isnan(val_err_coll): val_values['err_coll'].append(val_err_coll)
                    if not np.isnan(val_err_nocoll): val_values['err_nocoll'].append(val_err_nocoll)
                    val_nums['coll'] += np.sum(np.concatenate(val_outputs))
                    val_nums['nocoll'] += np.sum(1 - np.concatenate(val_outputs))

                    val_steps += 1

                plotter.add_val('err', np.mean(val_values['err']))
                plotter.add_val('err_coll', np.mean(val_values['err_coll']))
                plotter.add_val('err_nocoll', np.mean(val_values['err_nocoll']))
                plotter.add_val('cost', np.mean(val_values['cost']))
                plotter.add_val('cross_entropy', np.mean(val_values['cross_entropy']))
                plotter.plot()

                self._logger.debug(
                    'error: {0:5.2f}%,  error coll: {1:5.2f}%,  error nocoll: {2:5.2f}%,  pct coll: {3:4.1f}%,  cost: {4:4.2f}, ce: {5:4.2f} ({6:.2f} s per {7:04d} samples)'.format(
                        100 * np.mean(val_values['err']),
                        100 * np.mean(val_values['err_coll']),
                        100 * np.mean(val_values['err_nocoll']),
                        100 * val_nums['coll'] / (val_nums['coll'] + val_nums['nocoll']),
                        np.mean(val_values['cost']),
                        np.mean(val_values['cross_entropy']),
                        time.time() - epoch_start,
                        int(self.val_freq * self.batch_size)))
                
                fnames_condensed = defaultdict(int)
                for k, v in train_fnames_dict.items():
                    fnames_condensed[k.split(self._hash)[0]] += v
                for k, v in sorted(fnames_condensed.items(), key=lambda x: x[1]):
                    self._logger.debug('\t\t\t{0} : {1}'.format(k, v))

                epoch_start = time.time()

                ### save model
                self.save(new_model_file)

            ### train
            _, train_cost, train_cross_entropy, \
            train_err, train_err_coll, train_err_nocoll, \
            train_fnames, train_outputs = self.sess.run([self.d_train['optimizer'],
                                                         self.d_train['cost'],
                                                         self.d_train['cross_entropy'],
                                                         self.d_train['err'],
                                                         self.d_train['err_coll'],
                                                         self.d_train['err_nocoll'],
                                                         self.d_train['fnames'],
                                                         self.d_train['outputs']])

            train_values['cost'].append(train_cost)
            train_values['cross_entropy'].append(train_cross_entropy)
            train_values['err'].append(train_err)
            if not np.isnan(train_err_coll): train_values['err_coll'].append(train_err_coll)
            if not np.isnan(train_err_nocoll): train_values['err_nocoll'].append(train_err_nocoll)
            train_nums['coll'] += np.sum(np.concatenate(train_outputs))
            train_nums['nocoll'] += np.sum(1 - np.concatenate(train_outputs))

            # Print an overview fairly often.
            if step % self.display_batch == 0 and step > 0:
                plotter.add_train('err', step * self.batch_size, np.mean(train_values['err']))
                if len(train_values['err_coll']) > 0:
                    plotter.add_train('err_coll', step * self.batch_size, np.mean(train_values['err_coll']))
                if len(train_values['err_nocoll']) > 0:
                    plotter.add_train('err_nocoll', step * self.batch_size, np.mean(train_values['err_nocoll']))
                plotter.add_train('cost', step * self.batch_size, np.mean(train_values['cost']))
                plotter.add_train('cross_entropy', step * self.batch_size, np.mean(train_values['cross_entropy']))
                plotter.plot()

                self._logger.debug('\tstep pct: {0:.1f}%,  error: {1:5.2f}%,  error coll: {2:5.2f}%,  error nocoll: {3:5.2f}%,  pct coll: {4:4.1f}%,  cost: {5:4.2f}, ce: {6:4.2f}'.format(
                    100 * step / float(self.steps),
                    100 * np.mean(train_values['err']),
                    100 * np.mean(train_values['err_coll']),
                    100 * np.mean(train_values['err_nocoll']),
                    100 * train_nums['coll'] / (train_nums['coll'] + train_nums['nocoll']),
                    np.mean(train_values['cost']),
                    np.mean(train_values['cross_entropy'])))

                train_values = defaultdict(list)
                train_nums = defaultdict(float)

            if time.time() - save_start > 60.:
                plotter.save(os.path.dirname(new_model_file))
                save_start = time.time()

            step += 1

        self.save(new_model_file)
        plotter.save(os.path.dirname(new_model_file))
        plotter.close()

    ##################
    ### Evaluating ###
    ##################

    def eval(self, X, U, O, num_avg=1, pre_activation=False):
        return self.eval_batch([X], [U], [O], num_avg=num_avg, pre_activation=pre_activation)

    def eval_batch(self, Xs, Us, Os, num_avg=1, pre_activation=False):
        X_inputs, U_inputs, O_inputs = [], [], []
        for X, U, O in zip(Xs, Us, Os):
            assert(len(X) >= self.T)
            assert(len(U) >= self.T)
            assert(len(O) >= 1)

            X_input, U_input, O_input = self._create_input(X, U, O)
            X_input = X_input[:self.T]
            U_input = U_input[:self.T]
            O_input = O_input[0]
            assert(not np.isnan(X_input).any())
            assert(not np.isnan(U_input).any())
            assert(not np.isnan(O_input).any())
            for _ in xrange(num_avg):
                X_inputs.append(X_input)
                U_inputs.append(U_input)
                O_inputs.append(O_input)

        feed = {}
        for b in xrange(self.num_bootstrap):
            feed[self.d_eval['X_inputs'][b]] = X_inputs
            feed[self.d_eval['U_inputs'][b]] = U_inputs
            feed[self.d_eval['O_inputs'][b]] = O_inputs
        # want dropout for each X/U/O to be the same
        # want dropout for each num_avg to be different
        # -->
        # create num_avg different dropout for each dropout mask
        # use same dropout mask for each X/U/O
        #
        # if num_avg = 3
        # 0 1 2 0 1 2 0 1 2
        for dropout_placeholder in self.d_eval['dropout_placeholders']:
            length = dropout_placeholder.get_shape()[1].value
            feed[dropout_placeholder] = [(1/self.dropout) * (np.random.random(length) < self.dropout).astype(float)
                                         for _ in xrange(num_avg)] * len(Xs)

        if pre_activation:
            output_pred_mean, output_pred_std = self.sess.run([self.d_eval['output_mat_mean'],
                                                               self.d_eval['output_mat_std']],
                                                              feed_dict=feed)
        else:
            output_pred_mean, output_pred_std = self.sess.run([self.d_eval['output_pred_mean'],
                                                               self.d_eval['output_pred_std']],
                                                              feed_dict=feed)

        if num_avg > 1:
            mean, std = [], []
            for i in xrange(len(Xs)):
                mean.append(output_pred_mean[i*num_avg:(i+1)*num_avg].mean(axis=0))
                std.append(output_pred_std[i*num_avg:(i+1)*num_avg].mean(axis=0))
            output_pred_mean = np.array(mean)
            output_pred_std = np.array(std)

            assert(len(output_pred_mean) == len(Xs))
            assert(len(output_pred_std) == len(Xs))

        assert((output_pred_std >= 0).all())

        return output_pred_mean, output_pred_std

    def eval_sample(self, sample):
        X = sample.get_X()[:self.T, self.X_idxs(sample._meta_data)]
        U = sample.get_U()[:self.T, self.U_idxs(sample._meta_data)]
        O = sample.get_O()[:self.T, self.O_idxs(sample._meta_data)]

        return self.eval(X, U, O)

    def eval_sample_batch(self, samples, num_avg=1, pre_activation=False):
        Xs = [sample.get_X()[:self.T, self.X_idxs(sample._meta_data)] for sample in samples]
        Us = [sample.get_U()[:self.T, self.U_idxs(sample._meta_data)] for sample in samples]
        Os = [sample.get_O()[:self.T, self.O_idxs(sample._meta_data)] for sample in samples]

        return self.eval_batch(Xs, Us, Os, num_avg=num_avg, pre_activation=pre_activation)

    #############################
    ### Load/save/reset/close ###
    #############################

    def load(self, model_file):
        self.saver.restore(self.sess, model_file)

    def save(self, model_file):
        self.saver.save(self.sess, model_file, write_meta_graph=False)

    def close(self):
        """ Release tf session """
        if hasattr(self, 'coord'):
            assert(hasattr(self, 'threads'))
            self.coord.request_stop()
            self.coord.join(self.threads)
        self.sess.close()
        self.sess = None

    @staticmethod
    def checkpoint_exists(model_file):
        ckpt = tf.train.get_checkpoint_state(os.path.dirname(model_file))
        return ckpt is not None and model_file in ckpt.model_checkpoint_path

