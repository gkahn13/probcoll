import abc
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

from general.tf import tf_utils
from general.tf.nn.fc_nn import fcnn
from general.tf.nn.conv_nn import convnn
from general.tf.nn.rnn import rnn
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
        tf.logging.set_verbosity(tf.logging.ERROR)
        self.random_seed = params['random_seed']
        for k, v in params['model'].items():
            setattr(self, k, v)

        self.dX = len(self.X_idxs())
        self.dU = len(self.U_idxs())
        self.dO = len(self.O_idxs())
        self.doutput = len(self.output_idxs())
        self.dtype = tf_utils.str_to_dtype(params["model"]["dtype"])

        self.tfrecords_train_fnames = [
                self.tfrecords_no_coll_train_fnames,
                self.tfrecords_coll_train_fnames
            ]

        self.tfrecords_val_fnames = [
                self.tfrecords_no_coll_val_fnames,
                self.tfrecords_coll_val_fnames
            ]
                
        self.preprocess_fnames = []
        self.threads = []

        self._control_width = np.array(self.control_range["upper"]) - \
            np.array(self.control_range["lower"])
        self._control_mean = (np.array(self.control_range["upper"]) + \
            np.array(self.control_range["lower"]))/2.
        self.dropout = params["model"]["action_graph"].get("dropout", None)

        code_file_exists = os.path.exists(self._code_file)
        if code_file_exists:
            self._logger.info('Creating OLD graph')
        else:
            self._logger.info('Creating NEW graph')
            shutil.copyfile(self._this_file, self._code_file)
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
        for key in ('T', 'num_bootstrap', 'val_pct', 'X_order', 'U_order', 'O_order', 'output_order', 'save_type'):
            d[key] = pm_params[key]

        for key in ('X', 'U', 'O'):
            d[key] = params[key]

        for key in ('observation_graph', 'action_graph', 'output_graph'):
            d[key] = pm_params[key]['graph_type']

        return hashlib.md5(str(d)).hexdigest()

    def _next_model_file(self):
        latest_file = tf.train.latest_checkpoint(
            self._checkpoints_dir)
        if latest_file is None:
            next_num = 0
        else:
            # File should be some number.ckpt
            num = int(os.path.splitext(os.path.basename(latest_file))[0])
            next_num = num + 1
        return os.path.join(
            self._checkpoints_dir,
            "{0:d}.ckpt".format(next_num)), next_num

    ############
    ### DIR ####
    ############

    @property
    def _checkpoints_dir(self):
        dir = os.path.join(self.save_dir, "model_checkpoints") 
        if not os.path.exists(dir):
            os.makedirs(dir)
        return dir

    @property
    def _plots_dir(self):
        dir = os.path.join(self.save_dir, "plots") 
        if not os.path.exists(dir):
            os.makedirs(dir)
        return dir
    
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

        no_coll_data = {"X": [], "U": [], "O": [], "output": [], "len": []}
        coll_data = {"X": [], "U": [], "O": [], "output": [], "len": []}

        random.shuffle(npz_fnames)
        for npz_fname in npz_fnames:
            ### load samples
            # self._logger.debug('\tOpening {0}'.format(npz_fname))

            samples = Sample.load(npz_fname)
            ### add to data
            for og_sample in samples:
                for sample in self._modify_sample(og_sample):
                    s_params = sample._meta_data
                    X = sample.get_X()[:, self.X_idxs(p=s_params)]
                    U = sample.get_U()[:, self.U_idxs(p=s_params)]
                    O = sample.get_O()[:, self.O_idxs(p=s_params)]
                    output = sample.get_O()[:, self.output_idxs(p=s_params)].astype(np.uint8)
                    buffer_len = 1
                    if len(X) < 1 + buffer_len: # used to be self.T, but now we are extending
                        continue
                    
                    for arr in (X, U, O, output):
                        assert(np.all(np.isfinite(arr)))
                        assert(not np.any(np.isnan(arr)))

                    if output[-1, 0] == 1:
                        # For collision data extend collision by T-1 (and buffer)
#                        random_u = np.random.random((self.T - 1 - buffer_len, U.shape[1])) * \
#                            self._control_width + np.array(self.control_range["lower"])
                        extend_u = np.zeros((self.T - 1 - buffer_len, U.shape[1]))
                        U_coll = np.vstack((U[-self.T:], extend_u))

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
                   
                    # Only add if non-collision part is long enough
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
                # TODO figure out how to keep types properly
                feature['O'] = _floatlist_feature(np.ravel(O[j]).tolist())
                output_list = np.ravel(output[j:j+self.T])
                feature['output'] = _bytes_feature(output_list.tostring())
                index = np.argmax(output_list)
                max_value = output_list[index]
                if max_value == 0:
                    length = self.T
                else:
                    length = index + 1
                assert(length != 0)
                feature['len'] = _int64list_feature([length])
                example = tf.train.Example(features=tf.train.Features(feature=feature))
                writer.write(example.SerializeToString())
                record_num += 1

        writer.close()

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
        tot_coll = sum([np.argmax(coll_data["output"][i]) + 1 for i in range(coll_len)])
        tot_no = sum([len(no_coll_data["output"][i]) - self.T + 1 for i in range(no_coll_len)])
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
                        initializer=tf.constant([], dtype=tf.string),
                        validate_shape=False,
                        trainable=False),
                    tf.get_variable(
                        name + '_coll_fnames',
                        initializer=tf.constant([], dtype=tf.string),
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
            features['output'] = tf.FixedLenFeature([], tf.string)
            features['len'] = tf.FixedLenFeature([], tf.int64)
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
                bootstrap_O_input = [tf.reshape(parsed_example[b]['O'], (self.dO,))
                                     for b in xrange(self.num_bootstrap)]
                bootstrap_output = [tf.reshape(tf.decode_raw(parsed_example[b]['output'], tf.uint8), (self.T, self.doutput))  for b in xrange(self.num_bootstrap)]
                bootstrap_len = [tf.reshape(parsed_example[b]['len'], ())
                                 for b in xrange(self.num_bootstrap)]
                inputs[i] = (fname,) + tuple(bootstrap_X_input + bootstrap_U_input + bootstrap_O_input + bootstrap_output + bootstrap_len)

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
            bootstrap_lens = shuffled[1+4*self.num_bootstrap:1+5*self.num_bootstrap]

        return fname_batch, bootstrap_X_inputs, bootstrap_U_inputs, bootstrap_O_inputs, bootstrap_outputs,\
               bootstrap_lens, filename_queues, filename_places, filename_vars

    def _graph_inputs_from_placeholders(self):
        with tf.variable_scope('feed_input'):
            X_inputs = tf.placeholder(self.dtype, [None, self.T, self.dX])
            U_inputs = tf.placeholder(self.dtype, [None, self.T, self.dU])
            O_input = tf.placeholder(self.dtype, [1, self.dO])
        return X_inputs, U_inputs, O_input

    def get_embedding(self, observation, batch_size=1, reuse=False, scope=None):
        
        obg_type = params["model"]["observation_graph"]["graph_type"]
        if obg_type == "fc":
            observation_graph = fcnn
        elif obg_type == "cnn":
            observation_graph = convnn
        else:
            raise NotImplementedError(
                "Observation graph {0} is not valid".format(obg_type))

        obs_batch = observation.get_shape()[0].value
        obs_float = tf.cast(observation, self.dtype) / 255.
        if not params['O']['use_depth']:
            obs_float = obs_float - tf.reduce_mean(obs_float, axis=0)
        obs_shaped = tf.reshape(
            obs_float,
            [
                obs_batch,
                params["O"]["camera"]["height"],
                params["O"]["camera"]["width"],
                params["O"]["camera"]["num_channels"]
            ])
        conv_output = observation_graph(
            obs_shaped,
            params["model"]["observation_graph"],
            dtype=self.dtype,
            scope=scope,
            reuse=reuse)
        output = tf.contrib.layers.flatten(conv_output)
        if obs_batch == 1:
            output = tf.tile(output, [batch_size, 1])
        return output
   
    def _graph_inference(
            self, name, bootstrap_X_inputs, bootstrap_U_inputs, bootstrap_O_inputs,
            reuse=False, finalize=True, tf_debug={}):
        assert(name == 'train' or name == 'val')
        num_bootstrap = params['model']['num_bootstrap']

        bootstrap_output_mats = []
        bootstrap_output_preds = []
        ag_type = params["model"]["action_graph"]["graph_type"]
        
        if ag_type == "fc":
            action_graph = fcnn
            recurrent = False
        elif ag_type == "rnn":
            action_graph = rnn
            recurrent = True
        else:
            raise NotImplementedError(
                "Action graph {0} is not valid".format(ag_type))

        if recurrent:
            assert(self.dO > 0)

        with tf.name_scope(name + '_inference'):
            tf.set_random_seed(self.random_seed)

            for b in xrange(num_bootstrap):
                ### inputs
                x_input_b = bootstrap_X_inputs[b]
                u_input_b = bootstrap_U_inputs[b]
                o_input_b = bootstrap_O_inputs[b]
                batch_size = tf.shape(x_input_b)[0]

                ### concatenate inputs
                with tf.name_scope('inputs_b{0}'.format(b)):
                    concat_list = []
                    if self.dX > 0:
                        
                        if recurrent:
                            x_input_flat_b = tf.reshape(x_input_b, [batch_size, self.T, self.dX])
                        else:
                            x_input_flat_b = tf.reshape(x_input_b, [batch_size, self.T * self.dX])
                        
                        concat_list.append(x_input_flat_b)
                    
                    if self.dU > 0:
                        control_mean = (np.array(params['model']['control_range']['lower']) + \
                            np.array(params['model']['control_range']['upper']))/2.
                        control_width = (np.array(params['model']['control_range']['upper']) - \
                            control_mean)
                        u_input_b = tf.cast((u_input_b - control_mean) / control_width, self.dtype) 
                        
                        if recurrent:
                            u_input_flat_b = tf.reshape(u_input_b, [batch_size, self.T, self.dU])
                        else:
                            u_input_flat_b = tf.reshape(u_input_b, [batch_size, self.T * self.dU])
                        
                        concat_list.append(u_input_flat_b)
                    
                    
                    if self.dO > 0:
                        initial_state = self.get_embedding(
                            o_input_b,
                            batch_size=batch_size,
                            reuse=reuse,
                            scope="observation_graph_b{0}".format(b))

                        if not recurrent:
                            concat_list.append(initial_state)
                    
                    if recurrent:
                        input_layer = tf.concat(2, concat_list)
                    else:
                        input_layer = tf.concat(1, concat_list)

                    if recurrent:
                        ag_output, _  = action_graph(
                            inputs=input_layer,
                            initial_state=initial_state,
                            params=params["model"]["action_graph"],
                            dtype=self.dtype,
                            scope="action_graph_b{0}".format(b),
                            reuse=reuse)
                    else:
                        ag_output, _  = action_graph(
                            inputs=input_layer,
                            params=params["model"]["action_graph"],
                            dtype=self.dtype,
                            scope="action_graph_b{0}".format(b),
                            reuse=reuse)

                    if recurrent:
                        ag_output = tf.reshape(
                            ag_output,
                            (batch_size * self.T, int(ag_output.get_shape()[-1]))) 
                    else:
                        ag_output = tf.reshape(
                            ag_output,
                            (batch_size * self.T, int(ag_output.get_shape()[-1])/self.T)) 

                    params["model"]["output_graph"]["output_dim"] = self.doutput
                    params["model"]["output_graph"]["dropout"] = None
                    # TODO dropout acts only on output of layers
                    # Therefore, dropout should never be put on outputgraph
                    output_mat_b, _  = fcnn(
                        ag_output,
                        params["model"]["output_graph"],
                        dtype=self.dtype,
                        scope="output_graph_b{0}".format(b),
                        reuse=reuse)

                    output_mat_b = tf.reshape(output_mat_b, [batch_size, self.T, self.doutput])
                    # TODO not general because it assumes doutput = 1
                    if params["model"]["prob_coll_strictly_increasing"]:
                        output_mat_b = tf.reshape(output_mat_b, (batch_size, self.T))
                        output_mat_b = tf_utils.cumulative_increasing_sum(
                            output_mat_b,
                            self.dtype)
                        output_mat_b = tf.reshape(output_mat_b, (batch_size, self.T, self.doutput))

                    output_pred_b = tf.sigmoid(output_mat_b, name='output_pred_b{0}'.format(b))

                bootstrap_output_mats.append(output_mat_b)
                bootstrap_output_preds.append(output_pred_b)

        return bootstrap_output_mats

    def graph_eval_inference(
            self, X_input, U_input, O_input=None, bootstrap_initial_states=None,
            reuse=False, finalize=True, tf_debug={}):
        
        bootstrap_output_mats = []
        bootstrap_output_preds = []
        dp_masks = []
        given_initial_states = bootstrap_initial_states is not None
        num_bootstrap = params['model']['num_bootstrap']
        ag_type = params["model"]["action_graph"]["graph_type"]
        
        if ag_type == "fc":
            action_graph = fcnn
            recurrent = False
        elif ag_type == "rnn":
            action_graph = rnn
            recurrent = True
        else:
            raise NotImplementedError(
                "Action graph {0} is not valid".format(ag_type))

        if recurrent:
            assert(self.dO > 0 or given_initial_states)

        batch_size = tf.shape(U_input)[0]

        with tf.name_scope('eval_inference'):
            tf.set_random_seed(self.random_seed)
            x_input_b = X_input
            u_input_b = U_input
            o_input_b = O_input
           
            base_concat_list = []

            if self.dX > 0:
                
                if recurrent:
                    x_input_flat_b = tf.reshape(x_input_b, [batch_size, self.T, self.dX])
                else:
                    x_input_flat_b = tf.reshape(x_input_b, [batch_size, self.T * self.dX])
                
                base_concat_list.append(x_input_flat_b)
            
            if self.dU > 0:
                control_mean = (np.array(params['model']['control_range']['lower']) + \
                    np.array(params['model']['control_range']['upper']))/2.
                control_width = (np.array(params['model']['control_range']['upper']) - \
                    control_mean)
                u_input_b = tf.cast((u_input_b - control_mean) / control_width, self.dtype) 
                
                if recurrent:
                    u_input_flat_b = tf.reshape(u_input_b, [batch_size, self.T, self.dU])
                else:
                    u_input_flat_b = tf.reshape(u_input_b, [batch_size, self.T * self.dU])
                
                base_concat_list.append(u_input_flat_b)
            
            for b in xrange(num_bootstrap):
                concat_list = copy.copy(base_concat_list)
                ### concatenate inputs
                with tf.name_scope('inputs_b{0}'.format(b)):
                    
                    if given_initial_states:
                        if isinstance(bootstrap_initial_states, list):
                            initial_state = bootstrap_initial_states[b] 
                        else:
                            initial_state = bootstrap_initial_states
                        if not recurrent:
                            concat_list.append(initial_state)
                    elif self.dO > 0:
                        initial_state = self.get_embedding(
                            o_input_b,
                            batch_size=batch_size,
                            reuse=reuse,
                            scope="observation_graph_b{0}".format(b))

                        if not recurrent:
                            concat_list.append(initial_state)
                    
                    if recurrent:
                        input_layer = tf.concat(2, concat_list)
                    else:
                        input_layer = tf.concat(1, concat_list)

                    if b > 0:
                        if recurrent:
                            ag_output, action_dp_masks = action_graph(
                                inputs=input_layer,
                                initial_state=initial_state,
                                params=params["model"]["action_graph"],
                                dp_masks=dp_masks,
                                dtype=self.dtype,
                                scope="action_graph_b{0}".format(b),
                                reuse=reuse)
                        else:
                            ag_output, action_dp_masks = action_graph(
                                inputs=input_layer,
                                params=params["model"]["action_graph"],
                                dp_masks=dp_masks,
                                dtype=self.dtype,
                                scope="action_graph_b{0}".format(b),
                                reuse=reuse)

                    else:
                        if recurrent:
                            ag_output, action_dp_masks = action_graph(
                                inputs=input_layer,
                                initial_state=initial_state,
                                params=params["model"]["action_graph"],
                                dtype=self.dtype,
                                scope="action_graph_b{0}".format(b),
                                reuse=reuse)
                        else:
                            ag_output, action_dp_masks = action_graph(
                                inputs=input_layer,
                                params=params["model"]["action_graph"],
                                dtype=self.dtype,
                                scope="action_graph_b{0}".format(b),
                                reuse=reuse)

                        dp_masks += action_dp_masks

                    if recurrent:
                        ag_output = tf.reshape(
                            ag_output,
                            (batch_size * self.T, int(ag_output.get_shape()[-1]))) 
                    else:
                        ag_output = tf.reshape(
                            ag_output,
                            (batch_size * self.T, int(ag_output.get_shape()[-1])/self.T)) 

                    params["model"]["output_graph"]["output_dim"] = self.doutput
                    params["model"]["output_graph"]["dropout"] = None
                    # TODO dropout acts only on output of layers
                    # Therefore, dropout should never be put on outputgraph
                    output_mat_b, _  = fcnn(
                        ag_output,
                        params["model"]["output_graph"],
                        dtype=self.dtype,
                        scope="output_graph_b{0}".format(b),
                        reuse=reuse)

                    output_mat_b = tf.reshape(output_mat_b, [batch_size, self.T, self.doutput])
                    # TODO not general because it assumes doutput = 1
                    if params["model"]["prob_coll_strictly_increasing"]:
                        output_mat_b = tf.reshape(output_mat_b, (batch_size, self.T))
                        output_mat_b = tf_utils.cumulative_increasing_sum(
                            output_mat_b,
                            self.dtype)
                        output_mat_b = tf.reshape(output_mat_b, (batch_size, self.T, self.doutput))

                    # Only care about final probability of collision
                    mask = tf.concat(0, [
                            tf.zeros((self.T - 1, self.doutput), dtype=self.dtype),
                            tf.ones((1, self.doutput), dtype=self.dtype),
                        ])
                    output_mat_b = tf.reduce_sum(mask * output_mat_b, axis=1) 
                    
                    output_pred_b = tf.sigmoid(output_mat_b, name='output_pred_b{0}'.format(b))

                bootstrap_output_mats.append(output_mat_b)
                bootstrap_output_preds.append(output_pred_b)

            ### combination of all the bootstraps
            with tf.name_scope('combine_bootstraps'):
                output_pred_mean = (1. / num_bootstrap) * tf.add_n(bootstrap_output_preds, name='output_pred_mean')
                std_normalize = (1. / (num_bootstrap - 1)) if num_bootstrap > 1 else 1
                output_pred_std = tf.sqrt(std_normalize * tf.add_n(
                    [tf.square(tf.sub(output_pred_b, output_pred_mean)) for output_pred_b in bootstrap_output_preds]))

                output_mat_mean = (1. / num_bootstrap) * tf.add_n(bootstrap_output_mats, name='output_mat_mean')
                output_mat_std = tf.sqrt(std_normalize * tf.add_n(
                    [tf.square(tf.sub(output_mat_b, output_mat_mean)) for output_mat_b in bootstrap_output_mats]))

        return output_pred_mean, output_pred_std, output_mat_mean, output_mat_std
    
    def _graph_cost(self, name, bootstrap_output_mats, bootstrap_outputs, bootstrap_lengths, reg=0.):
        with tf.name_scope(name + '_cost_and_err'):
            costs = []
            num_coll = 0
            num_errs_on_coll = 0
            num_nocoll = 0
            num_errs_on_nocoll = 0

            for b, (output_mat_b, output_b, length_b) in enumerate(zip(bootstrap_output_mats, bootstrap_outputs, bootstrap_lengths)):
               
                if params["model"]["mask"] == "last":
                    one_mask = tf.one_hot(
                        tf.cast(length_b - 1, tf.int32),
                        self.T)
                    mask = tf.stack([one_mask] * self.doutput, 2)
                elif params["model"]["mask"] == "all":
                    one_mask = tf.sequence_mask(
                        tf.cast(length_b, tf.int32),
                        maxlen=self.T,
                        dtype=self.dtype)
                    mask = tf.stack([one_mask] * self.doutput, 2)
                    one_mask = tf.one_hot(
                        tf.cast(length_b - 1, tf.int32),
                        self.T) * tf.expand_dims(tf.cast(self.T - length_b, self.dtype), axis=1)
                    mask = tf.stack([one_mask] * self.doutput, 2) * mask + mask
                else:
                    raise NotImplementedError(
                        "Mask {0} is not valid".format(
                            params["model"]["mask"]))
            
                output_b = tf.cast(output_b, self.dtype)
                output_pred_b = tf.nn.sigmoid(output_mat_b)

                ### cost
                with tf.name_scope('cost_b{0}'.format(b)):
                    cross_entropy_b = tf.nn.sigmoid_cross_entropy_with_logits(output_mat_b, output_b)
                    masked_cross_entropy_b = cross_entropy_b * mask
#                    costs.append(tf.reduce_sum(masked_cross_entropy_b) / 
#                        tf.cast(tf.reduce_sum(length_b), self.dtype))
                    costs.append(tf.reduce_mean(masked_cross_entropy_b))
                ### accuracy
                with tf.name_scope('err_b{0}'.format(b)):
                    output_geq_b = tf.cast(tf.greater_equal(output_pred_b, 0.5), self.dtype)
                    output_incorrect_b = tf.cast(tf.not_equal(output_geq_b, output_b), self.dtype)

                    ### coll err
                    output_b_coll = output_b * mask
                    num_coll += tf.reduce_sum(output_b_coll)
                    num_errs_on_coll += tf.reduce_sum(output_b_coll * output_incorrect_b)

                    ### nocoll err
                    output_b_nocoll = (1 - output_b) * mask
                    num_nocoll += tf.reduce_sum(output_b_nocoll)
                    num_errs_on_nocoll += tf.reduce_sum(output_b_nocoll * output_incorrect_b)

            with tf.name_scope('total'):
                cross_entropy = tf.reduce_mean(tf.concat(0, costs))
                weight_decay = reg * tf.add_n(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
                cost = cross_entropy + weight_decay
                err = (1. / tf.cast(num_coll + num_nocoll, self.dtype)) * (num_errs_on_coll + num_errs_on_nocoll)
                # TODO should other value be nan or 0
                err_on_coll = tf.cond(
                    num_coll > 0,
                    lambda: (1. / tf.cast(num_coll, self.dtype)) * num_errs_on_coll,
#                    lambda: tf.constant(np.nan, dtype=self.dtype))
                    lambda: tf.constant(0, dtype=self.dtype))
                err_on_nocoll = tf.cond(
                    num_nocoll > 0,
                    lambda: (1. / tf.cast(num_nocoll, self.dtype)) * num_errs_on_nocoll,
#                    lambda: tf.constant(np.nan, dtype=self.dtype))
                    lambda: tf.constant(0, dtype=self.dtype))

        return costs, weight_decay, cost, cross_entropy, err, err_on_coll, err_on_nocoll, num_coll, num_nocoll 

    def _graph_optimize(self, bootstrap_costs, reg_cost):
        # TODO should optimize each bootstrap individually
        grads = []
        optimizers = []
        vars_before = tf.global_variables()
        with tf.variable_scope('optimizer'):
            opt = tf.train.AdamOptimizer(
                learning_rate=self.learning_rate,
                beta1=self.beta1,
                beta2=self.beta2)
            grad = opt.compute_gradients(
                tf.reduce_sum(bootstrap_costs) + reg_cost)
            optimizers.append(opt.apply_gradients(grad))
            grads += grad

        vars_after = tf.global_variables()
        optimizer_vars = list(set(vars_after).difference(set(vars_before)))

        return optimizers, grads, optimizer_vars
    
    def _graph_queue_update(self):
        self._no_coll_train_fnames_ph = tf.placeholder(tf.string, [None,])
        self._coll_train_fnames_ph = tf.placeholder(tf.string, [None,]) 
        self._no_coll_val_fnames_ph = tf.placeholder(tf.string, [None,])
        self._coll_val_fnames_ph  = tf.placeholder(tf.string, [None,])
        self._queue_update = [
                tf.assign(self.d_train['no_coll_queue_var'], self._no_coll_train_fnames_ph, validate_shape=False),
                tf.assign(self.d_train['coll_queue_var'], self._coll_train_fnames_ph, validate_shape=False),
                tf.assign(self.d_val['no_coll_queue_var'], self._no_coll_val_fnames_ph, validate_shape=False),
                tf.assign(self.d_val['coll_queue_var'], self._coll_val_fnames_ph, validate_shape=False)
            ]

    def _graph_dequeue(self, no_coll_queue, coll_queue):
        return [
                no_coll_queue.dequeue_many(10*self.batch_size),
                coll_queue.dequeue_many(10*self.batch_size)
            ]

    def _graph_init_vars(self):
        self.sess.run(
            self._global_initializer,)
#            feed_dict=dict([(p, []) for p in (
#                    self.d_train['no_coll_queue_placeholder'],
#                    self.d_train['coll_queue_placeholder'],
#                    self.d_val['no_coll_queue_placeholder'],
#                    self.d_val['coll_queue_placeholder']
#                )]))

    def _graph_setup(self):
        """ Only call once """

        tf.reset_default_graph()

        self.d_train = dict()
        self.d_val = dict()
        self.d_eval = dict()

        ### prepare for training
        for i, (name, d) in enumerate((('train', self.d_train), ('val', self.d_val))):
            d['fnames'], d['X_inputs'], d['U_inputs'], d['O_inputs'], d['outputs'], d['len'], \
            queues, queue_places, queue_vars = self._graph_inputs_outputs_from_file(name)
            d['no_coll_queue'], d['coll_queue'] = queues
            d['no_coll_dequeue'], d['coll_dequeue'] = self._graph_dequeue(*queues)
            d['no_coll_queue_placeholder'], d['coll_queue_placeholder']  = queue_places
            d['no_coll_queue_var'], d['coll_queue_var'] = queue_vars
            d['output_mats'] = self._graph_inference(
                name,
                d['X_inputs'],
                d['U_inputs'],
                d['O_inputs'],
                reuse=i>0,
                tf_debug=self.tf_debug)
            d['bootstraps_cost'], d['reg_cost'], d['cost'], d['cross_entropy'], d['err'], d['err_coll'], d['err_nocoll'], d['num_coll'], d['num_nocoll'] = \
                self._graph_cost(name, d['output_mats'], d['outputs'], d['len'], reg=self.reg)
        ### optimizer
        self.d_train['optimizer'], self.d_train['grads'], self.d_train['optimizer_vars'] = \
            self._graph_optimize(self.d_train['bootstraps_cost'], self.d_train['reg_cost'])

        ### prepare for eval
        self.d_eval['X_inputs'], self.d_eval['U_inputs'], self.d_eval['O_input'] = self._graph_inputs_from_placeholders()
        self.d_eval['output_pred_mean'], self.d_eval['output_pred_std'], self.d_eval['output_mat_mean'], \
        self.d_eval['output_mat_std'] = \
            self.graph_eval_inference(
                self.d_eval['X_inputs'],
                self.d_eval['U_inputs'],
                self.d_eval['O_input'],
                reuse=True)

        ### queues
        self._graph_queue_update()
        ### initialize
        self._global_initializer = tf.global_variables_initializer()
        os.environ["CUDA_VISIBLE_DEVICES"] = str(self.device)
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=self.gpu_fraction)
        config = tf.ConfigProto(
            gpu_options=gpu_options,
            log_device_placement=False,
            allow_soft_placement=True)
        # config.intra_op_parallelism_threads = 1
        # config.inter_op_parallelism_threads = 1
        self.sess = tf.Session(config=config)
        self.coord = tf.train.Coordinator()
        self._graph_init_vars()

        # Set logs writer into folder /tmp/tensorflow_logs
        merged = tf.summary.merge_all()
        writer = tf.summary.FileWriter(
            os.path.join('/tmp', params['exp_name']),
            graph=self.sess.graph)
        self.saver = tf.train.Saver(max_to_keep=None)

    ################
    ### Training ###
    ################

    def _create_input(self, X, U, O):
        return X, U, O

    def _create_output(self, output):
        return output

    def _flush_queue(self):
        for tfrecords_fnames, dequeue in (
                    (self.tfrecords_no_coll_train_fnames, self.d_train['no_coll_dequeue']),
                    (self.tfrecords_coll_train_fnames, self.d_train['coll_dequeue']),
                    (self.tfrecords_no_coll_val_fnames, self.d_val['no_coll_dequeue']),
                    (self.tfrecords_coll_val_fnames, self.d_val['coll_dequeue'])
                ):
            while not np.all([(fname in tfrecords_fnames) for fname in
                              self.sess.run(dequeue)]):
                pass

    def train(self, reset=False, **kwargs):

        if reset:
            self._graph_init_vars()
        else:
            self.recover()
        
        num_files = len(self.tfrecords_no_coll_train_fnames)
        
        new_model_file, model_num  = self._next_model_file()
        self.sess.run(
            self._queue_update,
            {
                self._no_coll_train_fnames_ph : self.tfrecords_no_coll_train_fnames,
                self._coll_train_fnames_ph : self.tfrecords_coll_train_fnames,
                self._no_coll_val_fnames_ph : self.tfrecords_no_coll_val_fnames,
                self._coll_val_fnames_ph : self.tfrecords_coll_val_fnames
            })
        if not hasattr(self, '_queue_threads'):
            self._logger.debug('Starting queue threads')
            self._queue_threads = tf.train.start_queue_runners(sess=self.sess, coord=self.coord)
            self.threads += self._queue_threads

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

        step = 0
        epoch_start = save_start = time.time()
        while step < self.steps:
            
            new_num_files = len(self.tfrecords_no_coll_train_fnames)

            if new_num_files > num_files:
                num_files = new_num_files
        
                self.sess.run(
                    self._queue_update,
                    {
                        self._no_coll_train_fnames_ph : self.tfrecords_no_coll_train_fnames,
                        self._coll_train_fnames_ph : self.tfrecords_coll_train_fnames,
                        self._no_coll_val_fnames_ph : self.tfrecords_no_coll_val_fnames,
                        self._coll_val_fnames_ph : self.tfrecords_coll_val_fnames
                    })
                self._logger.debug('Flushing queue')
                self._flush_queue()

            ### validation
            if (step != 0 and (step % self.val_freq) == 0):
                val_values = defaultdict(list)
                val_nums = defaultdict(float)
                val_steps = 0
                self._logger.debug('\tComputing validation...')
                while val_steps < self.val_steps:
                    val_cost, val_cross_entropy, \
                    val_err, val_err_coll, val_err_nocoll, \
                    val_fnames, val_coll, val_nocoll = \
                        self.sess.run(
                            [self.d_val['cost'], self.d_val['cross_entropy'],
                            self.d_val['err'], self.d_val['err_coll'], self.d_val['err_nocoll'],
                            self.d_val['fnames'], self.d_val['num_coll'], self.d_val['num_nocoll']])

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
#                plotter.plot()

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
                
                epoch_start = time.time()

                ### save model
                self.save(new_model_file)

            ### train
            _, train_cost, train_cross_entropy, \
            train_err, train_err_coll, train_err_nocoll, \
            train_fnames, train_coll, train_nocoll = self.sess.run(
                [
                    self.d_train['optimizer'],
                    self.d_train['cost'],
                    self.d_train['cross_entropy'],
                    self.d_train['err'],
                    self.d_train['err_coll'],
                    self.d_train['err_nocoll'],
                    self.d_train['fnames'],
                    self.d_train['num_coll'],
                    self.d_train['num_nocoll']
                ])

            # Keeps track of how many times files are read
            for fname in train_fnames:
                train_fnames_dict[fname] += 1
            
            train_values['cost'].append(train_cost)
            train_values['cross_entropy'].append(train_cross_entropy)
            train_values['err'].append(train_err)
            if not np.isnan(train_err_coll): train_values['err_coll'].append(train_err_coll)
            if not np.isnan(train_err_nocoll): train_values['err_nocoll'].append(train_err_nocoll)
            train_nums['coll'] += train_coll
            train_nums['nocoll'] += train_nocoll

            # Print an overview fairly often.
            if step % self.display_batch == 0 and step > 0:
                plotter.add_train('err', step * self.batch_size, np.mean(train_values['err']))
                if len(train_values['err_coll']) > 0:
                    plotter.add_train('err_coll', step * self.batch_size, np.mean(train_values['err_coll']))
                if len(train_values['err_nocoll']) > 0:
                    plotter.add_train('err_nocoll', step * self.batch_size, np.mean(train_values['err_nocoll']))
                plotter.add_train('cost', step * self.batch_size, np.mean(train_values['cost']))
                plotter.add_train('cross_entropy', step * self.batch_size, np.mean(train_values['cross_entropy']))
#                plotter.plot()

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
                plotter.save(self._plots_dir, suffix=str(model_num))
                save_start = time.time()

            step += 1

        # Logs the number of times files were accessed
        fnames_condensed = defaultdict(int)
        for k, v in train_fnames_dict.items():
            fnames_condensed[k.split(self._hash)[0]] += v
        for k, v in sorted(fnames_condensed.items(), key=lambda x: x[1]):
            self._logger.debug('\t\t\t{0} : {1}'.format(k, v))
        
        self.save(new_model_file)
        plotter.save(self._plots_dir, suffix=str(model_num))
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
            O_input = O_input[:self.T]
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
        # want analysis_dropout for each X/U/O to be the same
        # want dropout for each num_avg to be different
        # -->
        # create num_avg different dropout for each dropout mask
        # use same dropout mask for each X/U/O
        #
        # if num_avg = 3
        # 0 1 2 0 1 2 0 1 2
        if pre_activation:
            output_pred_mean, output_pred_std = self.sess.run(
                [self.d_eval['output_mat_mean'],
                self.d_eval['output_mat_std']],
                feed_dict=feed)
        else:
            output_pred_mean, output_pred_std = self.sess.run(
                [self.d_eval['output_pred_mean'],
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

        return self.eval_batch(
            Xs,
            Us,
            Os,
            num_avg=num_avg,
            pre_activation=pre_activation)

    def eval_control_batch(self, samples, num_avg=1, pre_activation=False):
        Xs = [sample.get_X()[:self.T, self.X_idxs(sample._meta_data)]
            for sample in samples]
        Us = [sample.get_U()[:self.T, self.U_idxs(sample._meta_data)]
            for sample in samples]
        O_input = samples[0].get_O()[0, self.O_idxs(sample._meta_data)].reshape(1, -1)
        assert(not np.isnan(O_input).any())
        X_inputs, U_inputs = [], []
        for X, U in zip(Xs, Us):
            assert(len(X) == self.T)
            assert(len(U) == self.T)

            # TODO remove create_input
            X_input, U_input, _ = self._create_input(X, U, None)
            assert(not np.isnan(X_input).any())
            assert(not np.isnan(U_input).any())
            for _ in xrange(num_avg):
                X_inputs.append(X_input)
                U_inputs.append(U_input)
        feed = {}
        feed[self.d_eval['X_inputs']] = X_inputs
        feed[self.d_eval['U_inputs']] = U_inputs
        feed[self.d_eval['O_input']] = O_input
        # want dropout for each X/U/O to be the same
        # want dropout for each num_avg to be different
        # -->
        # create num_avg different dropout for each dropout mask
        # use same dropout mask for each X/U/O
        #
        # if num_avg = 3
        # 0 1 2 0 1 2 0 1 2
        if pre_activation:
            output_pred_mean, output_pred_std = self.sess.run(
                [self.d_eval['output_mat_mean'],
                self.d_eval['output_mat_std']],
                feed_dict=feed)
        else:
            output_pred_mean, output_pred_std = self.sess.run(
                [self.d_eval['output_pred_mean'],
                self.d_eval['output_pred_std']],
                feed_dict=feed)

        if num_avg > 1:
            mean, std = [], []
            for i in xrange(len(Xs)):
                mean.append(
                    output_pred_mean[i*num_avg:(i+1)*num_avg].mean(axis=0))
                std.append(
                    output_pred_std[i*num_avg:(i+1)*num_avg].mean(axis=0))

            output_pred_mean = np.array(mean)
            output_pred_std = np.array(std)

            assert(len(output_pred_mean) == len(Xs))
            assert(len(output_pred_std) == len(Xs))

        if not (output_pred_std >= 0).all():
            import ipdb; ipdb.set_trace()

        assert((output_pred_std >= 0).all())
        return output_pred_mean, output_pred_std

    #############################
    ### Load/save/reset/close ###
    #############################

    def recover(self):
        try:
            latest_file = tf.train.latest_checkpoint(
                self._checkpoints_dir)
            self.load(latest_file)
        except:
            self._logger.debug("Could not find checkpoint file")

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
