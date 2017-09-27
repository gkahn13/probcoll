import numpy as np

class SplitReplayBuffer(object):

    def __init__(self, size, T, dU, dO_im, dO_vec, doutput, partition_pct):
        self._size = size
        self._T = T
        self._act_shape = (T, dU)
        self._obs_im_shape = (dO_im,)
        self._obs_vec_shape = (dO_vec,)
        self._output_shape = (T, doutput)
        self._partition_pct = partition_pct

        # Create Buffers
        self._actions = np.ones((self._size,) + self._act_shape, dtype=np.float32) * np.nan
        self._obs_ims = np.zeros((self._size,) + self._obs_im_shape, dtype=np.uint8)
        self._obs_vecs = np.ones((self._size,) + self._obs_vec_shape, dtype=np.float32) * np.nan 
        self._outputs = np.zeros((self._size,) + self._output_shape, dtype=np.uint8)
        self._lengths = np.zeros((self._size,), dtype=np.int32)
        
        cut_off_1 = int(self._size * self._partition_pct)
        self._indices = [0, 0]
        self._curr_sizes = [0, 0]
        self._bases = [0, cut_off_1]
        self._cut_offs = [cut_off_1, self._size - cut_off_1]
        
    def __len__(self):
        return sum(self._curr_sizes)

    def add_data_point(self, u, O_im, O_vec, output, length, partition):
        index = self._indices[partition] + self._bases[partition]
        self._actions[index] = u
        self._obs_ims[index] = O_im
        self._obs_vecs[index] = O_vec
        self._outputs[index] = output
        self._lengths[index] = length
        if self._curr_sizes[partition] < self._cut_offs[partition]:
            self._curr_sizes[partition] += 1
        self._indices[partition] = ((self._indices[partition] + 1) % self._cut_offs[partition])

    def can_sample(self):
        return self._curr_sizes[0] > 0 and self._curr_sizes[1] > 0

    def sample(self, batch_size):
        batch_size_1 = int(batch_size * self._partition_pct)
        batch_size_2 = batch_size - batch_size_1
        batch_sizes = [batch_size_1, batch_size_2]
        obs_im = []
        obs_vec = []
        action = []
        output = []
        length = []
        for i in range(2):
            indices = np.random.randint(self._bases[i], self._bases[i] + self._curr_sizes[i], batch_sizes[i])
            action.append(self._actions[indices])
            obs_im.append(self._obs_ims[indices])
            obs_vec.append(self._obs_vecs[indices])
            output.append(self._outputs[indices])
            length.append(self._lengths[indices])
            assert(np.isfinite(action[-1]).all())
        action = np.concatenate(action, axis=0)
        obs_im = np.concatenate(obs_im, axis=0)
        obs_vec = np.concatenate(obs_vec, axis=0)
        output = np.concatenate(output, axis=0)
        length = np.concatenate(length, axis=0)
        return action, obs_im, obs_vec, output, length

class ReplayBuffer(object):

    def __init__(self, size, T, dU, dO_im, dO_vec, doutput):
        self._size = size
        self._T = T
        self._act_shape = (T, dU)
        self._obs_im_shape = (dO_im,)
        self._obs_vec_shape = (dO_vec,)
        self._output_shape = (T, doutput)

        # Create Buffers
        self._actions = np.ones((self._size,) + self._act_shape, dtype=np.float32) * np.nan
        self._obs_ims = np.zeros((self._size,) + self._obs_im_shape, dtype=np.uint8)
        self._obs_vecs = np.ones((self._size,) + self._obs_vec_shape, dtype=np.float32) * np.nan 
        self._outputs = np.zeros((self._size,) + self._output_shape, dtype=np.uint8)
        self._lengths = np.zeros((self._size,), dtype=np.int32)

        self._index = 0
        self._curr_size = 0
        
    def __len__(self):
        return self._curr_size

    def add_data_point(self, u, O_im, O_vec, output, length, partition):
        index = self._index
        self._actions[index] = u
        self._obs_ims[index] = O_im
        self._obs_vecs[index] = O_vec
        self._outputs[index] = output
        self._lengths[index] = length
        if self._curr_size < self._size:
            self._curr_size += 1
        self._index = (self._index + 1) % self._size

    def can_sample(self):
        return self._curr_size > 0

    def sample(self, batch_size):
        indices = np.random.randint(0, self._curr_size, batch_size)
        action = self._actions[indices]
        obs_im = self._obs_ims[indices]
        obs_vec = self._obs_vecs[indices]
        output = self._outputs[indices]
        length = self._lengths[indices]
        assert(np.isfinite(action).all())
        return action, obs_im, obs_vec, output, length
