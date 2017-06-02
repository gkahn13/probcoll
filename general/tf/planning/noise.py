import tensorflow as tf

def ZeroNoise(params=None, shape=None, dtype=tf.float32):
    return tf.zeros([], dtype=dtype)

def UniformNoise(params, shape, dtype=tf.float32):
    distribution = tf.contrib.distributions.Uniform(
        params['lower'],
        params['upper'])
    return distribution.sample(sample_shape=shape, dtype=dtype)

def GaussianNoise(params, shape, dtype=tf.float32):
    distribution = tf.contrib.distributions.MultivariateNormalDiag(
       tf.zeros_like(params['std'], dtype=dtype),
       tf.constant(params['std'], dtype=dtype))
    return distribution.sample(sample_shape=shape, dtype=dtype)
