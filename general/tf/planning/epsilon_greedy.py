import tensorflow as tf

def epsilon_greedy(action, params, eps=0.0, dtype=tf.float32):
    distribution = tf.contrib.distributions.Uniform(
        params['lower'],
        params['upper'])
    cond = tf.less(tf.random_uniform([]), eps)
    sample = tf.cast(distribution.sample(), dtype=dtype)
    eps_action = tf.cond(cond, lambda : sample, lambda : action)
    return eps_action
