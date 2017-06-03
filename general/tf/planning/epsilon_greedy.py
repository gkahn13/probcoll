import tensorflow as tf

def epsilon_greedy(action, params, dtype):
    distribution = tf.contrib.distributions.Uniform(
        params['lower'],
        params['upper'])
    cond = tf.less(tf.random_uniform([]), params['epsilon'])
    sample = tf.cast(distribution.sample(), dtype=dtype)
    eps_action = tf.cond(cond, lambda : sample, lambda : action)
    return eps_action
