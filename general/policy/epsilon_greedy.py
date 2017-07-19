import tensorflow as tf

def epsilon_greedy(action_fn, lower, upper, eps=0.0, dtype=tf.float32):
    cond = tf.less(tf.random_uniform([]), eps)
    def sample():
        distribution = tf.contrib.distributions.Uniform(lower, upper)
        return tf.cast(distribution.sample(), dtype=dtype)
    eps_action = tf.cond(cond, sample, action_fn)
    return eps_action
