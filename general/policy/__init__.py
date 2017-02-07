from .lr.policy_lr_prior import PolicyLRPrior
from .lr.policy_prior_gmm import PolicyPriorGMM
try:
    from .caffe.caffe_policy import CaffePolicy
    from .caffe.policy_opt_caffe import PolicyOptCaffe
except:
    print('Could not import caffe')
from .cgt.cgt_policy import CGTPolicy
from .cgt.policy_opt_cgt import PolicyOptCGT


def _import_helper(module):
    if module.lower() == 'caffe':
        return dict(policy=CaffePolicy, opt=PolicyOptCaffe)
    elif module.lower() == 'cgt':
        return dict(policy=CGTPolicy, opt=PolicyOptCGT)
    elif module.lower() == 'theano':
        raise NotImplementedError('todo')
    else:
        raise ValueError('unknown module type')


def get_policy_class(module):
    return _import_helper(module)['policy']


def get_opt_class(module):
    return _import_helper(module)['opt']