##########################
# Runs experiment on ec2
# assumes the probcoll code base is in rllab/sandbox/<your name>
##########################


import argparse
import os, time
import yaml
import numpy as np, random

from rllab.misc.instrument import run_experiment_lite
from rllab.misc import logger
from botocore.exceptions import ClientError

try:
    from robots.sim_rccar.algorithm.probcoll_sim_rccar import ProbcollSimRCcar
except:
    print('main.py: not import sim RC car')

def run_ec2(variant):
    from config import set_params, params
    set_params(variant)
    params['exp_dir'] = os.path.abspath(logger.get_snapshot_dir())

    np.random.seed(params['random_seed'])
    random.seed(params['random_seed'])

    if params['robot'] == 'sim_rccar':
        prediction = ProbcollSimRCcar()
    else:
        raise Exception('Cannot run {0}'.format(params['run']))

    prediction.run()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('robot', type=str, choices=('sim_rccar',))
    parser.add_argument('-mode', type=str, default='local')
    parser.add_argument('--yamls', nargs='+')
    parser.add_argument('--confirm_remote', action='store_false')
    parser.add_argument('--dry', action='store_true')
    parser.add_argument('-region', type=str, choices=('us-east-1', 'us-east-2'), default='us-east-1')
    args = parser.parse_args()

    aws_config = {
        'security_groups': ['rllab-sg'],
        'key_name': 'id_rsa',
        'instance_type': 'p2.xlarge',
        'spot_price': '0.5',
    }
    if args.region == 'us-east-1':
        aws_config.update({
            'security_groups': [],
            'key_name': 'rllab-us-east-1',
            'image_id': 'ami-36828f4d',
            'region_name': 'us-east-1',
            'subnet_id': 'subnet-941746a8',  # TODO
            'security_group_ids': ['sg-9e9e00e0']
        })
    elif args.region == 'us-east-2':
        aws_config.update({
            'security_groups': [],
            # 'key_name': 'rllab-us-east-2',
            'image_id': 'ami-c7f8dba2',
            'region_name': 'us-east-2',
            'subnet_id': 'subnet-1ebddc77',  # TODO
            'security_group_ids': ['sg-ee707e87']
        })
    else:
        raise NotImplementedError

    for yaml_name in args.yamls:
        yaml_path = os.path.abspath('robots/{0}/yamls/{1}.yaml'.format(args.robot, yaml_name))
        assert (os.path.exists(yaml_path))
        with open(yaml_path, 'r') as f:
            params = yaml.load(f)
        with open(yaml_path, 'r') as f:
            yaml_txt = ''.join(f.readlines())
        params['yaml_txt'] = yaml_txt
        params['robot'] = args.robot
        params['mode'] = args.mode

        while True:
            try:
                run_experiment_lite(
                    run_ec2,
                    snapshot_mode="all",
                    exp_name=params['exp_name'],
                    exp_prefix=params['robot'],
                    variant=params,
                    use_gpu=True,
                    use_cloudpickle=True,
                    mode=args.mode,
                    sync_s3_pkl=True,
                    aws_config=aws_config,
                    confirm_remote=args.confirm_remote,
                    dry=args.dry
                )
                break
            except ClientError as e:
                print('ClientError: {0}\nSleep for a bit and try again'.format(e))
                time.sleep(30)

