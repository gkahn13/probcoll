############################
# Runs specific experiment #
############################

import argparse
import os
import numpy as np, random
from config import load_params, set_params, params

try:
    from robots.pointquad.algorithm.probcoll_pointquad import ProbcollPointquad
    from robots.pointquad.algorithm.analyze_pointquad import AnalyzePointquad
except:
    print('main.py: not importing pointquad')

try:
    from robots.bebop2d.algorithm.probcoll_bebop2d import ProbcollBebop2d
    from robots.bebop2d.algorithm.analyze_bebop2d import AnalyzeBebop2d
except:
    print('main.py: not importing Bebop2d')

try:
    from robots.rccar.algorithm.probcoll_rccar import ProbcollRCcar
except:
    print('main.py: not importing RC car')

try:
    from robots.rccar.algorithm.probcoll_rccar_remote import ProbcollRCcarRemote
except:
    print('main.py: not importing RC car remote')

try:
    from robots.sim_rccar.algorithm.probcoll_sim_rccar import ProbcollSimRCcar
    from robots.sim_rccar.analysis.analyze_sim_rccar import AnalyzeSimRCcar
    from robots.sim_rccar.analysis.train_sim_rccar import TrainSimRCcar
except:
    print('main.py: not import sim RC car')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()

    parser_probcoll = subparsers.add_parser('probcoll')
    parser_probcoll.set_defaults(run='probcoll')
    parser_analyze = subparsers.add_parser('analyze')
    parser_analyze.set_defaults(run='analyze')
    parser_replay_probcoll = subparsers.add_parser('replay_prediction')
    parser_replay_probcoll.set_defaults(run='replay_prediction')
    parser_train = subparsers.add_parser('train')
    parser_train.set_defaults(run='train')
    
    ### arguments common to all
    for subparser in (parser_probcoll, parser_analyze, parser_replay_probcoll, parser_train):
        subparser.add_argument('robot', type=str, choices=('pointquad', 'bebop2d', 'rccar_remote', 'rccar', 'sim_rccar'),
                               help='robot type')
        subparser.add_argument('-exp_name', type=str, default=None,
                                help='experiment name')
        subparser.add_argument('-yaml', type=str, default=None,
                               help='yaml path relative to robot, defaults to params_<robot>.yaml')

    parser_probcoll.add_argument('--server', type=str, default=None)
    parser_probcoll.add_argument('--username', type=str, default=None)
    parser_probcoll.add_argument('--password', type=str, default=None)

    ### replay specific arguments
    parser_replay_probcoll.add_argument('-itr', type=int, default=None,
                                          help='optionally set itr. else replay all itrs')

    ### analyze specific arguments
    parser_analyze.add_argument('--on_replay', action='store_true', help='run analyze on replay')
    parser_analyze.add_argument('--plot_single', action='store_true')
    parser_analyze.add_argument('--plot_traj', action='store_true')
    parser_analyze.add_argument('--plot_samples', action='store_true')
    parser_analyze.add_argument('--plot_groundtruth', action='store_true')
    parser_analyze.add_argument('--value_heat_map', action='store_true')
    parser_analyze.add_argument('-save_dir', type=str, default=None)

    parser_train.add_argument('--plot_dir', type=str, default=None)
    parser_train.add_argument('--data_dirs', type=str, default=None)
    parser_train.add_argument('--asynch', action='store_true')
    parser_train.add_argument('--add_data', action='store_true')

    args = parser.parse_args()
    run = args.run
    robot = args.robot
    exp_name = args.exp_name
    yaml_name = args.yaml

    print(yaml_name)
    # load yaml so all files can access
    if yaml_name is None:
        yaml_path = os.path.join(os.path.dirname(__file__), 'robots/{0}/params_{0}.yaml'.format(robot))
    else:
        if yaml_name[0] == "/":
            yaml_path = yaml_name
        else:
            yaml_path = os.path.join(os.path.dirname(__file__), 'robots/{0}'.format(robot), yaml_name)
    load_params(yaml_path)
    with open(yaml_path, 'r') as f:
        yaml_txt = ''.join(f.readlines())
    params['yaml_txt'] = yaml_txt

    np.random.seed(params['random_seed'])
    random.seed(params['random_seed'])

    if run is None:
        run = params['run'].lower()
    
    if run == 'probcoll':
        if robot == 'pointquad':
            prediction = ProbcollPointquad()
        elif robot == 'bebop2d':
            prediction = ProbcollBebop2d()
        elif robot == 'rccar':
            prediction = ProbcollRCcar(server=args.server, username=args.username, password=args.password)
        elif robot == 'rccar_remote':
            prediction = ProbcollRCcarRemote()
        elif robot == 'sim_rccar':
            prediction = ProbcollSimRCcar()
        else:
            raise Exception('Cannot run {0} for robot {1}'.format(run, robot))

        prediction.run()

    elif run == 'analyze':
        if robot == 'pointquad':
            analyze = AnalyzePointquad(on_replay=args.on_replay)
        elif robot == 'bebop2d':
            analyze = AnalyzeBebop2d(on_replay=args.on_replay)
        elif robot == 'rccar':
            analyze = AnalyzeRCcar(on_replay=args.on_replay)
        elif robot == 'sim_rccar':
            analyze = AnalyzeSimRCcar(value_heat_map=args.value_heat_map, save_dir=args.save_dir)
        else:
            raise Exception('Cannot run {0} for robot {1}'.format(run, robot))

        analyze.run()

    elif run == 'train':
        if args.data_dirs is None:
            data_dirs = args.data_dirs
        else:
            data_dirs = args.data_dirs.split()
        if robot == 'rccar':
            train = TrainRCcar(data_dirs=data_dirs, plot_dir=args.plot_dir, add_data=args.add_data, asynch=args.asynch)
        elif robot == 'sim_rccar':
            train = TrainSimRCcar(data_dirs=data_dirs, plot_dir=args.plot_dir, add_data=args.add_data, asynch=args.asynch)
        else:
            raise Exception('Cannot run {0} for robot {1}'.format(run, robot))

        train.run()

    elif run == 'replay_probcoll':
        if robot == 'pointquad':
            replay_prediction = ReplayPredictionPointquad()
        elif robot == 'bebop2d':
            replay_prediction = ReplayPredictionBebop2d()
        else:
            raise Exception('Cannot run {0} for robot {1}'.format(run, robot))

        itr = args.itr
        if itr is None:
            replay_prediction.replay()
        else:
            replay_prediction.replay_itr(itr)

    else:
        raise Exception('Action {0} not valid'.format(run))
