############################
# Runs specific experiment #
############################

import argparse
import os
import numpy as np, random

from config import load_params, set_params, params

try:
    from robots.pointquad.algorithms.prediction.dagger_prediction_pointquad import DaggerPredictionPointquad
    from robots.pointquad.algorithms.prediction.analyze_prediction_pointquad import AnalyzePredictionPointquad
    from robots.pointquad.algorithms.prediction.replay_prediction_pointquad import ReplayPredictionPointquad
except:
    print('main.py: not importing pointquad')

try:
    from robots.bebop2d.algorithms.prediction.dagger_prediction_bebop2d import DaggerPredictionBebop2d
    from robots.bebop2d.algorithms.prediction.replay_prediction_bebop2d import ReplayPredictionBebop2d
    from robots.bebop2d.algorithms.prediction.analyze_prediction_bebop2d import AnalyzePredictionBebop2d
except:
    print('main.py: not importing Bebop2d')

from robots.rccar.algorithms.prediction.dagger_prediction_rccar import DaggerPredictionRCcar
from robots.rccar.algorithms.prediction.analyze_prediction_rccar import AnalyzePredictionRCcar

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()

    parser_gps = subparsers.add_parser('gps')
    parser_gps.set_defaults(run='gps')
    parser_dagger = subparsers.add_parser('dagger')
    parser_dagger.set_defaults(run='dagger')
    parser_replay = subparsers.add_parser('replay')
    parser_replay.set_defaults(run='replay')
    parser_prediction = subparsers.add_parser('prediction')
    parser_prediction.set_defaults(run='prediction')
    parser_analyze_prediction = subparsers.add_parser('analyze_prediction')
    parser_analyze_prediction.set_defaults(run='analyze_prediction')
    parser_replay_prediction = subparsers.add_parser('replay_prediction')
    parser_replay_prediction.set_defaults(run='replay_prediction')

    ### arguments common to all
    for subparser in (parser_gps, parser_dagger, parser_replay, parser_prediction,
                      parser_analyze_prediction, parser_replay_prediction):
        subparser.add_argument('robot', type=str, choices=('quadrotor', 'pointquad', 'bebop2d', 'rccar', 'point2d', 'point1d'),
                               help='robot type')
        subparser.add_argument('-exp_folder', type=str, default=None,
                                help='experiment folder')
        subparser.add_argument('-yaml', type=str, default=None,
                               help='yaml path relative to robot, defaults to params_<robot>.yaml')

    ### replay specific arguments
    parser_replay.add_argument('-itr', type=int, default=None,
                               help='optionally set itr. else replay all itrs')
    parser_replay_prediction.add_argument('-itr', type=int, default=None,
                                          help='optionally set itr. else replay all itrs')

    ### prediction specific arguments

    ### analyze_prediction specific arguments
    parser_analyze_prediction.add_argument('--on_replay', action='store_true', help='run analyze on replay')
    parser_analyze_prediction.add_argument('--plot_single', action='store_true')
    parser_analyze_prediction.add_argument('--plot_traj', action='store_true')
    parser_analyze_prediction.add_argument('--plot_samples', action='store_true')
    parser_analyze_prediction.add_argument('--plot_groundtruth', action='store_true')

    args = parser.parse_args()
    run = args.run
    robot = args.robot
    exp_folder = args.exp_folder
    yaml = args.yaml

    # load yaml so all files can access
    if yaml is None:
        yaml_path = os.path.join(os.path.dirname(__file__), 'robots/{0}/params_{0}.yaml'.format(robot))
    else:
        yaml_path = os.path.join(os.path.dirname(__file__),
                                 'robots/{0}'.format(robot),
                                 yaml)
    load_params(yaml_path)
    params['yaml_path'] = yaml_path

    np.random.seed(params['random_seed'])
    random.seed(params['random_seed'])

    if run is None:
        run = params['run'].lower()

    if exp_folder is None:
        exp_folder = params['exp_folder']
    else:
        params['exp_folder'] = exp_folder

    if run == 'gps':
        if robot == 'quadrotor':
            alg = GPSQuadrotor()
        else:
            raise Exception('Cannot run {0} for robot {1}'.format(run, robot))

        alg.solve()

    elif run == 'replay':
        if robot == 'quadrotor':
            replay = ReplayQuadrotor(exp_folder)
        elif robot == 'point2d':
            replay = ReplayPoint2d(exp_folder)
        else:
            raise Exception('Cannot run {0} for robot {1}'.format(run, robot))

        itr = args.itr
        if itr is None:
            replay.run_all()
        else:
            replay.run(itr)

    elif run == 'dagger':
        if robot == 'quadrotor':
            dagger = DaggerQuadrotor()
        elif robot == 'point2d':
            dagger = DaggerPoint2d()
        else:
            raise Exception('Cannot run {0} for robot {1}'.format(run, robot))

        dagger.solve()

    elif run == 'prediction':
        if robot == 'quadrotor':
            prediction = DaggerPredictionQuadrotor()
        elif robot == 'pointquad':
            prediction = DaggerPredictionPointquad()
        elif robot == 'bebop2d':
            prediction = DaggerPredictionBebop2d()
        elif robot == 'rccar':
            prediction = DaggerPredictionRCcar()
        elif robot == 'point2d':
            prediction = DaggerPredictionPoint2d()
        elif robot == 'point1d':
            prediction = DaggerPredictionPoint1d()
        else:
            raise Exception('Cannot run {0} for robot {1}'.format(run, robot))

        prediction.run()

    elif run == 'analyze_prediction':
        if robot == 'pointquad':
            analyze_prediction = AnalyzePredictionPointquad(on_replay=args.on_replay)
        elif robot == 'bebop2d':
            analyze_prediction = AnalyzePredictionBebop2d(on_replay=args.on_replay)
        elif robot == 'rccar':
            analyze_prediction = AnalyzePredictionRCcar(on_replay=args.on_replay)
        elif robot == 'point2d':
            analyze_prediction = AnalyzePredictionPoint2d()
        elif robot == 'point1d':
            analyze_prediction = AnalyzePredictionPoint1d()
        else:
            raise Exception('Cannot run {0} for robot {1}'.format(run, robot))

        analyze_prediction.run(args.plot_single, args.plot_traj, args.plot_samples, args.plot_groundtruth)

    elif run == 'replay_prediction':
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
