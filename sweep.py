import argparse
import os
import yaml
from config import load_params, set_params, params
import copy
import subprocess
from pathos import multiprocessing

def build_experiments(base_params, sweep_params):
    def build_params(params, sweep_params, exp_names):
        all_params = []
        all_exp_names = [] 
        for param, exp_name in zip(params, exp_names):
            next_params = [param]
            next_exp_names = [exp_name]
            for k in param.keys():
                val = param[k]
                if val == 'sweep':
                    assert(isinstance(sweep_params[k], list))
                    updated_params = []
                    updated_exp_names = []
                    assert(len(next_params) == len(next_exp_names))
                    for next_param, next_exp_name in zip(
                            next_params,
                            next_exp_names):
                        for elem in sweep_params[k]:
                            new_param = copy.deepcopy(next_param)
                            new_param[k] = elem
                            updated_params.append(new_param)
                            if len(sweep_params[k]) > 1:
                                updated_exp_names.append(
                                    "{0}_{1}_{2}".format(
                                        next_exp_name,
                                        str(k),
                                        str(elem)))
                            else:
                                updated_exp_names.append(next_exp_name)
                    next_params = updated_params
                    next_exp_names = updated_exp_names
                elif isinstance(val, dict):
                    if k in sweep_params: 
                        new_params, new_exp_names = build_params(
                            [val],
                            sweep_params[k],
                            [""])
                        updated_params = []
                        updated_exp_names = []
                        for next_param, next_exp_name in zip(
                                next_params,
                                next_exp_names):
                            for new_param, new_exp_name in zip(
                                    new_params,
                                    new_exp_names):
                                p = copy.deepcopy(next_param)
                                p[k] = new_param
                                updated_params.append(p)
                                updated_exp_names.append(
                                    "{0}{1}".format(
                                        next_exp_name,
                                        new_exp_name))
                        next_params = updated_params
                        next_exp_names = updated_exp_names
            all_params += next_params
            all_exp_names += next_exp_names
        return all_params, all_exp_names
    params, exp_names = build_params(
        [base_params],
        sweep_params,
        [base_params['exp_name']])
    return params, exp_names 

def run_exp(args):
    try:
        exp_names = args[0]
        params = args[1]
        start_index = args[2]
        gpu = args[3]
        robot = args[4]
        for i, (param, exp_name) in enumerate(zip(params, exp_names)):
            param['model']['device'] = gpu
            param['exp_name'] = '{0:04d}_{1}'.format(i + start_index, exp_name)
            exp_dir = os.path.join(param['exp_dir'], param['exp_name'])
            if not os.path.exists(exp_dir):
                os.makedirs(exp_dir)
            exp_yaml = os.path.join(
                exp_dir,
                "params_{0}.yaml".format(robot))
            with open(exp_yaml, 'w') as yaml_file:
                yaml.dump(param, yaml_file, default_flow_style=False)

            # TODO
            subprocess.call(
                ["python", "main.py", "probcoll", robot, "-yaml", exp_yaml])
            subprocess.call(
                ["python", "main.py", "analyze", robot, "-yaml", exp_yaml])
    except KeyboardInterrupt:
        pass

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument('robot', type=str, choices=('quadrotor', 'pointquad', 'bebop2d', 'rccar', 'point2d', 'point1d'),
                           help='robot type')
    parser.add_argument('-exp_name', type=str, default=None,
                            help='experiment name')
    parser.add_argument('-base_yaml', type=str, default=None,
                           help='base yaml path relative to robot, defaults to params_<robot>_base.yaml')
    parser.add_argument('-sweep_yaml', type=str, default=None,
                           help='sweep yaml path relative to robot, defaults to params_<robot>_sweep.yaml')
    parser.add_argument('-gpus', type=list, default=[0],
                           help='list of gpus that are available')

    args = parser.parse_args()
    if args.base_yaml is None:
        base_yaml_path = os.path.join(
            os.path.dirname(__file__),
            'robots/{0}/params_{0}_base.yaml'.format(args.robot))
        sweep_yaml_path = os.path.join(
            os.path.dirname(__file__),
            'robots/{0}/params_{0}_sweep.yaml'.format(args.robot))
    else:
        base_yaml_path = os.path.join(
            os.path.dirname(__file__),
            'robots/{0}'.format(args.base_yaml))
        sweep_yaml_path = os.path.join(
            os.path.dirname(__file__),
            'robots/{0}'.format(args.sweep_yaml))
    
    base_params = {}
    sweep_params = {}
    
    with open(base_yaml_path, "r") as f:
        base_params.update(yaml.load(f))

    with open(sweep_yaml_path, "r") as f:
        sweep_params.update(yaml.load(f))

    d = base_params['exp_dir']
    base_path = os.path.join(d, os.path.split(base_yaml_path)[1])
    sweep_path = os.path.join(d, os.path.split(sweep_yaml_path)[1])
    with open(base_path, "w") as yaml_file:
        yaml.dump(base_params, yaml_file, default_flow_style=False)
    with open(sweep_path, "w") as yaml_file:
        yaml.dump(sweep_params, yaml_file, default_flow_style=False)

    params, exp_names = build_experiments(base_params, sweep_params) 
    print("Running experiments {0}".format(str(exp_names)))
    num_gpus = len(args.gpus)
    args_lists = []
    for i in range(num_gpus):
        args_list = [None, None, None, None, None]
        args_list[0] = exp_names[
            int(i/float(num_gpus)*len(exp_names)):\
            int((i+1)/float(num_gpus)*len(exp_names))]
        args_list[1] = params[
            int(i/float(num_gpus)*len(params)):\
            int((i+1)/float(num_gpus)*len(params))]
        args_list[2] = int(i/float(num_gpus) * len(params))
        args_list[3] = args.gpus[i]
        args_list[4] = args.robot
        args_lists.append(args_list)

    p = multiprocessing.Pool(num_gpus)
    p.map(run_exp, args_lists)
    p.close()
    p.join()
