import os, copy
import numpy as np

from general.utility.file_manager import FileManager
from general.traj_opt.conditions import Conditions

from general.algorithm.probcoll_model import ProbcollModel

from robots.pointquad.algorithm.prediction.probcoll_pointquad import ProbcollPointquad

from config import params, load_params

class ReplayPredictionPointquad(ProbcollPointquad):

    def __init__(self):
        np.random.seed(0)

        self._fm = FileManager(exp_folder=params['exp_folder'], read_only=True)

        yamls = [fname for fname in os.listdir(self._fm.dir) if '.yaml' in fname and '~' not in fname]
        assert (len(yamls) == 1)
        yaml_path = os.path.join(self._fm.dir, yamls[0])
        load_params(yaml_path)
        params['yaml_path'] = yaml_path

        ### turn off noise
        pred_dagger_params = params['prediction']['dagger']
        pred_dagger_params['control_noise']['type'] = 'zero'
        pred_dagger_params['epsilon_greedy'] = None

        ProbcollPointquad.__init__(self, read_only=True)

        cond_params = copy.deepcopy(pred_dagger_params['conditions'])
        cond_params['randomize_conds'] = False
        cond_params['randomize_reps'] = True
        cond_params['range']['position'] = {
            'min': [0, -0.8, 0.5], # TODO!!!!!!!!!!!!
            'max': [0, 0.8, 0.5],
            'num': [1, 50, 1]
            # 'min': [0, -0.1, 0.5],
            # 'max': [0, -0.1, 0.5],
            # 'num': [1, 1, 1]
            # 'min': [0., -3.0, 0.2],
            # 'max': [0, 3.0, 0.2],
            # 'num': [1, 50, 1]
        }
        cond_params['perturb']['position'] = [0., 0., 0.0]

        self.conditions = Conditions(cond_params=cond_params)

    #############
    ### Files ###
    #############

    def _itr_dir(self, itr, create=True):
        assert(type(itr) is int)
        dir = os.path.join(self._save_dir, 'replay/itr{0}'.format(itr))
        if not create:
            return dir
        if not os.path.exists(dir):
            os.makedirs(dir)
        return dir

    ##############
    ### Replay ###
    ##############

    def replay(self):
        itr = 0
        while self.replay_itr(itr):
            itr += 1

    def replay_itr(self, itr):
        model_file = self._itr_model_file(itr).replace('replay/', '')
        if not ProbcollModel.checkpoint_exists(model_file):
            return False

        self.logger.info('Replaying itr {0}'.format(itr))
        self.bootstrap.load(model_file=model_file)
        self._run_itr(itr)

        return True

