from general.traj_opt.traj_opt import Trajopt

class TrajoptRCcar(Trajopt):

    def __init__(self, dynamics, world, agent):
        Trajopt.__init__(self, dynamics, world, agent)

    def plan(self, x0, init_traj=None, cost_func=None, additional_costs=[], T=None, **kwargs):
        raise NotImplementedError('TrajoptRCcar not implemented')


