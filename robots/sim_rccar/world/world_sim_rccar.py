import numpy as np
from general.world.world import World
from general.state_info.sample import Sample

from config import params

class WorldSimRCcar(World):

    def __init__(self, agent, wp=None):
        self._agent = agent
        World.__init__(self, wp=wp)

    def reset(self):
        ### back car up straight and slow
        if params['world']['do_back_up']:
            if self._agent.coll:
                sample = Sample(meta_data=params, T=1)
                sample.set_U([np.random.uniform(*self.wp['back_up']['cmd_steer'])], t=0, sub_control='cmd_steer')
                sample.set_U([self.wp['back_up']['cmd_vel']], t=0, sub_control='cmd_vel')
                u = sample.get_U(t=0)
                self._logger.info('Backing the car up')
                for _ in xrange(int(self.wp['back_up']['duration'] / params['probcoll']['dt'])): 
                    self._agent.act(u)
                    if self._agent.coll:
                        break
                for _ in xrange(int(1.0 / params['probcoll']['dt'])):
                    self._agent.act()
        else:
            self._agent.reset()

    def is_collision(self, sample, t=None):
        return self._agent.coll

    def update_visualization(self, history_sample, planned_sample, t):
        pass

    def get_image(self, sample):
        pass

    def get_info(self):
        return dict()
