import numpy as np
import matplotlib.pyplot as plt
import copy
from collections import defaultdict

from general.world.world import World

from general.utility.utils import posquat_to_pose
from general.state_info.sample import Sample
from general.utility import transformations as tft

from general.simulation.openrave.rave_env import RaveEnv
from general.simulation.panda3d.panda3d_env import Panda3dEnv

from config import params


class WorldPointquad(World):

    def __init__(self, wp=None, **kwargs):
        World.__init__(self, wp=wp)

        self.randomize = self.wp['randomize']
        self.plot = self.wp['plot']
        self.cond_perturb = defaultdict(list)
        self.cond_forest = defaultdict(list)
        self.cond_hallway = defaultdict(list)

        view_rave = kwargs.get('view_rave', self.wp['view_rave'])
        assert(not (self.plot and view_rave))

        self.rave_env = RaveEnv(view=view_rave)

        if 'camera' in params['O']:
            self.panda_env = Panda3dEnv(params['O']['camera']['width'],
                                        params['O']['camera']['height'])
        else:
            self.panda_env = None

        # self.reset()
        if self.plot:
            self._init_visualization()

        import time; time.sleep(2.0)

    ########################
    ### Obstacle methods ###
    ########################

    def reset(self, cond=None, itr=None):
        self.clear()

        # create obstacles
        for i, obst_descr in enumerate(self.wp['obstacles']):
            if itr is not None and 'itrs' in obst_descr:
                if itr not in xrange(*obst_descr['itrs']):
                    continue

            if self.randomize and 'perturb' in obst_descr:
                perturb = np.array(obst_descr['perturb'])
                noise = np.random.uniform(-perturb, perturb)
            else:
                noise = 0

            if cond is not None:
                if i < len(self.cond_perturb[cond]):
                    noise = self.cond_perturb[cond][i]
                else:
                    self.cond_perturb[cond].append(noise)

            if obst_descr['type'] == 'box':
                pose = np.eye(4)
                pose[:3,3] = np.array(obst_descr['center']) + noise
                extents = obst_descr['extents']
                self.add_box(pose, extents)
            elif obst_descr['type'] == 'cylinder':
                pose = np.eye(4)
                pose[:3,3] = np.array(obst_descr['center']) + noise
                radius = obst_descr['radius']
                height = obst_descr['height']
                texture = obst_descr['texture']
                color = obst_descr['color']
                self.add_cylinder(pose, radius, height, texture=texture, color=color)
            elif obst_descr['type'] == 'forest':
                if cond is None or cond not in self.cond_forest.keys() or self.randomize:
                    self.cond_forest[cond] = self.create_forest(obst_descr)

                for cyl_descr in self.cond_forest[cond]:
                    pose = np.eye(4)
                    pose[:3,3] = np.array(cyl_descr['center'])
                    self.add_cylinder(pose,
                                      cyl_descr['radius'],
                                      cyl_descr['height'],
                                      texture=obst_descr['texture'],
                                      color=obst_descr['color'])

            elif obst_descr['type'] == 'hallway':
                if cond is None or cond not in self.cond_hallway.keys() or self.randomize:
                    self.cond_hallway[cond] = self.create_hallway(obst_descr)

                for wall_descr in self.cond_hallway[cond]:
                    self.add_box(wall_descr['pose'], wall_descr['extents'])

            else:
                raise Exception('Obstacle type {0} not supported'.format(obst_descr['type']))

    def load(self, rave_file, panda_file):
        self.rave_env.load_file(rave_file)
        if self.panda_env: self.panda_env.load_file(panda_file)

    def destroy(self):
        self.rave_env.destroy()

    def clear(self):
        self.rave_env.clear()
        if self.panda_env: self.panda_env.clear()

    def add_box(self, pose, extents, name=None):
        kinbody = self.rave_env.add_box(pose, extents, name=name)
        if self.panda_env: self.panda_env.add_mesh(kinbody.GetName(), *RaveEnv.get_kinbody_mesh(kinbody))

    def add_cylinder(self, pose, radius, height, name=None, texture=None, color=(1,0,0)):
        kinbody = self.rave_env.add_cylinder(pose, radius, height, name=name)
        if self.panda_env: self.panda_env.add_mesh(kinbody.GetName(),
                                                   *RaveEnv.get_kinbody_mesh(kinbody),
                                                   texture=texture,
                                                   color=color)

    def create_forest(self, forest_descr, max_time=2.):
        """
        Randomly generate forest

        :param max_time: maximum time in this method
        :return: list of cylinder descriptions
        """
        cylinders = []

        xrange = forest_descr['xrange']
        yrange = forest_descr['yrange']
        spacing = forest_descr['spacing']

        xlength = xrange[1] - xrange[0]
        ylength = yrange[1] - yrange[0]

        default_cyl = {
            'center': [0., 0., forest_descr['height']/2.],
            'radius': forest_descr['radius'],
            'height': forest_descr['height']
        }

        centers = []

        ### full random
        while len(centers) < 0.6*(xlength*ylength)/(spacing*spacing): # and time.time() - start < max_time:
            x = np.random.uniform(*xrange)
            y = np.random.uniform(*yrange)
            z = default_cyl['center'][2]

            center = np.array([x, y, z])

            if len(centers) == 0 or np.linalg.norm(center - centers, axis=1).min() > spacing:
                centers.append(center)

                new_cyl = copy.copy(default_cyl)
                new_cyl['center'] = center

                cylinders.append(new_cyl)

        dists = []
        for i, center in enumerate(centers):
            others = centers[:i] + centers[i+1:]
            dists.append(np.linalg.norm(center - others, axis=1).min())
        self._logger.info('Average spacing: {0}'.format(np.mean(dists)))

        return cylinders

    def create_hallway(self, hallway_descr):
        walls = []

        segments = hallway_descr['segments']
        delta_angle = hallway_descr['delta_angle']
        max_angle = hallway_descr['max_angle']
        width = hallway_descr['width']
        extents = hallway_descr['extents']

        default_wall = {
            'pose': np.eye(4),
            'extents': extents
        }

        wl = copy.deepcopy(default_wall)
        wl['pose'][:3,3] += np.array([0, -width/2. - extents[1], 0])
        wr = copy.deepcopy(default_wall)
        wr['pose'][:3,3] += np.array([0, width/2. + extents[1], 0])

        walls += [wl, wr]

        theta = 0.
        for segment in xrange(1, segments):
            wl_prev, wr_prev = wl, wr
            wl = copy.deepcopy(default_wall)
            wr = copy.deepcopy(default_wall)

            if segment*extents[0] < 3:
                # straight for first 3m
                theta = 0
            else:
                while True:
                    phi = np.random.uniform(-delta_angle/2., delta_angle/2.)
                    if abs(theta + phi) < max_angle:
                        theta += phi
                        break

            posel = np.copy(wl_prev['pose'])
            # move along x-axis
            posel[:3,3] += posel[:3,0]*extents[0]
            # rotate
            posel[:3,:3] = tft.euler_matrix(0, 0, theta)[:3,:3]
            # move along x-axis
            posel[:3,3] += posel[:3,0]*extents[0]
            wl['pose'] = posel

            poser = np.copy(wr_prev['pose'])
            # move along x-axis
            poser[:3,3] += poser[:3,0]*extents[0]
            # rotate
            poser[:3,:3] = tft.euler_matrix(0, 0, theta)[:3,:3]
            # move along x-axis
            poser[:3,3] += poser[:3,0]*extents[0]
            wr['pose'] = poser

            walls += [wl, wr]

        return walls

    #########################
    ### Collision methods ###
    #########################

    def is_collision(self, sample, t=None, ignore_height=True):
        if t is None:
            t = slice(0,sample._T)
        if type(t) is int:
            t = slice(t, t+1)
        t = np.r_[t.start:t.stop:t.step]

        for ti in t:
            pos = sample.get_X(t=ti, sub_state='position')
            quat = sample.get_X(t=ti, sub_state='orientation')
            pose = posquat_to_pose(pos, quat)
            if self.rave_env.is_collision(pose=pose):
                return True
            if not ignore_height and pos[-1] < 0:
                return True

        return False

    def closest_collision_distance(self, x):
        sample = Sample(meta_data=params, T=params['T'])
        pos = x[sample.get_X_idxs(sub_state='position')]
        quat = x[sample.get_X_idxs(sub_state='orientation')]
        pose = posquat_to_pose(pos, quat)

        robotpos_dist_contactpos = self.rave_env.closest_collision(pose=pose)
        if robotpos_dist_contactpos is None:
            return 1e3
        _, dist, _ = robotpos_dist_contactpos
        return dist

    #############################
    ### Visualization methods ###
    #############################

    def _init_visualization(self):
        self.fig, self.ax = plt.subplots(1, 1, figsize=(10, 10))
        # mng = plt.get_current_fig_manager()
        # mng.window.showMinimized()

        height = params['O']['camera']['height']
        width = params['O']['camera']['width']
        self.ax_im = self.ax.imshow(np.random.random((height, width)), cmap='Greys_r')

        plt.show(block=False)
        plt.pause(0.01)

    def update_visualization(self, history_sample, planned_sample, t):
        pose = posquat_to_pose(history_sample.get_X(t=t, sub_state='position'),
                               history_sample.get_X(t=t, sub_state='orientation'))
        self.panda_env.set_camera_pose(pose)
        self.panda_env.get_camera_image()

        if not self.plot:
            return

        height = params['O']['camera']['height']
        width = params['O']['camera']['width']
        curr_image = history_sample.get_O(t=t, sub_obs='camera')
        # curr_image -= curr_image.min()
        # curr_image /= curr_image.max()
        curr_image = curr_image.reshape((height, width))

        self.ax_im.set_data(curr_image)

        self.fig.canvas.update()
        self.fig.canvas.flush_events()
        plt.pause(0.01)

        # import IPython; IPython.embed()

    def get_image(self, sample):
        """
        Generates image of sample + environment
        :type sample: Sample
        :return: 2d np.ndarray
        """
        # if sample is not None:
        #     rave_env.clear_plots()
        #     for t in xrange(sample._T - 1):
        #         p_t = list(sample.get_X(t=t, sub_state='position'))
        #         p_tp1 = list(sample.get_X(t=t+1, sub_state='position'))
        #         rave_env.plot_point(p_t, color=(1,0,0), size=0.025)
        #         rave_env.plot_segment(p_t, p_tp1, color=(0,0,1))

        cam_pose = tft.euler_matrix(np.pi/2.-np.pi/10., np.pi, np.pi/2.)
        # cam_pose[:2,3] = sample.get_X(t=0, sub_state='position')
        # cam_pose[0,3] += -4.0
        # cam_pose[2,3] = 3.0
        # cam_pose = tft.euler_matrix(0, np.pi, np.pi/2.)
        # cam_pose[:3,3] = sample.get_X(t=int(sample._T/2), sub_state='position') + [0, 0, 15]
        cam_pose[:3, 3] = sample.get_X(t=0, sub_state='position') + [-1., 0, 0.5]
        # rave_env.plot_transform(cam_pose)

        try:
            viewer = self.rave_env.env.GetViewer()
            viewer.SendCommand('SetFiguresInCamera 1')
            viewer.SetCamera(cam_pose, 0.01)
            width, height = 640, 480

            I = viewer.GetCameraImage(width, height,
                                      cam_pose, [width,width,width/2.,height/2.])

            return I
        except:
            self._logger.warn('Failed to get image')
            return None

    def interact(self, init_pose=np.eye(4), step=0.1, radstep=0.1):
        pose = init_pose
        ch = None

        mapping = {
            'w': np.array([step, 0, 0, 0, 0, 0]),
            'x': np.array([-step, 0, 0, 0, 0, 0]),
            'd': np.array([0, step, 0, 0, 0, 0]),
            'a': np.array([0, -step, 0, 0, 0, 0]),
            '+': np.array([0, 0, step, 0, 0, 0]),
            '-': np.array([0, 0, -step, 0, 0, 0]),
            'p': np.array([0, 0, 0, radstep, 0, 0]),
            'o': np.array([0, 0, 0, -radstep, 0, 0]),
            'l': np.array([0, 0, 0, 0, radstep, 0]),
            'k': np.array([0, 0, 0, 0, -radstep, 0]),
            'm': np.array([0, 0, 0, 0, 0, radstep]),
            'n': np.array([0, 0, 0, 0, 0, -radstep]),
        }
        mapping = defaultdict(lambda: np.zeros(6), mapping)

        while ch != 'q':
            viewer = self.rave_env.env.GetViewer()
            if viewer is not None:
                rave_pose = pose.dot(tft.euler_matrix(np.pi / 2., np.pi, np.pi / 2.))
                viewer.SetCamera(rave_pose, 0.01)

            if self.panda_env:
                self.panda_env.set_camera_pose(pose)
                self.panda_env.get_camera_image()

            ch = utils.Getch.getch()
            pos = pose[:3, 3]
            rpy = tft.euler_from_matrix(pose)

            pos += mapping[ch][:3]
            rpy += mapping[ch][3:]

            pose = tft.euler_matrix(*rpy)
            pose[:3, 3] = pos

            print('pos: {0:.2f}, {1:.2f}, {2:.2f}'.format(*list(pos)))
            print('rpy: {0:.2f}, {1:.2f}, {2:.2f}'.format(*list(rpy)))
            print('quaternion: {0}\n'.format(tft.quaternion_from_matrix(pose)))
