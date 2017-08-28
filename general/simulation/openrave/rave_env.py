import os, copy
import time
import numpy as np
import scipy.misc
import trimesh

import rll_quadrotor.utility.transformations as tft
import rll_quadrotor.utility.utils as utils
from rll_quadrotor.utility import geometry3d

import openravepy as rave
import openravepy.databases.convexdecomposition as cd
import trajoptpy
# import cloudprocpy as cp
# import trajoptpy.make_kinbodies as mk

from rll_quadrotor.config import rll_quadrotor_folder

class RaveEnv(object):
    _kinbody_num = 0

    """ OpenRave environment wrapper """
    def __init__(self, view=False):
        env_file = os.path.join(rll_quadrotor_folder,
                                'simulation/models/trajopt_quadrotor.xml') # TODO

        self.env = rave.Environment()
        physics = rave.RaveCreatePhysicsEngine(self.env,'bullet')
        self.env.SetPhysicsEngine(physics)
        self.env.StopSimulation()
        self.env.Load(env_file)

        self.kinbodies = set([b for b in self.env.GetBodies() if not b.IsRobot()])

        self.handles = list()
        self.view = view
        self.viewer = None
        if view:
            self.env.SetViewer('qtcoin')
            self.viewer = self.env.GetViewer()
            time.sleep(0.1)

        assert(len(self.env.GetRobots()) == 1)
        self.robot = self.env.GetRobots()[0]

        self.cc = trajoptpy.GetCollisionChecker(self.env)

        self.aabb_cache = dict()
        self.is_watertight_cache = dict()

    def destroy(self):
        self.env.Destroy()

    ######################
    # update environment #
    ######################

    def load_file(self, filepath):
        self.env.Load(filepath)
        self.kinbodies.update([b for b in self.env.GetBodies() if not b.IsRobot()])

    def update_local_environment(self, pos, dist):
        """
        Only keeps bodies in openrave environment that have AABB within dist of pos
        If pos or dist is None, all bodies are added back to the environment
        :param pos: robot position
        :param dist: distance
        """
        start_total = time.time()
        body_total = 0.
        rect_total = 0.

        if pos is None or dist is None:
            for b in self.kinbodies:
                if b not in self.env.GetBodies():
                    self.add(b)
        else:
            ### TODO temp
            # for b in self.env.GetBodies():
            #     if not b.IsRobot():
            #         self.env.Remove(b)

            for b in copy.copy(self.kinbodies):

                # self.add(b) # TODO temp

                if b not in self.aabb_cache:
                    self.aabb_cache[b] = b.ComputeAABB()

                aabb = self.aabb_cache[b]
                # print('{0} dist (needs o be greater than {1}'.format(np.linalg.norm(pos - aabb.pos()) - np.linalg.norm(2*aabb.extents()), dist))
                if np.linalg.norm(pos - aabb.pos()) - np.linalg.norm(2*aabb.extents()) > dist:
                    if b in self.env.GetBodies():
                        self.env.Remove(b)
                    continue
                rect_prism = geometry3d.RectangularPrism.from_aabb(aabb.pos(), aabb.extents())
                rect_prism.plot(self, color=np.random.random(3))
                # for tri in rect_prism.triangles:
                #     self.plot_triangle((tri.a, tri.b, tri.c), color=np.random.random(3))
                # for pt in [rect_prism.a0, rect_prism.b0, rect_prism.c0, rect_prism.d0,
                #             rect_prism.a1, rect_prism.b1, rect_prism.c1, rect_prism.d1]:
                #     self.plot_point(pt, size=20.)
                rect_start = time.time()
                dist_to_aabb = rect_prism.distance_to(pos)
                rect_total += time.time() - rect_start

                start_body = time.time()
                if dist_to_aabb < dist and b not in self.env.GetBodies():
                    self.add(b)
                    # print('keep')
                elif dist_to_aabb >= dist and b in self.env.GetBodies():
                    self.env.Remove(b)
                    # print('remove')
                # else:
                #     print('remove')
                body_total += time.time() - start_body

                # wldve_cont = np.linalg.norm(pos - aabb.pos()) - np.linalg.norm(2*aabb.extents()) > dist
                # print('wldve_cont {0}'.format(wldve_cont))

                # raw_input('{0}'.format(dist_to_aabb))
                # self.env.Remove(b) # TODO temp
                # self.clear_plots()


            elapsed_total = time.time() - start_total
            # print('body pct {0}'.format(body_total/elapsed_total))
            # print('rect pct {0}'.format(rect_total/elapsed_total))

            # for b in self.env.GetBodies():
            #     if not b.IsRobot():
            #         self.env.Remove(b)
            #
            # for b in copy.copy(self.kinbodies):
            #
            #     if b not in self.aabb_cache:
            #         self.aabb_cache[b] = b.ComputeAABB()
            #
            #     aabb = self.aabb_cache[b]
            #     if np.linalg.norm(pos - aabb.pos()) - np.linalg.norm(2*aabb.extents()) > dist:
            #         continue
            #     rect_prism = geometry3d.RectangularPrism.from_aabb(aabb.pos(), aabb.extents())
            #     # rect_prism.plot(self, color=np.random.random(3))
            #     # for tri in rect_prism.triangles:
            #     #     self.plot_triangle((tri.a, tri.b, tri.c), color=np.random.random(3))
            #     # for pt in [rect_prism.a0, rect_prism.b0, rect_prism.c0, rect_prism.d0,
            #     #             rect_prism.a1, rect_prism.b1, rect_prism.c1, rect_prism.d1]:
            #     #     self.plot_point(pt, size=20.)
            #     dist_to_aabb = rect_prism.distance_to(pos)
            #
            #     if dist_to_aabb < dist and b not in self.env.GetBodies():
            #         self.add(b)

    def clear(self):
        """
        Removes all user added kinbodies
        """
        for body in self.env.GetBodies():
            if not body.IsRobot():
                self.env.Remove(body)

        self.kinbodies = set()

    def add(self, kinbody):
        self.env.Add(kinbody, True)
        self.kinbodies.add(kinbody)

    def add_kinbody(self, vertices, triangles, name=None, check_collision=False):
        """
        :param vertices: list of 3d np.ndarray corresponding to points in the mesh
        :param triangles: list of 3d indices corresponding to vertices
        :param name: name of the kinbody to be added
        :param check_collision: if True, will not add kinbody if it collides with the robot
        :return False if check_collision=True and there is a collision
        """
        name = name if name is not None else 'kinbody'+str(self._kinbody_num)
        self._kinbody_num += 1

        body = rave.RaveCreateKinBody(self.env, "")
        body.InitFromTrimesh(trimesh=rave.TriMesh(vertices, triangles), draw=True)
        body.SetName(name)
        self.add(body)
        self.kinbodies.add(body)

        # randcolor = np.random.rand(3)
        randcolor = [ 0.6,  0.3,  0.] # [0., 1., 1.] # [1, 0, 0]
        body.GetLinks()[0].GetGeometries()[0].SetAmbientColor(randcolor)
        body.GetLinks()[0].GetGeometries()[0].SetDiffuseColor(randcolor)

        if check_collision:
            if self.env.CheckCollision(self.robot, body):
                self.env.Remove(body)
                self.kinbodies.difference_update([body])
                return None

        return body

    def add_box(self, pose, extents, name=None):
        """
        :param pose: 4x4 np.ndarray in frame world
        :param extents: length 3 list/np.ndarray of axis lengths
        :param name: name of kinbody to be added
        :return kinbody
        """
        name = name if name is not None else 'box'+str(self._kinbody_num)

        box = rave.KinBody.Link.GeometryInfo()
        box._type = rave.KinBody.Link.GeomType.Box
        # box._t = pose
        box._vGeomData = list(extents)
        box._vDiffuseColor = np.array([1., 0., 0.])
        kinbox = rave.RaveCreateKinBody(self.env, '')
        kinbox.InitFromGeometries([box])
        kinbox.SetTransform(pose)
        kinbox.SetName(name)

        self.add(kinbox)
        self._kinbody_num += 1

        # l = kinbox.GetLinks()[0]
        # g = l.GetGeometries()[0]
        # trimesh = l.GetCollisionData()
        # trimesh.indices = trimesh.indices
        # g.SetCollisionMesh(trimesh)
        # self.env.Add(kinbox, True)
        # self.kinbodies.add(kinbox)
        # self._kinbody_num += 1

        return kinbox

    def add_cylinder(self, pose, radius, height, name=None):
        """
        :param pose: 4x4 np.ndarray
        :return kinbody
        """
        name = name if name is not None else 'cylinder'+str(self._kinbody_num)

        cyl = rave.KinBody.Link.GeometryInfo()
        cyl._type = rave.KinBody.Link.GeomType.Cylinder
        # cyl._t = pose
        cyl._vGeomData = [radius, height]
        cyl._vDiffuseColor = np.array([0.6, 0.3, 0.])
        kincyl = rave.RaveCreateKinBody(self.env, '')
        kincyl.InitFromGeometries([cyl])
        kincyl.SetTransform(pose)
        kincyl.SetName(name)
        self.add(kincyl)
        self._kinbody_num += 1

        # to prevent extra triangles showing up in ogre .mesh
        # l = kincyl.GetLinks()[0]
        # g = l.GetGeometries()[0]
        # trimesh = l.GetCollisionData()
        # trimesh.indices = trimesh.indices[4:]
        # g.SetCollisionMesh(trimesh)
        # self.env.Add(kincyl, True)
        # self.added_kinbody_names.append(name)
        # self._kinbody_num += 1

        return kincyl

    def add_sphere(self, pos, radius, name=None):
        name = name if name is not None else 'sphere'+str(self._kinbody_num)

        kinsph = rave.RaveCreateKinBody(self.env, '')
        kinsph.InitFromSpheres(np.array([[0,0,0,radius]]))
        kinsph.SetName(name)
        self.add(kinsph)
        self.kinbodies.add(kinsph)
        self._kinbody_num += 1

        pose = np.eye(4)
        pose[:3,3] = pos
        kinsph.SetTransform(pose)

        return kinsph

    @staticmethod
    def get_kinbody_mesh(kinbody):
        """
        :return: (vertices, indices)
        """
        vertices, indices = [], []
        for link in kinbody.GetLinks():
            T_link = link.GetTransform()
            for geom in [g for g in link.GetGeometries() if g.IsModifiable()]:
                orig_tri_mesh = geom.GetCollisionMesh()
                indices += (orig_tri_mesh.indices + len(vertices)).tolist()
                v = orig_tri_mesh.vertices
                v = T_link[:3,:3].dot(v.T).T + T_link[:3,3]
                vertices += v.tolist()

        return vertices, indices

    @staticmethod
    def simplify_file(filepath):
        from sklearn import cluster

        env = rave.Environment()
        env.Load(filepath)
        env.SetViewer('qtcoin')

        ### get bodies, then remove from env
        print('Getting bodies')
        bodies = env.GetBodies()
        for body in bodies:
            env.Remove(body)

        ### get all vertices and indices
        print('Getting all vertices and indices')
        all_vertices, all_indices = [], []
        for body in bodies:
            if body.IsRobot(): continue
            for link in body.GetLinks():
                T_link = link.GetTransform()
                for geom in [g for g in link.GetGeometries() if g.IsModifiable()]:
                    orig_tri_mesh = geom.GetCollisionMesh()
                    all_indices += (orig_tri_mesh.indices + len(all_vertices)).tolist()
                    v = orig_tri_mesh.vertices
                    v = T_link[:3,:3].dot(v.T).T + T_link[:3,3]
                    all_vertices += v.tolist()
        all_indices_array = np.asarray(all_indices)

        ### kmeans on vertices
        print('KMeans')
        kmeans = cluster.MiniBatchKMeans(n_clusters=int(len(all_vertices) / 1e4),
                                verbose=0)
        kmeans.fit(all_vertices)
        all_vertices_labels = np.copy(kmeans.labels_)


        print('Creating clusters')
        num_clusters = np.max(all_vertices_labels) + 1
        indices_clusters = [[] for _ in range(num_clusters)]
        vertices_clusters = [[] for _ in range(num_clusters)]
        added_indices = set() # keep track of triangles that have already been added to a cluster
        for c in range(num_clusters): # for each cluster
            print('Cluster {0} / {1}'.format(c+1, num_clusters))

            c_vertices_indices = (all_vertices_labels == c).nonzero()[0] # all vertices indices in cluster

            total_nonzero, total_indices = 0, 0
            c_indices = [] # triangles involved in this cluster
            for index in c_vertices_indices.flatten():
                start_nonzero = time.time()


                # indices_with_index = (all_indices_array[:,0] == index).nonzero()[0]

                # indices_with_index = (all_indices_array == index).max(axis=1).nonzero()[0] # which triangle contain index

                indices_with_index = set()
                for i in range(3):
                    indices_with_index_i = (all_indices_array[:,i] == index).nonzero()[0]
                    indices_with_index.update(indices_with_index_i.tolist())
                indices_with_index = list(indices_with_index)

                total_nonzero += time.time() - start_nonzero
                start_indices = time.time()
                # only add triangle to one cluster
                for indice in all_indices_array[indices_with_index].tolist():
                    indice_sorted = sorted(indice)
                    if tuple(indice_sorted) not in added_indices:
                        c_indices.append(indice)
                        added_indices.add(tuple(indice_sorted))
                total_indices += time.time() - start_indices

            ### the indices need to be made starting from 0 and sequential
            c_num_vertices = len(set(np.asarray(c_indices).flatten()))
            c_remapping = dict([(old, new) for new, old in enumerate(set(np.asarray(c_indices).flatten()))])
            c_indices_remapped = [] # new triangle indexing
            for indice in c_indices:
                c_indices_remapped.append([c_remapping[old_index] for old_index in indice])
            c_vertices_remapped = [None for _ in range(c_num_vertices)] # new vertex ordering
            for index in set(np.asarray(c_indices).flatten()):
                assert(c_vertices_remapped[c_remapping[index]] is None)
                c_vertices_remapped[c_remapping[index]] = all_vertices[index]
            assert(None not in c_vertices_remapped)
            assert(np.max(c_indices_remapped) < len(c_vertices_remapped))
            assert(len(set(np.asarray(c_indices_remapped).flatten().tolist())) == len(c_vertices_remapped))

            indices_clusters[c] = c_indices_remapped
            vertices_clusters[c] = c_vertices_remapped


        ### create new rave object
        print('Creating openrave objects')
        for c, (indices_c, vertices_c) in enumerate(zip(indices_clusters, vertices_clusters)):
            name = 'cluster_{0}'.format(c)
            body = rave.RaveCreateKinBody(env, "")
            body.InitFromTrimesh(trimesh=rave.TriMesh(vertices_c, indices_c), draw=True)
            body.SetName(name)
            env.Add(body)

            randcolor = np.random.rand(3)
            # randcolor = [ 0.6,  0.3,  0.] # [0., 1., 1.] # [1, 0, 0]
            body.GetLinks()[0].GetGeometries()[0].SetAmbientColor(randcolor)
            body.GetLinks()[0].GetGeometries()[0].SetDiffuseColor(randcolor)

        dirname = os.path.dirname(filepath)
        basename = os.path.basename(filepath)
        basename_simplified = basename.split('.')[0] + '_simplified.zae'
        filename_simplified = os.path.join(dirname, basename_simplified)
        env.Save(filename_simplified, rave.Environment.SelectionOptions.Everything)

    def is_watertight(self, kinbody):
        if kinbody.GetName() in self.is_watertight_cache:
            return self.is_watertight_cache[kinbody.GetName()]

        for link in kinbody.GetLinks():
            for geom in link.GetGeometries():
                mesh = geom.GetCollisionMesh()
                tm_mesh = trimesh.Trimesh(mesh.vertices, mesh.indices)
                if not tm_mesh.is_watertight:
                    self.is_watertight_cache[kinbody.GetName()] = False
                    return False

        self.is_watertight_cache[kinbody.GetName()] = True
        return True


    #######################
    # collision detection #
    #######################

    def is_collision(self, pose=None):
        """
        TODO: will not work if bodies are not watertight
        :param pose: 4x4 np.ndarray
        :return: True if in collision with env
        """

        if pose is not None:
            self.robot.SetTransform(pose)

        self.cc.SetContactDistance(0.0)
        bodies = [b for b in self.env.GetBodies() if not b.IsRobot()]
        cols = self.cc.BodiesVsBodies([self.robot], bodies)
        # cols = self.cc.BodyVsAll(self.robot)

        for col in cols:
            if col.GetDistance() < 0:
                return True
        return False

        return len(cols) > 0

    def closest_collision(self, pose=None, plot=False, contact_dist=1e3):
        """
        Finds nearest collision
        :param pose: 4x4 np.ndarray
        :return: contact on robot, distance, contact in env
        """
        if pose is not None:
            self.robot.SetTransform(pose)

        self.cc.SetContactDistance(contact_dist)
        bodies = [b for b in self.env.GetBodies() if not b.IsRobot()]
        cols = self.cc.BodiesVsBodies([self.robot], bodies)
        # cols = self.cc.BodyVsAll(self.robot)

        if len(cols) == 0:
            return None

        contact_pos, min_dist, obstacle_pos, diff = None, np.inf, None, None
        for col in cols:
            dist = col.GetDistance()
            col_kinbody = self.env.GetKinBody(col.GetLinkBParentName())

            dist_was_negative = dist < 0
            if not self.is_watertight(col_kinbody):
                dist = abs(dist)

            if dist < min_dist:
                min_dist = dist
                contact_pos, obstacle_pos = col.GetPtA(), col.GetPtB()
                diff = contact_pos - obstacle_pos
                if dist_was_negative:
                    diff *= -1

        if plot:
            self.plot_point(contact_pos, color=(1,0,0))
            self.plot_point(obstacle_pos, color=(0,0,1))
            self.plot_segment(contact_pos, obstacle_pos, color=(1,1,0))

        return contact_pos, dist, obstacle_pos, diff


    ############
    # plotting #
    ############

    def set_camera_pose(self, pose, focal_length=0.01):
        local_pose = tft.euler_matrix(-np.pi/2., 0., -np.pi/2.)
        self.viewer.SendCommand('SetFiguresInCamera 1')
        self.viewer.SetCamera(pose.dot(local_pose), focal_length)

    def clear_plots(self, num_to_clear=-1):
        """
        :param num_to_clear: if num_to_clear < 0, clear all plots, else clear num_to_clear
        """
        if num_to_clear < 0:
            self.handles = list()
        else:
            self.handles = self.handles[:-int(min(num_to_clear, len(self.handles)))]

    def plot_point(self, pos_array, size=.01, color=(0,1,0)):
        """
        :param pos_array: 3d np.array
        :param size: radius in meters
        :param color: rgb [0,1], default green
        """
        self.handles += [self.env.plot3(points=pos_array,
                                        pointsize=size,
                                        colors=np.array(color),
                                        drawstyle=1)]

    def plot_segment(self, start, end, color=(1,0,0), linewidth=3.0):
        """
        :param start: 3d np.array
        :param end: 3d np.array
        :param color: rgb [0,1], default red
        """
        start = np.array(start)
        end = np.array(end)

        self.handles += [self.env.drawlinestrip(points=np.array([start, end]),
                                                linewidth=linewidth, colors=np.array([color,color]))]

    def plot_triangle(self, points, color=(1,0,0), alpha=1.):
        """
        :param points: length 3 list of 3d list/np.array
        :param color: (r,g,b) [0,1]
        :param alpha: [0,1]
        """
        self.handles += [self.env.drawtrimesh(points=np.vstack(points),
                                              indices=None,
                                              colors=np.array(color+(alpha,)))]

    def plot_transform(self, T, s=0.1):
        """
        :param T: 4x4 np.array
        :param s: length of axes in meters
        """
        T = np.array(T)
        x = T[0:3,0]
        y = T[0:3,1]
        z = T[0:3,2]
        o = T[0:3,3]
        self.handles.append(self.env.drawlinestrip(points=np.array([o, o+s*x]), linewidth=3.0, colors=np.array([(1,0,0),(1,0,0)])))
        self.handles.append(self.env.drawlinestrip(points=np.array([o, o+s*y]), linewidth=3.0, colors=np.array(((0,1,0),(0,1,0)))))
        self.handles.append(self.env.drawlinestrip(points=np.array([o, o+s*z]), linewidth=3.0, colors=np.array(((0,0,1),(0,0,1)))))

    def plot_arrow(self, T, color=(1,0,0), s=0.1):
        T = np.array(T)
        x = T[0:3,0]
        o = T[0:3,3]

        delta = x
        delta *= s/np.linalg.norm(delta)
        self.handles.append(self.env.drawlinestrip(points=np.array([o, o+delta]),
                                                   linewidth=3.0, colors=np.array([color,color])))
        self.plot_point(o+delta, size=0.1*s, color=color)


    def save_view(self, file_name):
        """
        :param file_name: path string
        """
        self.env.GetViewer().SendCommand('SetFiguresInCamera 1') # also shows the figures in the image
        I = self.env.GetViewer().GetCameraImage(640,480,  self.env.GetViewer().GetCameraTransform(),[640,640,320,240])
        scipy.misc.imsave(file_name, I)
        self.env.GetViewer().SendCommand('SetFiguresInCamera 0')
