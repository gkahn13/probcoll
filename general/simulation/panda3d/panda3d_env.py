# python makepanda/makepanda.py --everything --no-artoolkit --no-fcollada --no-fmodex --no-opencv --no-squish --no-vrpn --no-rocket --no-fftw --no-ffmpeg --threads 6 --installer

from PIL import Image
import StringIO

import numpy as np
import cv2

from panda3d.core import *
from pandac.PandaModules import GeomVertexFormat, GeomVertexData, GeomVertexWriter, GeomTriangles, Geom, GeomNode, NodePath, GeomPoints
from direct.showbase.ShowBase import ShowBase
from direct.task import Task
from direct.stdpy import threading
from direct.filter.FilterManager import FilterManager

from rll_quadrotor.utility import utils
from rll_quadrotor.utility import transformations as tft

class Panda3dEnv(ShowBase):
    """
    Graphics renderer
    """

    def __init__(self, width, height):
        ShowBase.__init__(self)

        ### image size
        # props = WindowProperties()
        # props.setSize(width, height)
        # self.win.requestProperties(props)
        self.width = width
        self.height = height

        ### FOV
        self.camLens.setFov(100) # TODO: make parameter
        self.camLens.setNear(0.1)
        self.camLens.setFar(1000)

        self.screenshot = PNMImage()
        self.dr = self.camNode.getDisplayRegion(0)

        self.origin = tft.euler_matrix(-np.pi, -np.pi, np.pi/2.)

        # self.render.setAntialias(AntialiasAttrib.MPolygon)
        self.setBackgroundColor(1, 1, 1)

        # plight = PointLight('plight')
        # plight.setAttenuation((1, 0, 1))
        # plight.setColor(VBase4(0.2, 0.2, 0.2, 1))
        # plnp = self.render.attachNewNode(plight)
        # plnp.setPos(10, 20, 0)
        # self.render.setLight(plnp)

    ######################
    ### Camera methods ###
    ######################

    def set_camera_pos(self, pos):
        T = np.eye(4)
        T[:3,3] = pos
        pos = T.dot(self.origin)[:3,3]
        self.cam.set_pos(*list(pos))

    def set_camera_quat(self, quat):
        T = tft.quaternion_matrix(quat).dot(self.origin)
        quat = tft.quaternion_from_matrix(T)
        self.cam.set_quat(LQuaternionf(*list(quat)))

    def set_camera_pose(self, pose):
        pos, quat = utils.pose_to_posquat(pose)
        self.set_camera_pos(pos)
        self.set_camera_quat(quat)

    def get_camera_image(self, grayscale=True):
        ### get screenshot
        self.taskMgr.step()
        self.taskMgr.step()
        self.dr.getScreenshot(self.screenshot)

        ### get screenshot bytes
        ss = StringStream()
        self.screenshot.write(ss, 'jpeg')
        bytes = ss.getData()

        ### create image
        tempBuff = StringIO.StringIO()
        tempBuff.write(bytes)
        tempBuff.seek(0) #need to jump back to the beginning before handing it off to PIL
        im = np.array(Image.open(tempBuff))

        if grayscale:
            def rgb2gray(rgb):
                return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])
            im = rgb2gray(im)
            im = cv2.resize(im, (self.height, self.width), interpolation=cv2.INTER_AREA)
        else:
            for i in range(3):
                im[:,:,i] = cv2.imresize(im[:,:,i], (self.height, self.width), interpolation=cv2.INTER_AREA)

        im /= 255.

        return im

    ####################
    ### Mesh methods ###
    ####################

    def add_mesh(self, name, vertices, faces, color=(1,0,0), texture=None):
        assert(self.render.find(name).isEmpty()) # name is unique

        format = GeomVertexFormat.getV3()
        vertex_data = GeomVertexData("Data", format, Geom.UHStatic)
        vertex_writer = GeomVertexWriter(vertex_data, "vertex")

        for vertex in vertices:
            vertex_writer.addData3f(*list(vertex))

        triangles = GeomTriangles(Geom.UHStatic)
        for face in faces:
            triangles.addVertices(*list(face))
            triangles.addVertices(*list(face)[::-1])
        triangles.closePrimitive()

        geom = Geom(vertex_data)
        geom.addPrimitive(triangles)

        node = GeomNode(name)
        node.addGeom(geom)
        node_path = NodePath(node)

        node_path.reparentTo(self.render)

        if texture is not None:
            node_path.setTexGen(TextureStage.getDefault(), TexGenAttrib.MEyeSphereMap)
            node_path.setTexture(self.loader.loadTexture(texture))
        else:
            node_path.setColor(*list(color))

        return node_path

    def load_file(self, filepath):
        scene = self.loader.loadModel(filepath)
        scene.reparentTo(self.render)

    def remove_mesh(self, name):
        node_path = self.render.find(name)
        assert(not node_path.is_empty())
        node_path.removeNode()

    def clear(self):
        """ Removes everything """
        for node_path in self.render.get_children():
            if node_path.get_name() == 'camera':
                continue

            node_path.removeNode()
