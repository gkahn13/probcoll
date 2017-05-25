#!/usr/bin/env python
import rospy
import threading
import Queue
import os
import time
import numpy
import sys
import cv_bridge
import bair_car.srv
from ros_utils import ImageROSPublisher
import std_msgs.msg
import geometry_msgs.msg
from car_srv import CarSrv

from panda3d.core import loadPrcFile
from pandac.PandaModules import loadPrcFileData
#loadPrcFileData('', 'window-type offscreen')
from panda3d_camera_sensor import Panda3dCameraSensor
import direct.directbase.DirectStart
from direct.showbase.DirectObject import DirectObject
from direct.showbase.InputStateGlobal import inputState
from panda3d.core import AmbientLight
from panda3d.core import DirectionalLight
from panda3d.core import Vec3
from panda3d.core import Vec4
from panda3d.core import Point3
from panda3d.core import TransformState
from panda3d.core import BitMask32
from panda3d.core import RigidBodyCombiner, NodePath
from panda3d.core import CollisionTraverser
from panda3d.core import CollisionHandlerEvent
from panda3d.bullet import BulletGhostNode
from panda3d.bullet import BulletHelper
from panda3d.bullet import BulletWorld
from panda3d.bullet import BulletPlaneShape
from panda3d.bullet import BulletBoxShape
from panda3d.bullet import BulletRigidBodyNode
from panda3d.bullet import BulletDebugNode
from panda3d.bullet import BulletVehicle
from panda3d.bullet import ZUp
from panda3d.bullet import BulletConvexHullShape

class HallwaySrvNode(CarSrv):

    def doReset(self, pos=None, quat=None):
        self.cleanup()
        self.setup()
        if pos is None:
            rand_val = numpy.random.random() * 8 - 4.0 
            pos = (rand_val, 0.0, -0.6)
            self.load_vehicle(pos=pos)
        else:
            self.load_vehicle(pos=pos, quat=quat)

    def setup(self):
        self.worldNP = render.attachNewNode('World')

        # World
        self.debugNP = self.worldNP.attachNewNode(BulletDebugNode('Debug'))

        self.world = BulletWorld()
        self.world.setGravity(Vec3(0, 0, -9.81))
        self.world.setDebugNode(self.debugNP.node())

        # Plane
        shape = BulletPlaneShape(Vec3(0, 0, 1), 0)

        np= self.ground = self.worldNP.attachNewNode(BulletRigidBodyNode('Ground'))
        np.node().addShape(shape)
        np.setPos(0, 0, -1)
        np.setCollideMask(BitMask32.allOn())

        self.world.attachRigidBody(np.node())

        # collision
        self.maze = []
        for pos in [
                (0.0, 72.0, 0.0),
                (-11.0, 60.0, 0.0),
                (11.0, 60.0, 0.0),
                (-11.0, 48.0, 0.0),
                (11.0, 48.0, 0.0),
                (-11.0, 36.0, 0.0),
                (11.0, 36.0, 0.0),
                (-11.0, 24.0, 0.0),
                (11.0, 24.0, 0.0),
                (-11.0, 12.0, 0.0),
                (11.0, 12.0, 0.0),
                (-11.0, 0.0, 0.0),
                (11.0, 0.0, 0.0),
                (0.0, -12.0, 0.0),
                (0.5, 12.0, 1.0),
                (-0.5, 12.0, 1.0)]:
            translate = False
            if (abs(pos[0]) == 0.5):
                translate = True
                visNP = loader.loadModel('../models/ball.egg')
            else:
                visNP = loader.loadModel('../models/maze.egg')
            visNP.clearModelNodes()
            visNP.reparentTo(self.ground)
            visNP.setPos(pos[0], pos[1], pos[2])

            bodyNPs = BulletHelper.fromCollisionSolids(visNP, True);
            for bodyNP in bodyNPs:
                bodyNP.reparentTo(render)
                if translate:
                    bodyNP.setPos(pos[0], pos[1], pos[2]-1)
                else:
                    bodyNP.setPos(pos[0], pos[1], pos[2])

                if isinstance(bodyNP.node(), BulletRigidBodyNode):
                    bodyNP.node().setMass(0.0)
                    bodyNP.node().setKinematic(True)
                    self.maze.append(bodyNP)
            
        for bodyNP in self.maze:
            self.world.attachRigidBody(bodyNP.node())

if __name__ == '__main__':
    rospy.init_node('car_node', anonymous=True)
    game = HallwaySrvNode()
    base.run()

