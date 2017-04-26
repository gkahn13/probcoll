#!/usr/bin/env python
import rospy
import threading
import Queue
import os
import time
import numpy
import cv_bridge
import bair_car.srv
from ros_utils import ImageROSPublisher
import std_msgs.msg

from panda3d.core import loadPrcFile
from pandac.PandaModules import loadPrcFileData
#loadPrcFileData('', 'load-display tinydisplay')
loadPrcFileData('', 'window-type offscreen')

from panda3d_camera_sensor import Panda3dCameraSensor

import sys
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

class Game(DirectObject):

    def __init__(self):
        base.setBackgroundColor(0.1, 0.1, 0.8, 1)
#        base.setFrameRateMeter(True)
        assert("offscreen" == base.config.GetString("window-type", "offscreen"))

        # Light
        alight = AmbientLight('ambientLight')
        alight.setColor(Vec4(0.5, 0.5, 0.5, 1))
        alightNP = render.attachNewNode(alight)

        dlight = DirectionalLight('directionalLight')
        dlight.setDirection(Vec3(1, 1, -1))
        dlight.setColor(Vec4(0.7, 0.7, 0.7, 1))
        dlightNP = render.attachNewNode(dlight)

        render.clearLight()
        render.setLight(alightNP)
        render.setLight(dlightNP)

        # Input
        self.accept('escape', self.doExit)
        self.accept('r', self.doReset)
        self.accept('f1', self.toggleWireframe)
        self.accept('f2', self.toggleTexture)
        self.accept('f3', self.toggleDebug)
        self.accept('f5', self.doScreenshot)

        # Car Simulator
        self.dt = 0.25
        self.setup()
        
        # ROS
        self.crash_pub = rospy.Publisher('crash', std_msgs.msg.Empty, queue_size = 1)
        self.bridge = cv_bridge.CvBridge()
        self.start_update_server()
    
    # _____HANDLER_____

    def doExit(self):
        self.cleanup()
        sys.exit(1)

    def doReset(self):
        self.cleanup()
        self.setup()

    def toggleWireframe(self):
        base.toggleWireframe()

    def toggleTexture(self):
        base.toggleTexture()

    def toggleDebug(self):
        if self.debugNP.isHidden():
            self.debugNP.show()
        else:
            self.debugNP.hide()

    def doScreenshot(self):
        base.screenshot('Bullet')

    def get_ros_image(self, cv_image, image_format="rgb8"):
        return self.bridge.cv2_to_imgmsg(cv_image, image_format)
   
    def get_handler(self):
        def sim_env_handler(req):
            start_time = time.time()
            cmd_steer = req.steer
            motor = req.motor
            vel = req.vel
            reset = req.reset
            
            # If motor is default then use velocity
            if motor==0.0 and vel!=0.0:
                cmd_motor = numpy.clip(vel * 3 + 49.5, 0., 99.)
            else:
                cmd_motor = numpy.clip(motor, 0., 99.)
            
            self.steering = self.steeringClamp * \
                ((cmd_steer - 49.5) / 49.5)
            self.engineForce = self.engineClamp * \
                ((cmd_motor - 49.5) / 49.5)

            if reset:
                self.steering = 0.0       # degree
                self.engineForce = 0.0
                self.doReset()
            
            self.vehicle.setSteeringValue(self.steering, 0)
            self.vehicle.setSteeringValue(self.steering, 1)
            self.vehicle.setBrake(100.0, 2)
            self.vehicle.setBrake(100.0, 3)
            self.vehicle.applyEngineForce(self.engineForce, 2)
            self.vehicle.applyEngineForce(self.engineForce, 3)
            
            # TODO maybe change number of timesteps
            self.world.doPhysics(self.dt, 5, 0.05)
            
            # Collision detection
            result = self.world.contactTest(self.vehicle_node)
            collision = result.getNumContacts() > 0
            if collision:
                self.crash_pub.publish(std_msgs.msg.Empty())
            
            # Get observation
            obs = self.camera_sensor.observe()
            cam_image = self.get_ros_image(obs[0])
            cam_depth = self.get_ros_image(obs[1], image_format="passthrough")
            self.camera_pub.publish_image(obs[0])
            self.depth_pub.publish_image(
                obs[1],
                image_format="passthrough")
            print("Time to serve request: {}".format(time.time()-start_time))
            return [collision, cam_image, cam_depth] 
        return sim_env_handler

    def start_update_server(self):
        self.steering = 0.0       # degree
        self.steeringClamp = rospy.get_param('~steeringClamp')
        self.engineForce = 0.0
        self.engineClamp = rospy.get_param('~engineClamp')
        self.camera_pub = ImageROSPublisher("image")
        self.depth_pub = ImageROSPublisher("depth")
        s = rospy.Service('sim_env', bair_car.srv.sim_env, self.get_handler())
        rospy.spin()

    def cleanup(self):
        self.world = None
        self.worldNP.removeNode()

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
        # Chassis
        self._mass = rospy.get_param('~mass')
        #chassis_shape = rospy.get_param('~chassis_shape')
        self.load_vehicle()

    def load_vehicle(self):
        shape = BulletBoxShape(Vec3(0.6, 1.4, 0.5))
        ts = TransformState.makePos(Point3(0, 0, 0.5))

        self.vehicle_pointer = np = self.worldNP.attachNewNode(BulletRigidBodyNode('Vehicle'))
        np.node().addShape(shape, ts)
        rand_val = numpy.random.random() * 8 - 4.0 
        np.setPos(rand_val, 0.0, -0.6)
        np.node().setMass(self._mass)
        np.node().setDeactivationEnabled(False)

        first_person = rospy.get_param('~first_person')
        self.camera_sensor = Panda3dCameraSensor(
            base,
            color=True,
            depth=True,
            size=(160,90))

        self.camera_node = self.camera_sensor.cam
        if first_person:
            self.camera_node.reparentTo(np)
            self.camera_node.setPos(0.0, 1.0, 1.0)
            self.camera_node.lookAt(0.0, 6.0, 0.0)
        else:
            self.camera_node.reparentTo(np)
            self.camera_node.setPos(0.0, -10.0, 5.0)
            self.camera_node.lookAt(0.0, 5.0, 0.0)
        
        self.world.attachRigidBody(np.node())

        np.node().setCcdSweptSphereRadius(1.0)
        np.node().setCcdMotionThreshold(1e-7)

        # Vehicle
        self.vehicle_node = np.node()
        self.vehicle = BulletVehicle(self.world, np.node())
        self.vehicle.setCoordinateSystem(ZUp)
        self.world.attachVehicle(self.vehicle)

        self.yugoNP = loader.loadModel('../models/yugo/yugo.egg')
        self.yugoNP.reparentTo(np)

        self._wheels = []
        # Right front wheel
        np = loader.loadModel('../models/yugo/yugotireR.egg')
        np.reparentTo(self.worldNP)
        self.addWheel(Point3( 0.70,    1.05, 0.3), True, np)
        # Left front wheel
        np = loader.loadModel('../models/yugo/yugotireL.egg')
        np.reparentTo(self.worldNP)
        self.addWheel(Point3(-0.70,    1.05, 0.3), True, np)
        # Right rear wheel
        np = loader.loadModel('../models/yugo/yugotireR.egg')
        np.reparentTo(self.worldNP)
        self.addWheel(Point3( 0.70, -1.05, 0.3), False, np)
        # Left rear wheel
        np = loader.loadModel('../models/yugo/yugotireL.egg')
        np.reparentTo(self.worldNP)
        self.addWheel(Point3(-0.70, -1.05, 0.3), False, np)

    def addWheel(self, pos, front, np):
        wheel = self.vehicle.createWheel()

        wheel.setNode(np.node())
        wheel.setChassisConnectionPointCs(pos)
        wheel.setFrontWheel(front)

        wheel.setWheelDirectionCs(Vec3(0, 0, -1))
        wheel.setWheelAxleCs(Vec3(1, 0, 0))
        wheel.setWheelRadius(0.25)
        wheel.setMaxSuspensionTravelCm(40.0)

        wheel.setSuspensionStiffness(40.0)
        wheel.setWheelsDampingRelaxation(2.3)
        wheel.setWheelsDampingCompression(4.4)
        wheel.setFrictionSlip(1e3)
        wheel.setRollInfluence(0.0)
        self._wheels.append(np.node())

if __name__ == '__main__':
    rospy.init_node('car_node', anonymous=True)
    game = Game()
    base.run()

