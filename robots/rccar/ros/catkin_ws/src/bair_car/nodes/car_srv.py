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
import abc

from panda3d.core import loadPrcFile
from pandac.PandaModules import loadPrcFileData
loadPrcFileData('', 'window-type offscreen')
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

class CarSrv(DirectObject):

    def __init__(self):
        base.setBackgroundColor(0.1, 0.1, 0.8, 1)
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

        self.params = rospy.get_param('~sim')
        # Car Simulator
        self.dt = rospy.get_param('~dt')
        self.setup()
        self.load_vehicle()
        
        # ROS
        self.crash_pub = rospy.Publisher('crash', std_msgs.msg.Empty, queue_size = 1)
        self.bridge = cv_bridge.CvBridge()
#        taskMgr.add(self.update, 'updateWorld')
        self.start_update_server()
    
    # _____HANDLER_____

    def doExit(self):
        self.cleanup()
        sys.exit(1)

    def doReset(self, pos=None, quat=None):
        self.cleanup()
        self.setup()
        if pos is None or quat is None:
            self.load_vehicle()
        else:
            self.load_vehicle(pos=pos, quat=quat)

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
            pose = req.pose 
            # If motor is default then use velocity
            if motor==0.0:
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
                pos = pose.position.x, pose.position.y, pose.position.z
                quat = pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w
                if numpy.all(numpy.array(pos) == 0.):
                    self.doReset()
                else:
                    self.doReset(pos=pos, quat=quat)

            self.vehicle.setSteeringValue(self.steering, 0)
            self.vehicle.setSteeringValue(self.steering, 1)
            self.vehicle.setBrake(100.0, 2)
            self.vehicle.setBrake(100.0, 3)
            self.vehicle.applyEngineForce(self.engineForce, 2)
            self.vehicle.applyEngineForce(self.engineForce, 3)
           
            pos = numpy.array(self.vehicle_pointer.getPos())
            np_quat = self.vehicle_pointer.getQuat()
            quat = numpy.array(np_quat)
            self.previous_pos = pos
            self.previous_quat = np_quat

            # TODO maybe change number of timesteps
            self.world.doPhysics(self.dt, 10, 0.05)
            # Collision detection
            result = self.world.contactTest(self.vehicle_node)
            collision = result.getNumContacts() > 0
            
            if collision:
                # TODO figure out why this causes problems
#                self.crash_pub.publish(std_msgs.msg.Empty())
                self.doReset(pos=self.previous_pos, quat=self.previous_quat)

            state = geometry_msgs.msg.Pose()
            pos = numpy.array(self.vehicle_pointer.getPos())
            np_quat = self.vehicle_pointer.getQuat()
            quat = numpy.array(np_quat)
            self.previous_pos = pos
            self.previous_quat = np_quat
            state.position.x, state.position.y, state.position.z = pos
            state.orientation.x, state.orientation.y, \
                    state.orientation.z, state.orientation.w = quat
            
            # Get observation
            obs = self.camera_sensor.observe()
            back_obs = self.back_camera_sensor.observe()
            cam_image = self.get_ros_image(obs[0])
            cam_depth = self.get_ros_image(obs[1], image_format="passthrough")
            self.camera_pub.publish_image(obs[0])
            self.depth_pub.publish_image(
                obs[1],
                image_format="passthrough")
            back_cam_image = self.get_ros_image(back_obs[0])
            back_cam_depth = self.get_ros_image(back_obs[1], image_format="passthrough")
            self.back_camera_pub.publish_image(back_obs[0])
            self.back_depth_pub.publish_image(
                back_obs[1],
                image_format="passthrough")
            return [collision, cam_image, cam_depth, back_cam_image, back_cam_depth, state] 
        return sim_env_handler

    def load_vehicle(self, pos=(0.0, -20.0, -0.6), quat=None):
        # Chassis
        self._mass = self.params['mass']
        #chassis_shape = self.params['chassis_shape']
        shape = BulletBoxShape(Vec3(0.6, 1.4, 0.5))
        ts = TransformState.makePos(Point3(0, 0, 0.5))

        self.vehicle_pointer = self.worldNP.attachNewNode(BulletRigidBodyNode('Vehicle'))
        self.vehicle_node = self.vehicle_pointer.node()
        self.vehicle_node.addShape(shape, ts)
        self.previous_pos = pos
        self.vehicle_pointer.setPos(pos[0], pos[1], pos[2])
        if quat is not None:
            self.vehicle_pointer.setQuat(quat)
        self.previous_quat = self.vehicle_pointer.getQuat()
        self.vehicle_node.setMass(self._mass)
        self.vehicle_node.setDeactivationEnabled(False)

#        first_person = self.params['first_person']
        self.camera_sensor = Panda3dCameraSensor(
            base,
            color=True,
            depth=True,
            size=(160,90))

        self.camera_node = self.camera_sensor.cam
#        if first_person:
#            self.camera_node.setPos(0.0, 1.0, 1.0)
#            self.camera_node.lookAt(0.0, 6.0, 0.0)
#        else:
#            self.camera_node.setPos(0.0, -10.0, 5.0)
#            self.camera_node.lookAt(0.0, 5.0, 0.0)

        self.camera_node.reparentTo(self.vehicle_pointer)
        self.camera_node.setPos(0.0, 1.0, 1.0)
        self.camera_node.lookAt(0.0, 6.0, 0.0)

        self.back_camera_sensor = Panda3dCameraSensor(
            base,
            color=True,
            depth=True,
            size=(160,90))

        self.back_camera_node = self.back_camera_sensor.cam
#        if first_person:
#            self.camera_node.setPos(0.0, 1.0, 1.0)
#            self.camera_node.lookAt(0.0, 6.0, 0.0)
#        else:
#            self.camera_node.setPos(0.0, -10.0, 5.0)
#            self.camera_node.lookAt(0.0, 5.0, 0.0)

        self.back_camera_node.reparentTo(self.vehicle_pointer)
        self.back_camera_node.setPos(0.0, -1.0, 1.0)
        self.back_camera_node.lookAt(0.0, -6.0, 0.0)

        self.world.attachRigidBody(self.vehicle_node)

        self.vehicle_node.setCcdSweptSphereRadius(1.0)
        self.vehicle_node.setCcdMotionThreshold(1e-7)

        # Vehicle
        self.vehicle = BulletVehicle(self.world, self.vehicle_node)
        self.vehicle.setCoordinateSystem(ZUp)
        self.world.attachVehicle(self.vehicle)

        self.yugoNP = loader.loadModel('../models/yugo/yugo.egg')
        self.yugoNP.reparentTo(self.vehicle_pointer)

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

    def start_update_server(self):
        self.steering = 0.0       # degree
        self.steeringClamp = self.params['steeringClamp']
        self.engineForce = 0.0
        self.engineClamp = self.params['engineClamp']
        self.camera_pub = ImageROSPublisher("image")
        self.depth_pub = ImageROSPublisher("depth")
        self.back_camera_pub = ImageROSPublisher("back_image")
        self.back_depth_pub = ImageROSPublisher("back_depth")
        s = rospy.Service('sim_env', bair_car.srv.sim_env, self.get_handler())
        rospy.spin()

    def update(self, task):
        dt = globalClock.getDt()

        self.world.doPhysics(dt, 10, 0.008)
        obs = self.camera_sensor.observe()
        obs = self.back_camera_sensor.observe()
        return task.cont
    
    def cleanup(self):
        self.world = None
        self.worldNP.removeNode()

    @abc.abstractmethod
    def setup(self):
        pass
