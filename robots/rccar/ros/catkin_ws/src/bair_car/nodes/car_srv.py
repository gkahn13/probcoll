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
from panda3d.bullet import BulletVehicle
from panda3d.bullet import ZUp
from panda3d.bullet import BulletConvexHullShape

class CarSrv(DirectObject):

    def __init__(self):
        self.params = rospy.get_param('~sim')
        self.dt = rospy.get_param('~dt')
        
        base.setBackgroundColor(0.1, 0.1, 0.8, 1)

        # World
        self.worldNP = render.attachNewNode('World')
        self.world = BulletWorld()
        self.world.setGravity(Vec3(0, 0, -9.81))
        
        
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

        # Camera
        self.camera_sensor = Panda3dCameraSensor(
            base,
            color=True,
            depth=True,
            size=(160,90))
        self.camera_node = self.camera_sensor.cam
        self.camera_node.setPos(0.0, 1.0, 1.0)
        self.camera_node.lookAt(0.0, 6.0, 0.0)
        
        self.back_camera_sensor = Panda3dCameraSensor(
            base,
            color=True,
            depth=True,
            size=(160,90))
        self.back_camera_node = self.back_camera_sensor.cam
        self.back_camera_node.setPos(0.0, -1.0, 1.0)
        self.back_camera_node.lookAt(0.0, -6.0, 0.0)
        
        # Models
        self._yugoNP = loader.loadModel('../models/yugo/yugo.egg')
        self._right_front_np = loader.loadModel('../models/yugo/yugotireR.egg')
        self._left_front_np = loader.loadModel('../models/yugo/yugotireL.egg')
        self._right_rear_np = loader.loadModel('../models/yugo/yugotireR.egg')
        self._left_rear_np = loader.loadModel('../models/yugo/yugotireL.egg')

        # Vehicle
        shape = BulletBoxShape(Vec3(0.6, 1.4, 0.5))
        ts = TransformState.makePos(Point3(0, 0, 0.5))
        self.vehicle_node = BulletRigidBodyNode('Vehicle')
        self.vehicle_node.addShape(shape, ts)
        self._mass = self.params['mass']
        self.vehicle_node.setMass(self._mass)
        self.vehicle_node.setDeactivationEnabled(False)
        self.vehicle_node.setCcdSweptSphereRadius(1.0)
        self.vehicle_node.setCcdMotionThreshold(1e-7)
        # TODO
        self.vehicle_pointer = self.worldNP.attachNewNode(self.vehicle_node)
        self.camera_node.reparentTo(self.vehicle_pointer)

        self.back_camera_node.reparentTo(self.vehicle_pointer)

        self.world.attachRigidBody(self.vehicle_node)

        # Vehicle
        self.vehicle = BulletVehicle(self.world, self.vehicle_node)
        self.vehicle.setCoordinateSystem(ZUp)
        self.world.attachVehicle(self.vehicle)
        self._yugoNP.reparentTo(self.vehicle_pointer)
        self._right_front_np.reparentTo(self.worldNP)
        self.addWheel(Point3( 0.70,    1.05, 0.3), True, self._right_front_np)
        self._left_front_np.reparentTo(self.worldNP)
        self.addWheel(Point3(-0.70,    1.05, 0.3), True, self._left_front_np)
        self._right_rear_np.reparentTo(self.worldNP)
        self.addWheel(Point3( 0.70, -1.05, 0.3), False, self._right_rear_np)
        self._left_rear_np.reparentTo(self.worldNP)
        self.addWheel(Point3(-0.70, -1.05, 0.3), False, self._left_rear_np)
        
        # Car Simulator
        self.setup()
        self.load_vehicle()
        
        # Input
        self.accept('escape', self.doExit)
        self.accept('r', self.doReset)
        self.accept('f1', self.toggleWireframe)
        self.accept('f2', self.toggleTexture)
        self.accept('f5', self.doScreenshot)

        self.accept('w', self.forward)
        self.accept('a', self.left)
        self.accept('s', self.backward)
        self.accept('d', self.right)

        # ROS
        self.crash_pub = rospy.Publisher('crash', std_msgs.msg.Empty, queue_size = 1)
        self.bridge = cv_bridge.CvBridge()
#        taskMgr.add(self.update_task, 'updateWorld')
        self.start_update_server()
    
    # _____HANDLER_____

    def doExit(self):
        self.cleanup()
        sys.exit(1)

    def doReset(self, pos=None, quat=None):
        if pos is None or quat is None:
            self.load_vehicle()
        else:
            self.load_vehicle(pos=pos, quat=quat)

    def toggleWireframe(self):
        base.toggleWireframe()

    def toggleTexture(self):
        base.toggleTexture()

    def doScreenshot(self):
        base.screenshot('Bullet')

    def forward(self):
        self.engineForce = 1000.0
    
    def backward(self):
        self.engineForce = -1000.0

    def left(self):
        self.steering = -22.5

    def right(self):
        self.steering = 22.5

    # Vehicle and ROS
        
    def update(self):
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
        self.collision = result.getNumContacts() > 0
        
        if self.collision:
            # TODO figure out why this causes problems
#                self.crash_pub.publish(std_msgs.msg.Empty())
            self.doReset(pos=self.previous_pos, quat=self.previous_quat)

        self.state = geometry_msgs.msg.Pose()
        pos = numpy.array(self.vehicle_pointer.getPos())
        np_quat = self.vehicle_pointer.getQuat()
        quat = numpy.array(np_quat)
        self.previous_pos = pos
        self.previous_quat = np_quat
        self.state.position.x, self.state.position.y, self.state.position.z = pos
        self.state.orientation.x, self.state.orientation.y, \
                self.state.orientation.z, self.state.orientation.w = quat
        
        # Get observation
        self.obs = self.camera_sensor.observe()
        self.back_obs = self.back_camera_sensor.observe()
        self.camera_pub.publish_image(self.obs[0])
        self.depth_pub.publish_image(
            self.obs[1],
            image_format="passthrough")
        self.back_camera_pub.publish_image(self.back_obs[0])
        self.back_depth_pub.publish_image(
            self.back_obs[1],
            image_format="passthrough")
    
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

            self.update()
            vel = self.vehicle.getCurrentSpeedKmHour()
            cam_image = self.get_ros_image(self.obs[0])
            cam_depth = self.get_ros_image(self.obs[1], image_format="passthrough")
            back_cam_image = self.get_ros_image(self.back_obs[0])
            back_cam_depth = self.get_ros_image(self.back_obs[1], image_format="passthrough")
            return [self.collision, cam_image, cam_depth, back_cam_image, back_cam_depth, self.state, vel] 
        return sim_env_handler

    def load_vehicle(self, pos=(0.0, -20.0, -0.6), quat=None):
        self.vehicle.setSteeringValue(0.0, 0)
        self.vehicle.setSteeringValue(0.0, 1)
        self.vehicle.setBrake(1000.0, 2)
        self.vehicle.setBrake(1000.0, 3)
        self.vehicle.applyEngineForce(0.0, 2)
        self.vehicle.applyEngineForce(0.0, 3)
        for wheel in self.vehicle.getWheels():
            wheel.setRotation(0.0)

        self.previous_pos = pos
        self.vehicle_pointer.setPos(pos[0], pos[1], pos[2])
        if quat is not None:
            self.vehicle_pointer.setQuat(quat)
        self.previous_quat = self.vehicle_pointer.getQuat()
        self.world.doPhysics(1.0, 10, 0.008)

        self.vehicle.setSteeringValue(0.0, 0)
        self.vehicle.setSteeringValue(0.0, 1)
        self.vehicle.setBrake(1000.0, 2)
        self.vehicle.setBrake(1000.0, 3)
        self.vehicle.applyEngineForce(0.0, 2)
        self.vehicle.applyEngineForce(0.0, 3)
        for wheel in self.vehicle.getWheels():
            wheel.setRotation(0.0)

        self.previous_pos = pos
        self.vehicle_pointer.setPos(pos[0], pos[1], pos[2])
        if quat is not None:
            self.vehicle_pointer.setQuat(quat)
        self.previous_quat = self.vehicle_pointer.getQuat()
        self.world.doPhysics(1.0, 10, 0.008)

    def addWheel(self, pos, front, wheel_np):
        wheel = self.vehicle.createWheel()
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
        wheel.setNode(wheel_np.node())
    
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

    def update_task(self, task):
        dt = globalClock.getDt()
        self.update()
        print(self.vehicle.getCurrentSpeedKmHour())
        return task.cont
    
    def cleanup(self):
        pass

    @abc.abstractmethod
    def setup(self):
        pass
