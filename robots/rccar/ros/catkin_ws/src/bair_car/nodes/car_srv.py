#!/usr/bin/env python
import rospy
import threading
import Queue
import os
import time
import numpy as np
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
        self.mass = self.params['mass']
        self.vehicle_node.setMass(self.mass)
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

        self.accept('q', self.forward_0)
        self.accept('w', self.forward_1)
        self.accept('e', self.forward_2)
        self.accept('a', self.left)
        self.accept('s', self.stop)
        self.accept('x', self.backward)
        self.accept('d', self.right)

        # ROS
        self.crash_pub = rospy.Publisher('crash', std_msgs.msg.Empty, queue_size = 1)
        self.bridge = cv_bridge.CvBridge()
        
        self.steering = 0.0       # degree
        self.steeringClamp = self.params['steeringClamp']
        self.engineForce = 0.0
        self.brakeForce = 0.0
        self.p = self.params['p']
        self.i = self.params['i']
        self.d = self.params['d']
        self.des_vel = None
        self.last_err = 0.0
        self.curr_time = 0.0
        self.accelClamp = self.params['accelClamp']
        self.engineClamp = self.accelClamp * self.mass
        self.camera_pub = ImageROSPublisher("image")
        self.depth_pub = ImageROSPublisher("depth")
        self.back_camera_pub = ImageROSPublisher("back_image")
        self.back_depth_pub = ImageROSPublisher("back_depth")
        
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

    def forward_0(self):
        self.des_vel = 14.4
#        self.engineForce = 1000.0
        self.brakeForce = 0.0
    
    def forward_1(self):
        self.des_vel = 28.8
#        self.engineForce = 1000.0
        self.brakeForce = 0.0
    
    def forward_2(self):
        self.des_vel = 48.
#        self.engineForce = 1000.0
        self.brakeForce = 0.0
   
    def stop(self):
        self.des_vel = 0.0
        self.brakeForce = 0.0

    def backward(self):
        self.des_vel = -28.8
#        self.engineForce = -1000.0
        self.brakeForce = 0.0
    
    def right(self):
        self.steering = np.min([np.max([-15, self.steering - 5]), 0.0])

    def left(self):
        self.steering = np.max([np.min([15, self.steering + 5]), 0.0])

    # Vehicle and ROS
 
    def default_pos(self):
        return (0.0, 0.0, 0.0)

    def default_quat(self):
        return (1.0, 0.0, 0.0, 0.0)

    def update(self, dt=1.0):
        self.vehicle.setSteeringValue(self.steering, 0)
        self.vehicle.setSteeringValue(self.steering, 1)
        self.vehicle.setBrake(self.brakeForce, 2)
        self.vehicle.setBrake(self.brakeForce, 3)
      
        pos = np.array(self.vehicle_pointer.getPos())
        np_quat = self.vehicle_pointer.getQuat()
        quat = np.array(np_quat)
        self.previous_pos = pos
        self.previous_quat = np_quat
        
        step = 0.05
        if dt > step:
            # TODO maybe change number of timesteps
            for i in xrange(int(dt/step)):
                if self.des_vel is not None:
                    vel = self.vehicle.getCurrentSpeedKmHour()
                    err = self.des_vel - vel
                    d_err = (err - self.last_err)/step
                    self.last_err = err
                    self.engineForce = np.clip(self.p * err + self.d * d_err, -self.engineClamp, self.engineClamp)
                self.vehicle.applyEngineForce(self.engineForce, 2)
                self.vehicle.applyEngineForce(self.engineForce, 3)
                self.world.doPhysics(step, 1, step)
                # Collision detection
                result = self.world.contactTest(self.vehicle_node)
                self.collision = result.getNumContacts() > 0
                if self.collision:
                    break
        else:
            self.curr_time += dt
            if self.curr_time > 0.05:
                if self.des_vel is not None:
                    vel = self.vehicle.getCurrentSpeedKmHour()
                    print(vel, self.curr_time)
                    err = self.des_vel - vel
                    d_err = (err - self.last_err)/0.05
                    self.last_err = err
                    self.engineForce = np.clip(self.p * err + self.d * d_err, -self.engineClamp, self.engineClamp)
                self.curr_time = 0.0
            self.vehicle.applyEngineForce(self.engineForce, 2)
            self.vehicle.applyEngineForce(self.engineForce, 3)
            self.world.doPhysics(dt, 1, dt)

            # Collision detection
            result = self.world.contactTest(self.vehicle_node)
            self.collision = result.getNumContacts() > 0
        
        if self.collision:
            # TODO figure out why this causes problems
#                self.crash_pub.publish(std_msgs.msg.Empty())
            self.doReset(pos=self.previous_pos, quat=self.previous_quat)

        self.state = geometry_msgs.msg.Pose()
        pos = np.array(self.vehicle_pointer.getPos())
        np_quat = self.vehicle_pointer.getQuat()
        quat = np.array(np_quat)
        self.previous_pos = pos
        self.previous_quat = np_quat
        self.state.position.x, self.state.position.y, self.state.position.z = pos
        self.state.orientation.x, self.state.orientation.y, \
                self.state.orientation.z, self.state.orientation.w = quat
        
        # Get observation
        self.back_obs = self.back_camera_sensor.observe()
        self.obs = self.camera_sensor.observe()
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
                # conversion from m/s to km/h
                self.des_vel = vel * 3.6
#                cmd_motor = np.clip(vel * 3 + 49.5, 0., 99.)
#                self.engineForce = self.engineClamp * \
#                    ((cmd_motor - 49.5) / 49.5)
#                self.des_vel = None
            else:
                cmd_motor = np.clip(motor, 0., 99.)
                self.engineForce = self.engineClamp * \
                    ((cmd_motor - 49.5) / 49.5)
                self.des_vel = None
            
            self.steering = self.steeringClamp * \
                ((cmd_steer - 49.5) / 49.5)

            if self.engineForce == 0.0 and (self.des_vel is None or self.des_vel == 0.0):
                self.brakeForce = 1000.0
            else:
                self.brakeForce = 0.0

            if reset:
                self.steering = 0.0       # degree
                self.engineForce = 0.0
                self.brakeForce = 1000.0
                self.des_vel = None
                pos = pose.position.x, pose.position.y, pose.position.z
                quat = pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w
                if np.all(np.array(pos) == 0.):
                    self.doReset()
                else:
                    self.doReset(pos=pos, quat=quat)

            self.update(dt=self.dt)
            vel = self.vehicle.getCurrentSpeedKmHour()
            cam_image = self.get_ros_image(self.obs[0])
            cam_depth = self.get_ros_image(self.obs[1], image_format="passthrough")
            back_cam_image = self.get_ros_image(self.back_obs[0])
            back_cam_depth = self.get_ros_image(self.back_obs[1], image_format="passthrough")
            return [self.collision, cam_image, cam_depth, back_cam_image, back_cam_depth, self.state, vel] 
        return sim_env_handler

    def load_vehicle(self, pos=None, quat=None):
       
        if pos is None:
            pos = self.default_pos()
        if quat is None:
            quat = self.default_quat()
        self.steering = 0.0
        self.engineForce = 0.0
        self.brakeForce = 1000.0
        self.vehicle.setSteeringValue(0.0, 0)
        self.vehicle.setSteeringValue(0.0, 1)
        self.vehicle.setBrake(1000.0, 2)
        self.vehicle.setBrake(1000.0, 3)
        self.vehicle.applyEngineForce(0.0, 2)
        self.vehicle.applyEngineForce(0.0, 3)

        self.previous_pos = pos
        self.vehicle_pointer.setPos(pos[0], pos[1], pos[2])
        if quat is not None:
            self.vehicle_pointer.setQuat(quat)
        self.previous_quat = self.vehicle_pointer.getQuat()
        
        while abs(self.vehicle.getCurrentSpeedKmHour()) > 4.0:
            self.world.doPhysics(self.dt, int(self.dt/0.05), 0.05)
            self.previous_pos = pos
            self.vehicle_pointer.setPos(pos[0], pos[1], pos[2])
            if quat is not None:
                self.vehicle_pointer.setQuat(quat)
            self.previous_quat = self.vehicle_pointer.getQuat()
        

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
        wheel.setFrictionSlip(1e2)
        wheel.setRollInfluence(0.1)
        wheel.setNode(wheel_np.node())
    
    def start_update_server(self):
        s = rospy.Service('sim_env', bair_car.srv.sim_env, self.get_handler())
        rospy.spin()

    def update_task(self, task):
        dt = globalClock.getDt()
        self.update(dt=dt)
        return task.cont
    
    def cleanup(self):
        pass

    @abc.abstractmethod
    def setup(self):
        pass
