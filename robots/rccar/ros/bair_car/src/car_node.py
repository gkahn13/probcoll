# !/usr/bin/env python

import rospy
import std_msgs.msg
import geometry_msgs.msg
import threading
import Queue
import os
import time
import numpy
from ros_utils import ImageROSPublisher

from panda3d.core import loadPrcFile
#from pandac.PandaModules import loadPrcFileData
#loadPrcFileData('', 'load-display tinydisplay')

assert "CITYSIM3D_DIR" in os.environ                             
loadPrcFile(os.path.expandvars('${CITYSIM3D_DIR}/config.prc')) 
from citysim3d.envs import Panda3dCameraSensor

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
        base.setFrameRateMeter(True)

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

        # ROS
        self.crash_pub = rospy.Publisher('crash', std_msgs.msg.Empty, queue_size = 100)
        ### subscribers (info sent to Arduino)
        self.cmd_steer_sub = rospy.Subscriber(
            'cmd/steer',
            std_msgs.msg.Float32,
            callback=self._cmd_steer_callback)
        self.cmd_motor_sub = rospy.Subscriber(
            'cmd/motor',
            std_msgs.msg.Float32,
            callback=self._cmd_motor_callback)
        self.cmd_vel_sub = rospy.Subscriber(
            'cmd/vel',
            std_msgs.msg.Float32,
            callback=self._cmd_vel_callback)
        self.reset_sub = rospy.Subscriber(
            'reset',
            std_msgs.msg.Empty,
            callback=self._reset_callback)
        self.cmd_steer_queue = Queue.Queue(maxsize=1)
        self.cmd_motor_queue = Queue.Queue(maxsize=1)
        # Task
        taskMgr.add(self.update, 'updateWorld')
        # Physics
        self.setup()
        print('Starting ROS thread')
        threading.Thread(target=self._ros_servos_thread).start()
        threading.Thread(target=self._ros_crash_thread).start() 
        threading.Thread(target=self._ros_image_thread).start()
    
    # Callbacks

    def _cmd_steer_callback(self, msg):
        if msg.data >= 0 and msg.data <= 99.0:
            self.cmd_steer_queue.put(msg.data)

    def _cmd_motor_callback(self, msg):
        if msg.data >= 0 and msg.data <= 99.0:
            self.cmd_motor_queue.put(msg.data)
    
    def _cmd_vel_callback(self, msg):
        data = numpy.clip(msg.data * 3 + 49.5, 0, 99.0)
        self.cmd_motor_queue.put(data)
    
    def _reset_callback(self, msg):
        self.doReset()

    # ROS thread

    def _ros_servos_thread(self):
        """
        Publishes/subscribes to Ros.
        """
        self.steering = 0.0       # degree
        self.steeringClamp = rospy.get_param('~steeringClamp')
        self.engineForce = 0.0
        self.engineClamp = rospy.get_param('~engineClamp')
        r = rospy.Rate(60)
        while not rospy.is_shutdown():
            r.sleep()
            for var, queue in (('steer', self.cmd_steer_queue),
                    ('motor', self.cmd_motor_queue)):
                if not queue.empty():
                    try:
                        if var == 'steer':
                            self.steering = self.steeringClamp * ((queue.get() - 49.5) / 49.5)
                            self.vehicle.setSteeringValue(self.steering, 0)
                            self.vehicle.setSteeringValue(self.steering, 1)
                        elif var == 'motor':
                            self.vehicle.setBrake(100.0, 2)
                            self.vehicle.setBrake(100.0, 3)
                            self.engineForce = self.engineClamp * ((queue.get() - 49.5) / 49.5)
                            self.vehicle.applyEngineForce(self.engineForce, 2)
                            self.vehicle.applyEngineForce(self.engineForce, 3)
                    except Exception as e:
                        print(e)
    
    def _ros_crash_thread(self):
        crash = 0
        r = rospy.Rate(30)
        while not rospy.is_shutdown():
            r.sleep()
            try:
                result = self.world.contactTest(self.vehicle_node)
                if result.getNumContacts() > 0:
                    self.crash_pub.publish(std_msgs.msg.Empty())
            except Exception as e:
                print(e)

    def _ros_image_thread(self):
        camera_pub = ImageROSPublisher("image")
        depth_pub = ImageROSPublisher("depth")
        r = rospy.Rate(30)
        i = 0
        while not rospy.is_shutdown():
            r.sleep()
            # sometimes fails to observe
            try:
                obs = self.camera_sensor.observe()
                camera_pub.publish_image(obs[0])
                depth_pub.publish_image(
                    obs[1],
                    image_format="passthrough")
            except:
                pass

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

    # ____TASK___

    def update(self, task):
        dt = globalClock.getDt()

        self.world.doPhysics(dt, 10, 0.008)

        #print self.vehicle.getWheel(0).getRaycastInfo().isInContact()
        #print self.vehicle.getWheel(0).getRaycastInfo().getContactPointWs()

        #print self.vehicle.getChassis().isKinematic()

        return task.cont

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
                (0.0, 36.0, 0.0),
                (-11.0, 24.0, 0.0),
                (11.0, 24.0, 0.0),
                (-11.0, 12.0, 0.0),
                (11.0, 12.0, 0.0),
                (-11.0, 0.0, 0.0),
                (11.0, 0.0, 0.0),
                (-11.0, -12.0, 0.0),
                (11.0, -12.0, 0.0),
                (0.0, -24.0, 0.0),
                (0.5, 12.0, 0.5),
                (-0.5, 12.0, 0.5)]:
            translate = False
            if (pos == (0.5, 12.0, 0.5)) or (pos == (-0.5, 12.0, 0.5)):
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
        mass = rospy.get_param('~mass')
        #chassis_shape = rospy.get_param('~chassis_shape')
        shape = BulletBoxShape(Vec3(0.6, 1.4, 0.5))
        ts = TransformState.makePos(Point3(0, 0, 0.5))

        np = self.worldNP.attachNewNode(BulletRigidBodyNode('Vehicle'))
        np.node().addShape(shape, ts)
        rand_vals = numpy.random.random(2) * 8 - 4.0 
        np.setPos(rand_vals[0], 0.0, -0.6)
        np.node().setMass(mass)
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
            self.camera_node.setPos(0.0, 2.0, 1.0)
        else:
            self.camera_node.reparentTo(np)
            self.camera_node.setPos(0.0, -10.0, 5.0)
            self.camera_node.lookAt(0.0, 5.0, 0.0)
        base.cam.reparentTo(np)
        base.cam.setPos(0.0, -10.0, 5.0)
        base.cam.lookAt(0.0, 5.0, 0.0)
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
        

        # Collision handle
        #base.cTrav = CollisionTraverser()
        #self.notifier = CollisionHandlerEvent()
        #self.notifier.addInPattern("%fn")
        #self.accept("Vehicle", self.onCollision)

    def onCollision(self):
        print("crash")

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

if __name__ == '__main__':
    rospy.init_node('car_node', anonymous=True)
    game = Game()
    base.run()

