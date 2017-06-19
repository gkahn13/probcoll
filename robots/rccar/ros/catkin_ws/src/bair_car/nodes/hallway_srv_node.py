#!/usr/bin/env python
import rospy
import numpy as np
from car_srv import CarSrv
from panda3d.core import Vec3
from panda3d.core import BitMask32
from panda3d.bullet import BulletHelper
from panda3d.bullet import BulletPlaneShape
from panda3d.bullet import BulletRigidBodyNode

class HallwaySrvNode(CarSrv):

    def doReset(self, pos=None, quat=None):
        if pos is None:
            rand_val = np.random.random() * 8 - 4.0 
            pos = (rand_val, 0.0, -0.6)
            self.load_vehicle(pos=pos)
        else:
            self.load_vehicle(pos=pos, quat=quat)

    def setup(self):
        shape = BulletPlaneShape(Vec3(0, 0, 1), 0)
        ground = self.worldNP.attachNewNode(BulletRigidBodyNode('Ground'))
        ground.node().addShape(shape)
        ground.setPos(0, 0, -1)
        ground.setCollideMask(BitMask32.allOn())

        self.world.attachRigidBody(ground.node())

        # collision
        maze = []
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
            visNP.reparentTo(ground)
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
                    maze.append(bodyNP)
            
        for bodyNP in maze:
            self.world.attachRigidBody(bodyNP.node())

if __name__ == '__main__':
    rospy.init_node('car_node', anonymous=True)
    game = HallwaySrvNode()
    base.run()

