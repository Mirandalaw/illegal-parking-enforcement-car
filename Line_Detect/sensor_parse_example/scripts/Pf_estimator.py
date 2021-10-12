#!/usr/bin/env python

import rospy
import numpy as np
import tf
from pyproj import Proj

from sensor_msgs.msg import Imu
from morai_msgs.msg import GPSMessage,EgoVehicleStatus
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist, Pose, PoseArray #PoseWithCovariance
from gps_imu_parser import Converter
from ekf_estimator import CMDParser

def pdf_multivariate_gauss(dx, cov):

    part2 = np.exp(-0.5*np.matmul(np.matmul(dx.T,np.linalg.inv(cov)),dx))

    part1 = 1 / ((2*np.pi)**(cov.shape[0]/2)*(np.linalg.det(cov)**(1/2)))

    return np.diag(part1 * part2) + 0.00001

class ParticleFilter:

    def __init__(self, dt=0.05, NP=500, l_vehicle=2.0, tau=0.4, K=10.0):

        self.T = dt

        self.l = l_vehicle
        self.tau = tau
        self.K = K

        self.NP = NP
        
        self.XP = np.random.randn(4, self.NP)*4 + np.array([20, 1100.,0, 0]).reshape([4,1])

        self.limit_state()

        self.pw = np.ones((NP, ))/NP

        self.Q = self.T*np.diag([0.4, 0.4, 0.0, 10])
        self.R = np.diag([2, 2])/self.T

        self.H = np.array(
            [
                [1,0,0,0],
                [0,1,0,0]
            ],
            dtype=float)
        
        self.pose_pub = rospy.Publisher('/pose_pf', PoseArray,queue_size=1)

    def prediction(self, u):

        dX_pre = np.zeros((4, self.NP),dtype=float)

        dX_pre[0, :] = self.XP[3, :]*np.cos(self.XP[2, :])
        dX_pre[1, :] = self.XP[3, :]*np.sin(self.XP[2, :])
        dX_pre[2, :] = self.XP[3, :]*np.tan(u[1])/self.l
        dX_pre[3, :] = (1/self.tau)*(-self.XP[3, :]+self.K*u[0])

        self.XP+=(self.T*dX_pre)

        self.limit_state()
    
    def limit_state(self):

        over_bool = self.XP[2, :] < -np.pi

        self.XP[2, over_bool] = self.XP[2, over_bool] + 2*np.pi

        under_bool = self.XP[2, :] > np.pi

        self.XP[2,under_bool] = self.XP[2,under_bool] - 2*np.pi

        vel_under_bool = self.XP[3,:] < 0

        self.XP[3, vel_under_bool] = 0.0


    def correction(self, Z):

        dz = self.H.dot(self.XP)-Z

        pdf_z = pdf_multivariate_gauss(dz, self.R)

        self.pw = pdf_z/pdf_z.sum()

        self.Xm = np.dot(self.XP, self.pw).reshape([-1,1])

        self.Xcov = np.dot(
            (self.XP-self.Xm),
            np.diag(self.pw)
            ).dot((self.XP-self.Xm).T)

        re_sample_ID, self.pw = self.re_sampling(self.pw)

        XP_new = self.XP[:, re_sample_ID]

        self.XP = XP_new + np.diag([0.02, 0.02, 0.01, 0.5]).dot(np.random.randn(4, self.NP))

        self.limit_state()

    def re_sampling(self, w):

        re_sample_ID = np.random.choice(np.arange(self.NP),self.NP, p=w)

        w = np.ones((self.NP,))/self.NP

        return re_sample_ID, w

    def send_estimated_state(self):

        particles_msg = PoseArray()
        particles_msg.header.frame_id='/map'

        for p_i in range(self.NP):

            particle_msg = Pose()

            px = self.XP[:, p_i]

            q = tf.transformations.quaternion_from_euler(0, 0, px[2])

            particle_msg.position.x = px[0]
            particle_msg.position.y = px[1]
            particle_msg.position.z = 0.

            particle_msg.orientation.x = q[0]
            particle_msg.orientation.y = q[1]
            particle_msg.orientation.z = q[2]
            particle_msg.orientation.w = q[3]

            particles_msg.poses.append(particle_msg)

        self.pose_pub.publish(particles_msg)

if __name__ == '__main__':

    rospy.init_node('PF_estimator',anonymous=True)

    rate = rospy.Rate(20)

    loc_sensor = Converter()

    pf = ParticleFilter(dt = 0.05)

    cmd_gen = CMDParser()

    while not rospy.is_shutdown():

        if loc_sensor.x is not None and loc_sensor.y is not None:

            #decide the u
            u = cmd_gen.u

            #prediction step
            pf.prediction(u)

            #measurement locations
            z = np.array([loc_sensor.x, loc_sensor.y]).reshape([-1,1])


            #correction step
            pf.correction(z)

            #get the estimated states
            pf.send_estimated_state()

        else: 
            pass

        rate.sleep()