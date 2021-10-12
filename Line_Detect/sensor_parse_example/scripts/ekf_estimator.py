#!/usr/bin/env python

import rospy
import numpy as np
import tf
from pyproj import Proj

from sensor_msgs.msg import Imu
from morai_msgs.msg import GPSMessage,EgoVehicleStatus
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist, Pose, PoseWithCovariance
from gps_imu_parser import Converter
class ExtendedKalmanFilter:
    def __init__(self, T=0.05, l_vehicle=2.0, tau=0.4, K=3.0):

        self.T = T

        self.l = l_vehicle
        self.tau = tau
        self.K = K

        self.X = np.array([10,1000,0,0], dtype=float).reshape([-1,1])
        self.P = np.diag([0,0,0.2,0.2])

        self.Q = self.T*np.diag([0,0,0.0,5])
        self.R = np.diag([0.01,0.01])/self.T

        self.H = np.array([
            [1,0,0,0],
            [0,1,0,0]
        ], dtype = float)

        self.pose_pub = rospy.Publisher('/pose_ekf', Odometry,queue_size=1)

        self.odom_msg= Odometry()
        self.odom_msg.header.frame_id='/map'

    def pediction_step(self, u):

        dX_pre = np.zeros((4,1),dtype=float)

        dX_pre[0, :] = self.X[3, :]*np.cos(self.X[2, :])
        dX_pre[1, :] = self.X[3, :]*np.sin(self.X[2, :])
        dX_pre[2, :] = self.X[3, :]*np.tan(u[1])/self.l
        dX_pre[3, :] = (1/self.tau)*(-self.X[3,:]+self.K*u[0])

        self.calc_F(u)

        self.X = self.X + (self.T*dX_pre)

        if self.X[2,:]<-np.pi:
            self.X[2,:]=self.X[2,:]+2*np.pi

        elif self.X[2,:]>np.pi:
            self.X[2,:]=self.X[2,:]-2*np.pi

        else:
            pass

        self.P = self.F.dot(self.P).dot(self.F.T)+self.Q

    def correction(self,Z):

        K = self.P.dot(self.H.T).dot(np.linalg.inv(self.H.dot(self.P).dot(self.H.T)+self.R))

        Y = self.H.dot(self.X)

        self.X+=K.dot(Z-Y)

        self.P-=K.dot(self.H).dot(self.P)

    def clac_F(self,u):

        self.F = np.identity(4, dtype=float)
        self.F[0,2] += -self.T*self.X[3,:]*np.sin(self.X[2,:])
        self.F[0,3] += self.T*np.cos(self.X[2,:])
        self.F[1,2] += self.T*self.X[3,:]*np.cos(self.X[2,:])
        self.F[1,3] += self.T*np.sin(self.X[2,:])
        self.F[2,3] += self.T*np.tan(u[1])/self.l
        self.F[3,3] += self.T/(-self.tau)

    def send_estimated_state(self):

        pose_cov_ekf = PoseWithCovariance()

        q = tf.transformations.quaternion_from_euler(0, 0, self.X[2][0])

        pose_cov_ekf.pose.position.x = self.X[0][0]
        pose_cov_ekf.pose.position.y = self.X[1][0]
        pose_cov_ekf.pose.position.z = 0.

        pose_cov_ekf.pose.orientation.x = q[0]
        pose_cov_ekf.pose.orientation.y = q[1]
        pose_cov_ekf.pose.orientation.z = q[2]
        pose_cov_ekf.pose.orientation.w = q[3]

        P_3d = np.zeros((6,6))
        P_3d[:2,:2] = self.P[:2, :2]
        P_3d[-1,:2] = self.P[-1, :2]
        P_3d[:2,-1] = self.P[:2, -1]
        P_3d[-1,-1] = self.P[-1, -1]

        pose_cov_ekf.covariance = P_3d.reshape([-1]).tolist()

        self.odom_msg.pose = pose_cov_ekf
        self.pose_pub.publish(self.odom_msg)

class CMDParser:

    def __init__(self):

        self.status_sub = rospy.Subscriber("/Ego_topic",EgoVehicleStatus,self.status_callback)

        self.u = np.zeros((2,))

    def status_callback(self,msg):

        accel = msg.accel
        brake = msg.brake

        wheel_angle = msg.wheel_angle

        if accel >=0.5:
            
            self.u[0] = accel

        elif brake >=0.5:

            self.u[0]= -brake

        else:

            self.u[0] = 0.0

        self.u[1] = -wheel_angle / 180 *np.pi

if __name__ =='__main__':

    rospy.init_node('EKF_estimator',anonymous=True)

    rate = rospy.Rate(20)

    loc_sensor = Converter()

    ekf = ExtendedKalmanFilter(dt=0.05)

    cmd_gen = CMDParser()

    while not rospy.is_shutdown():

        if loc_sensor.x is not None and loc_sensor.y is not None:

            #decide the u
            u = cmd_gen.u

            #prediction step
            ekf.prediction(u)

            #measurement locations
            z = np.array([loc_sensor.x, loc_sensor.y]).reshape([-1,1])

            #correction step
            ekf.correction(z)

            #get the estimated states
            ekf.send_estimated_state()

        else:
            pass

        rate.sleep()
