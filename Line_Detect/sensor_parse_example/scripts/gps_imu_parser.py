#!/usr/bin/env python

import rospy
import numpy as np
import tf
from pyproj import Proj

from sensor_msgs.msg import Imu
from morai_msgs.msg import GPSMessage
from nav_msgs.msg import Odometry

class Converter:
    def __init__(self,zeon=52):

        self.gps_sub = rospy.Subscriber("/gps",GPSMessage,self.navsat_callback)
        self.imu_sub = rospy.Subscriber("/imu",Imu,self.imu_callback)

        self.odom_pub = rospy.Publisher('/odom',Odometry, queue_size=1)

        self.x,self.y,self.heading = None,None,None

        self.proj_UTM = Proj(proj = 'utm',zone=52,ellps='WGS84',preserve_units=False)

        self.odom_msg = Odometry()
        self.odom_msg.header.frame_id='/map'
        self.odom_msg.child_frame_id='/base_link'

    def imu_callback(self, imu_msg):

        qx = imu_msg.orientation.x
        qy = imu_msg.orientation.y
        qz = imu_msg.orientation.z
        qw = imu_msg.orientation.w

        self.odom_msg.pose.pose.orientation.x = qx
        self.odom_msg.pose.pose.orientation.y = qy
        self.odom_msg.pose.pose.orientation.z = qz
        self.odom_msg.pose.pose.orientation.w = qw

        (_,_, yaw) = tf.transformations.euler_from_quaternion([qx,qy,qz,qw])

        self.heading = yaw

    def convertLL2UTM(self):
        
        xy_zone = self.proj_UTM(self.lon,self.lat)

        self.x = xy_zone[0]-self.e_o
        self.y = xy_zone[1]-self.n_o

    def navsat_callback(self, gps_msg):

        self.lat = gps_msg.latitude
        self.lon = gps_msg.longitude

        self.e_o = gps_msg.eastOffset
        self.n_o = gps_msg.northOffset

        self.convertLL2UTM()

        if self.heading is not None:

            br = tf.TransformBroadcaster()

            br.sendTransform((self.x, self.y, 0.),
                            tf.transformations.quaternion_from_euler(0,0,self.heading),
                            rospy.Time.now(),
                            "base_link",
                            "map")

            self.odom_msg.header.stamp = rospy.get_rostime()

            self.odom_msg.pose.pose.position.x = self.x
            self.odom_msg.pose.pose.position.y = self.y
            self.odom_msg.pose.pose.position.z = 0

            self.odom_pub.publish(self.odom_msg)

if __name__=='__main__':

    rospy.init_node('gps_imu_parser',anonymous=True)

    gps_parser = Converter()

    rospy.spin()