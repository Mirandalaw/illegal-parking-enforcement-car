#!/usr/bin/env python

import rospy
import rospkg
import cv2
import os
import numpy as np
import math
import time
import sensor_msgs.point_cloud2 as pc2
import json
import tf

from sensor_msgs.msg import PointCloud2, CompressedImage
from morai_msgs.msg import ObjectStatus,ObjectStatusList, EgoVehicleStatus
from darknet_ros_msgs.msg import BoundingBox,BoundingBoxes
from nav_msgs.msg import Odometry
# from utils import Lidar2Camtransform

class SensorCalib:

    def __init__(self, params_cam, params_lidar):

        self.scan_sub = rospy.Subscriber("/velodyne_points", PointCloud2, self.scan_callback)

        self.image_sub = rospy.Subscriber("/image_jpeg/compressed", CompressedImage, self.img_callback)

        self.l2c_trans = Lidar2Camtransform(params_cam, params_lidar)
        
        self.limit_lidar_z = params_lidar["Z"]

        self.pc_np = None
        self.img = None
    
    def img_callback(self,msg):

        np_arr = np.frombuffer(msg.data, np.uint8)

        self.img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    def scan_callback(self,msg):

        self.pc_np = self.pointcloud2_to_xyz(msg)

    def pointcloud2_to_xyz(self,cloud_msg):

        point_list = []

        for point in pc2.read_points(cloud_msg, skip_nans=True):

            dist = np.sqrt(point[0]**2+point[1]**2+point[2]**2)

            if point[0]>0 and dist < 50 and point[2] > -self.limit_lidar_z:
                point_list.append((point[0],point[1],point[2],dist))

        point_np=np.array(point_list,np.float32)

        return point_np

class OBEstimator:

    def __init__(self):

        self.bbox_sub = rospy.Subscriber("/darknet_ros/bounding_boxes",BoundingBoxes, self.bbox_callback)

        self.odom_sub = rospy.Subscriber("odom", Odometry,self.odom_callback)

        self.obinfo_pub = rospy.Publisher("/object_topic", ObjectStatusList,queue_size=3)

        self.bounding_boxes=None

    def bbox_callback(self,msg):

        self.bounding_boxes = msg.bounding_boxes

    def odom_callback(self,msg):

        self.x,self.y = msg.pose.pose.position.x, msg.pose.pose.position.y

        qx = msg.pose.pose.orientation.x
        qy = msg.pose.pose.orientation.y
        qz = msg.pose.pose.orientation.z
        qw = msg.pose.pose.orientation.w

        (_,_,yaw) = tf.transformations.euler_from_quaternion([qx,qy,qz,qw])

        self.heading = yaw

        self.RT = np.matmul(
            np.array([[1,0,self.x],[0,1,self.y],[0,0,1]]),
            np.array([
                [np.cos(self.heading),-np.sin(self.heading),0],
                [np.sin(self.heading),np.cos(self.heading),0],
                [0,0,1]])
        )

    def transform_global(self,x,y):

        xy_g = np.matmul(
            self.RT, np.array([[x],[y],[1.]])
        )

        return xy_g[:2, :].reshape([-1])

    def calc_ob_xy(self,xyii):

        self.obinfo_msg = ObjectStatusList()
        
        object_type = []

        object_id = []

        pose_x = []
        pose_y = []

        num_of_npcs = 0
        num_of_pedestrian = 0

        for bbox_msg in self.bounding_boxes:

            tmp_ob_status = ObjectStatus()

            x = bbox_msg.xmin
            w = bbox_msg.xmax - bbox_msg.xmin
            y = bbox_msg.ymin
            h = bbox_msg.ymax -  bbox_msg.ymin

            cx = int(x + w/2)
            cy = int(y + h/2)

            xy_o = xyii[np.logical_and(xyii[:, 0]>=cx-0.5*w, xyii[:,0]<cx+0.5*w), :]
            xy_o = xy_o[np.logical_and(xy_o[:, 1]>=cy-0.5*h, xy_o[:,0]<cy+0.5*h), :]
            
            xy_o = np.mean(xy_o[:, 2:], axis=0)[:2]

            xy_g = self.transform_global(xy_o[0],xy_o[1])

            if not np.isnan(xy_g[0]) and not np.isnan(xy_g[1]):

                tmp_ob_status.position.x = xy_g[0]
                tmp_ob_status.position.y = xy_g[1]

                print(bbox_msg.Class, xy_g)

                if bbox_msg.Class =='car':

                    num_of_npcs+=1

                    tmp_ob_status.type = 1

                    self.obinfo_msg.npc_list.append(tmp_ob_status)

                elif bbox_msg.Class == 'person':

                    num_of_pedestrian+=1

                    tmp_ob_status.type = 0

                    self.obinfo_msg.pedestrian_list.append(tmp_ob_status)

                else :

                    pass

            else:

                pass

        self.obinfo_msg.num_of_npcs = num_of_npcs
        self.obinfo_msg.num_of_pedestrian = num_of_pedestrian

    def publish_object_info(self):

        self.obinfo_pub.publish(self.obinfo_msg)

if __name__ =='__main__':

    rp = rospkg.RosPack()

    currentPath = rp.get_path("traffic_example")

    with open(os.path.join(currentPath, 'sensor/sensor_params.json'),'r') as fp:
        sensor_params = json.load(fp)

    params_cam = sensor_params["params_cam"]

    params_lidar = sensor_params["params_lidar"]

    rospy.init_node('yolo_object_xy_publisher',anonymous=True)

    ex_calib_transform = SensorCalib(params_cam,params_lidar)

    objectinfo_estimator = OBEstimator()
    
    time.sleep(1)

    rate = rospy.Rate(10)

    time.sleep(1)

    rate = rospy.Rate(10)

    while not rospy.is_shutdown():
        
        if objectinfo_estimator.bounding_boxes is not None:

            xyz_p = ex_calib_transform.pc_np[:, :3]

            xyz_c = ex_calib_transform.l2c_trans.transform_lidar2cam(xyz_p)

            xy_i = ex_calib_transform.l2c_trans.project_pts2img(xyz_c,False)

            xyii = np.concatenate([xy_i,xyz_p],axis=1)

            xyii = ex_calib_transform.l2c_trans.crop_pts(xyii)

            objectinfo_estimator.calc_ob_xy(xyii)

            objectinfo_estimator.publish_object_info()

            rate.sleep()    