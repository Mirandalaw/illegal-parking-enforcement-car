#!/usr/bin/env python

import rospy
import cv2
import numpy as np
import os, rospkg
import json

from sensor_msgs.msg import CompressedImage
from cv_bridge import C2vBridgeError

class IMGParser:
    def __init__(self):

        self.image_sub = rospy.Subscriber("/image_jpeg/compressed",
        CompressedImage,self.callback)

    def callback(self,msg):
        try:
            np_arr = np.fromstring(msg.data, np.uint8)
            img_bgr = cv2.imdecode(np_arr,cv2.IMREAD_COLOR)
        except CvBridgeError as e:
            print(e)

        img_hsv = cv2.cvtColor(img_bgr,cv2.COLOR_BGR2HSV)

        h = img_bgr.shape[0]
        w = img_bgr.shape[1]

        lower_sig_y=np.array([])
        upper_sig_y=np.array([])

        lower_sig_r=np.array([])
        upper_sig_r=np.array([])
        
        lower_sig_g=np.array([50,230,230])
        upper_sig_g=np.array([70,255,255])
        
        img_r=cv2.resize(cv2.inRange(img_hsv,lower_sig_r,upper_sig_r),(w/2,h/2))

        img_y=cv2.resize(cv2.inRange(img_hsv,lower_sig_y,upper_sig_y),(w/2,h/2))

        img_g=cv2.resize(cv2.inRange(img_hsv,lower_sig_g,upper_sig_g),(w/2,h/2))

        img_r[int(h/3/2):,:] =0
        img_y[int(h/3/2):,:] =0
        img_g[int(h/3/2):,:] =0

        img_concat=np.concatenate([img_r,img_y,img_g],axis=1)

        cv2.imshow("Image window",img_concat)

        cv2.waitKey(1)

    def mouseRGB(self,event,x,y,flags,param):
        if event == cv2.EVENT_LBUTTONDOWN :#checks mouse left button down condition
            colorsR =self.img_hsv[y,x,0]
            colorsG =self.img_hsv[y,x,1]
            colorsB =self.img_hsv[y,x,2]
            colors=self.img_hsv[y,x]
            print("Red : ",colorsR)
            print("Green : ",colorsG)
            print("Blue : ",colorsB)
            print("RGB Format : ",colors)
            print("Coordinates of pixel : X:",x,"Y: ",y)

if __name__=='__main__':

    rp = rospkg.RosPack()

    currentPath = rp.get_path("Lane_detection exmple")

    with open(os.path.join(currentPath, 'sensor/sensor_params.json'),'r')as fp:
        sensor_params=json.load(fp)

    params_cam=sensor_params["params_cam"]

    rospy.init_node('image_parser',annoymous=True)

    image_parser=IMGParser()
    bev_op=BEVTransform(params_cam=params_cam)

    rate = rospy.Rate(30)

    while not rospy.is_shutdown():

        if image_parser.edges is not None :

            img_warp = bev_op.warp_bev_img(image_parser.edges)
            lane_pts = bev_op.recon_lane_pts(image_parser.edges)

            x_pred,y_pred_l,y_pred_r = curve_learner.fit_curve(lane_pts)
            cv2.imshow("Image window", img_warp)
            cv2.waitKey(1)

            rate.sleep()
