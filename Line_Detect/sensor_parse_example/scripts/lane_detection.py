#!/usr/bin/env python

import rospy
import cv2
import numpy as np
import os, rospkg
import json

from sensor_msgs.msg import CompressedImage
from cv_bridge import CvBridgeError

from utils import BEVTransform, CURVEFit, draw_lane_img

class IMGParser:

    def __init__(self):
        self.image_sub = rospy.Subscriber("/image_jpeg/compressed", CompressedImage, self.callback)

        self.source_prop = np.float32([[0.05,0.60],
        [0.5-0.25,0.52],
        [0.5+0.25,0.52],
        [1-0.05,0.60]
        ])

        self.img_lane = None

    def callback(self, msg):
        try:
            np_arr = np.fromstring(msg.data, np.uint8)
            img_bgr = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        except CvBridgeError as e:
            print(e)
        
        img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

        lower_wlane = np.array([0,0,210])
        upper_wlane = np.array([30,30,255])

        lower_ylane=np.array([0,100,190])
        upper_ylane=np.array([40,175,255])

        img_wlane = cv2.inRange(img_hsv, lower_wlane, upper_wlane)
        img_ylane = cv2.inRange(img_hsv,lower_ylane,upper_ylane)

        self.img_lane = cv2.bitwise_or(img_wlane,img_ylane)


if __name__ == '__main__':
    rp = rospkg.RosPack()

    currentPath = rp.get_path("sensor_parse_example")

    with open(os.path.join(currentPath, 'sensor/sensor_params.json'), 'r') as fp:
        sensor_params = json.load(fp)
    
    params_cam = sensor_params["params_cam"]

    rospy.init_node('image_parser', anonymous=True)

    image_parser = IMGParser()
    bev_op = BEVTransform(params_cam=params_cam, xb=10, zb=10)
    curve_learner = CURVEFit(
        order=3
        # ,lane_width=2,
        # y_margin=0.8,
        # x_range=20,
        # dx=0.5,
        # min_pts=50
        )

    rate = rospy.Rate(20)

    while not rospy.is_shutdown():

        if image_parser.img_lane is not None:

            img_warp = bev_op.warp_bev_img(image_parser.img_lane)
            lane_pts = bev_op.recon_lane_pts(image_parser.img_lane)

            x_pred, y_pred_l, y_pred_r = curve_learner.fit_curve(lane_pts)

            curve_learner.write_path_msg(x_pred,y_pred_l,y_pred_r)

            curve_learner.pub_path_msg()

            xyl, xyr = bev_op.project_lane2img(x_pred, y_pred_l, y_pred_r)

            img_warp1 = draw_lane_img(img_warp, xyl[:, 0].astype(np.int32),
                                                xyl[:, 1].astype(np.int32),
                                                xyr[:, 0].astype(np.int32),
                                                xyr[:, 1].astype(np.int32))
            
            cv2.imshow("image window", img_warp)
            cv2.waitKey(1)

            rate.sleep()