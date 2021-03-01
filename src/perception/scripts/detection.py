#!/usr/bin/env python3
from __future__ import print_function

import roslib
import sys
import time
import rospy
import os.path
import cv2
import argparse
import numpy as np
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PoseStamped
from cv_bridge import CvBridge, CvBridgeError

import torch
import torch.nn as nn
from torchvision import transforms

# from dd2419_detector_baseline_OG import utils as NNutils
# from dd2419_detector_baseline_OG.detector import Detector
# import dd2419_detector_baseline_OG.utils as NNutils
from dd2419_detector_baseline_OG import utils
from dd2419_detector_baseline_OG.detector import Detector


#TODO fix gitignore and detector_baseline location

my_path = os.path.abspath(os.path.dirname(__file__))
MODEL_PATH = os.path.join(
    my_path, "../models/det_2021-02-26_09-39-26-142893.pt")  # Current model

class image_converter:

  def __init__(self, device):
        self.image_pub = rospy.Publisher("/myresult", Image, queue_size=2)

        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber(
            "/cf1/camera/image_raw", Image, self.callback, queue_size=1, buff_size=2**28)
        self.camera_param_sub = rospy.Subscriber("/cf1/camera/camera_info", CameraInfo, self.cam_callback)
        self.camera_params = None

        # self.detector = Detector().to("cpu")
        self.__detector__ = Detector()
        self.device_type = device

        if self.device_type == "cuda":
            self.device = torch.device("cuda:0")
            self.detector = utils.load_model(self.__detector__, MODEL_PATH, self.device)
            self.detector = self.detector.to(self.device)
        elif self.device_type == "cpu":
            self.detector = utils.load_model(self.__detector__, MODEL_PATH, "cpu")
        else:
            print("Error")
        # self.model = torch.load(MODEL, map_location=torch.device('cpu'))
        # self.model.eval()
        self.trans = transforms.ToTensor()

        # self.model = utils.load_model(self.model, MODEL, torch.device('cpu'))

  def cam_callback(self, data):
      self.camera_params = data

  def draw_axis(self, img, R, t, K):
    # unit is mm
    rotV, _ = cv2.Rodrigues(R)
    points = np.float32([[55, 0, 0], [0, 55, 0], [0, 0, 55], [0, 0, 0]]).reshape(-1, 3)
    axisPoints, _ = cv2.projectPoints(points, rotV, t, K, (0, 0, 0, 0))
    img = cv2.line(img, tuple(axisPoints[3].ravel()), tuple(axisPoints[0].ravel()), (255,0,0), 3)
    img = cv2.line(img, tuple(axisPoints[3].ravel()), tuple(axisPoints[1].ravel()), (0,255,0), 3)
    img = cv2.line(img, tuple(axisPoints[3].ravel()), tuple(axisPoints[2].ravel()), (0,0,255), 3)
    return img

  def draw(self, img, center, imgpts):
    pts = imgpts.astype("int32")
    img = cv2.line(img, center, tuple(pts[0].ravel()), (255,0,0), 5)
    img = cv2.line(img, center, tuple(pts[1].ravel()), (0,255,0), 5)
    img = cv2.line(img, center, tuple(pts[2].ravel()), (0,0,255), 5)
    return img

  def callback(self, data):
    # Convert the image from OpenCV to ROS format
    # torch.set_num_threads(12)
    # print("NUMBER OF THREADS:",torch.get_num_threads())

    try:
        cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
    except CvBridgeError as e:
        print(e)
    torch_im = self.trans(cv_image)
    if self.device_type == "cuda":
        torch_im = torch.unsqueeze(torch_im, 0).to(self.device)
    else:
        torch_im = torch.unsqueeze(torch_im, 0)

    self.detector.eval()
    with torch.no_grad():
        start = time.time()
        out = self.detector(torch_im)
        end = time.time()
        print("MODEL TIME: {}".format(end - start))
        bb = self.detector.decode_output(out, 0.5)

        if bb:
            height = bb[0][0]["height"].item()*-1
            width = bb[0][0]["width"].item() # shitfix
            # bb[0][0]["width"] = torch.tensor(bb[0][0]["width"].item()*-1)
            # bb[0][0]["height"] = torch.tensor(bb[0][0]["height"].item()*-1) # shitfix
            top_left = (int(round(bb[0][0]['x'].item())), round(bb[0][0]['y'].item()))
            top_right = (top_left[0] + round(width), top_left[1])
            bottom_left = (top_left[0], top_left[1] - round(height))
            bottom_right = (top_left[0] + round(width),
                            top_left[1] - round(height))
            center = (int(round(top_left[0]+width/2)), int(round(top_left[1]-height/2)))
            category = utils.get_category_dict("src/perception/scripts/dd2419_detector_baseline_OG/dd2419_coco/annotations/training.json")[bb[0][0]['category']]['name']

            top_left = tuple((int(i) if i > 0 else 0 for i in top_left))
            top_right = tuple((int(i) if i > 0 else 0 for i in top_right))
            bottom_left = tuple((int(i) if i > 0 else 0 for i in bottom_left))
            bottom_right = tuple((int(i) if i > 0 else 0 for i in bottom_right))

            print("\n\n{TL}----------{TR}\n{BL}----------{BR}\n\n".format(TL=top_left, TR=top_right, BL=bottom_left, BR=bottom_right))

            cv2.line(cv_image, top_left, top_right, (0, 0, 255), 2) # red
            cv2.line(cv_image, bottom_left, top_left, (0, 0, 255), 2) # green
            cv2.line(cv_image, bottom_right, bottom_left, (0, 0, 255), 2) # blue
            cv2.line(cv_image, bottom_right, top_right, (0, 0, 255), 2) # yellow
            cv2.putText(cv_image, text=category, org=(bottom_left[0], bottom_left[1] + 20), fontFace=cv2.FONT_HERSHEY_SIMPLEX,  
                   fontScale=0.5, color=(0,0,0), thickness=3, lineType=cv2.LINE_AA)
            cv2.putText(cv_image, text=category, org=(bottom_left[0], bottom_left[1] + 20), fontFace=cv2.FONT_HERSHEY_SIMPLEX,  
            fontScale=0.5, color=(255,255,255), thickness=1, lineType=cv2.LINE_AA)

            if self.camera_params:
                D = np.array(self.camera_params.D)
                K = np.array(self.camera_params.K).reshape(3,3)
                P = np.array(self.camera_params.P).reshape(3,4)
                R = np.array(self.camera_params.R).reshape(3,3)

                # 2D points in image (later would be detected keypoints with SIFT?)
                image_points = np.array([
                    top_left,
                    top_right,
                    center,
                    bottom_left,
                    bottom_right
                ], dtype="double")

                # 3D coordinates of keypoints in world frame (arbitrary as of now, not quite sure see: https://learnopencv.com/head-pose-estimation-using-opencv-and-dlib/)
                model_points = np.array([
                    (-width/2, height/2, 0.0), # top left
                    (width/2, height/2, 0.0), # top right
                    (0.0, 0.0, 0.0), # center (refrence origin?)
                    (-width/2, -height/2, 0.0), # bottom left
                    (width/2, -height/2, 0.0) # bottom right
                ])

                success, rvec, tvec = cv2.solvePnP(model_points, image_points, K, D)
                axis = np.array([[45.0, 0.0, 0.0], [0.0, 45.0, 0.0], [0.0, 0.0, -45.0]]).reshape(-1,3)
                axis_points_2D, jac = cv2.projectPoints(axis, rvec, tvec, K, D)

                # Both draw() and draw_axis seem to return reasonable results. Are they correct though?
                # Will do for now (Hardcoded for MS1)
                # cv_image = self.draw(cv_image, center, a/xis_points_2D)
                cv_image = self.draw_axis(cv_image, R, tvec, K)


    try:
        self.image_pub.publish(self.bridge.cv2_to_imgmsg(cv_image, "bgr8"))
    except CvBridgeError as e:
        print(e)

    DEBUG = None


def main(args):
  rospy.init_node('detection', anonymous=True)


  ic = image_converter(args)

  print("running...")
  try:
    rospy.spin()
  except KeyboardInterrupt:
    print("Shutting down")

  cv2.destroyAllWindows()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    device = parser.add_mutually_exclusive_group(required=True)
    device.add_argument("--cpu", dest="device", action="store_const", const="cpu")
    device.add_argument("--gpu", dest="device", action="store_const", const="cuda")
    args = parser.parse_args()
    main(args.device)
