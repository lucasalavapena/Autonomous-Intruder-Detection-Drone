#!/usr/bin/env python3
from __future__ import print_function

import roslib
import sys
import time
import rospy
import os.path
import cv2
import numpy as np
from std_msgs.msg import String
from sensor_msgs.msg import Image
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

  def __init__(self):
        self.image_pub = rospy.Publisher("/myresult", Image, queue_size=2)

        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber(
            "/cf1/camera/image_raw", Image, self.callback, queue_size=1, buff_size=2**28)

        # self.detector = Detector().to("cpu")
        self.__detector__ = Detector()

        self.gpu_device = torch.device("cuda:0")
        self.detector = utils.load_model(self.__detector__, MODEL_PATH, self.gpu_device)
        self.detector = self.detector.to(self.gpu_device)
        # self.model = torch.load(MODEL, map_location=torch.device('cpu'))
        # self.model.eval()
        self.trans = transforms.ToTensor()

        # self.model = utils.load_model(self.model, MODEL, torch.device('cpu'))

  def callback(self, data):
    # Convert the image from OpenCV to ROS format
    # torch.set_num_threads(12)
    # print("NUMBER OF THREADS:",torch.get_num_threads())

    try:
        cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
    except CvBridgeError as e:
        print(e)
    torch_im = self.trans(cv_image)
    torch_im = torch.unsqueeze(torch_im, 0).to(self.gpu_device)

    self.detector.eval()
    with torch.no_grad():
        start = time.time()
        out = self.detector(torch_im)
        end = time.time()
        print("MODEL TIME: {}".format(end - start))
        bb = self.detector.decode_output(out, 0.5)

        if bb:
            # bb[0][0]["width"] = torch.tensor(bb[0][0]["width"].item()*-1)
            bb[0][0]["height"] = torch.tensor(bb[0][0]["height"].item()*-1) # shitfix
            top_left = (int(round(bb[0][0]['x'].item())), round(bb[0][0]['y'].item()))
            top_right = (top_left[0] + round(bb[0][0]['width'].item()), top_left[1])
            bottom_left = (top_left[0], top_left[1] - round(bb[0][0]['height'].item()))
            bottom_right = (top_left[0] + round(bb[0][0]['width'].item()),
                            top_left[1] - round(bb[0][0]['height'].item()))
            category = utils.get_category_dict("src/perception/scripts/dd2419_detector_baseline_OG/dd2419_coco/annotations/training.json")[bb[0][0]['category']]['name']

            top_left = tuple((int(i) if i > 0 else 0 for i in top_left))
            top_right = tuple((int(i) if i > 0 else 0 for i in top_right))
            bottom_left = tuple((int(i) if i > 0 else 0 for i in bottom_left))
            bottom_right = tuple((int(i) if i > 0 else 0 for i in bottom_right))

            print("\n\n{TL}----------{TR}\n{BL}----------{BR}\n\n".format(TL=top_left, TR=top_right, BL=bottom_left, BR=bottom_right))

            cv2.line(cv_image, top_left, top_right, (0, 0, 255), 2) # red
            cv2.line(cv_image, bottom_left, top_left, (0, 255, 0), 2) # green
            cv2.line(cv_image, bottom_right, bottom_left, (255, 0, 0), 2) # blue
            cv2.line(cv_image, bottom_right, top_right, (0, 255, 255), 2) # yellow
            cv2.putText(cv_image, text=category, org=(bottom_left[0], bottom_left[1] + 20), fontFace=cv2.FONT_HERSHEY_SIMPLEX,  
                   fontScale=0.5, color=(0,0,0), thickness=3, lineType=cv2.LINE_AA)
            cv2.putText(cv_image, text=category, org=(bottom_left[0], bottom_left[1] + 20), fontFace=cv2.FONT_HERSHEY_SIMPLEX,  
            fontScale=0.5, color=(255,255,255), thickness=1, lineType=cv2.LINE_AA)

        # - "x": Top-left corner column
        # - "y": Top-left corner row
        # - "width": Width of bounding box in pixel
        # - "height": Height of bounding box in pixel
        # - "category": Category (not implemented yet!)

    # res = cv2.bitwise_and(cv_image, cv_image, mask=combined_mask)

    # Publish the image
    try:
        self.image_pub.publish(self.bridge.cv2_to_imgmsg(cv_image, "bgr8"))
    except CvBridgeError as e:
        print(e)

    DEBUG = None


def main(args):
  rospy.init_node('detection', anonymous=True)


  ic = image_converter()

  print("running...")
  try:
    rospy.spin()
  except KeyboardInterrupt:
    print("Shutting down")

  cv2.destroyAllWindows()


if __name__ == '__main__':
    main(sys.argv)
