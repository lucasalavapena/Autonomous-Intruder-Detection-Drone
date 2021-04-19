#!/usr/bin/env python3
from __future__ import print_function

import math
import rospy
import pickle
import os.path
import cv2
import argparse
import numpy as np
import matplotlib.pyplot as plt

import tf
from sensor_msgs.msg import Image
from geometry_msgs.msg import TransformStamped
from cv_bridge import CvBridge, CvBridgeError
from tf.transformations import quaternion_from_euler
import torchvision.transforms.functional as TF

import torch
import torch.nn as nn
from torchvision import transforms

from dd2419_detector_baseline_OG import utils
from dd2419_detector_baseline_OG.detector import Detector

DETECTOR = "SURF"
LABELS_PATH = "src/perception/scripts/dd2419_detector_baseline_OG/dd2419_coco/annotations/training.json"
CWD = os.path.abspath(os.path.dirname(__file__))
MODEL_PATH = os.path.join(
    CWD, "../models/det_2021-04-13_10-36-30-100473.pt")  # Current model
AXIS = np.float32([[3, 0, 0], [0, 3, 0], [0, 0, 3]]).reshape(-1, 3)


class image_converter:

    def __init__(self, device="cpu"):
        # Publishers
        self.image_pub = rospy.Publisher("/myresult", Image, queue_size=2)
        self.br = tf.TransformBroadcaster()

        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber(
            "/cf1/camera/image_raw", Image, self.callback, queue_size=1, buff_size=2**28)

        self.camera_params = {
            "D": np.array([0.061687, -0.049761, -0.008166, 0.004284, 0.0]),
            "K": np.array([231.250001, 0.0, 320.519378, 0.0, 231.065552, 240.631482, 0.0, 0.0, 1.0]).reshape(3, 3),
            "P": np.array([231.25, 0.0, 322.360322, 0.0, 0.0, 231.06, 240.631, 0.0, 0.0, 0.0, 1.0, 0.0]).reshape(3, 4),
            "R": np.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]).reshape(3, 3)
        }

        self.__detector__ = Detector()
        self.device_type = device

        # Unpickling and deserializing precomputed featurepoints
        self.features = self.unpickle()
        self.categories = utils.get_category_dict(LABELS_PATH)

        # Initializing different feature detectors
        self.feature_detector = {
            "SIFT": cv2.xfeatures2d.SIFT_create(),
            "SURF": cv2.xfeatures2d.SURF_create(500),  # 500
            "ORB": cv2.ORB_create(),
            "STAR": cv2.xfeatures2d.StarDetector_create(),
            "BRIEF": cv2.xfeatures2d.BriefDescriptorExtractor_create(),
            "MATCHER": cv2.BFMatcher()
        }

        if self.device_type == "cuda":
            self.device = torch.device("cuda:0")
            self.detector = utils.load_model(
                self.__detector__, MODEL_PATH, self.device)
            self.detector = self.detector.to(self.device)
        elif self.device_type == "cpu":
            self.detector = utils.load_model(
                self.__detector__, MODEL_PATH, "cpu")
        else:
            print("Error")
        self.trans = transforms.ToTensor()


    def unpickle(self):
        # Unpickling and deserializing precomputed featurepoints
        with open(os.path.join(CWD, 'features.pickle'), 'rb') as handle:
            features = pickle.load(handle)

        signs = features.keys()
        keys = features[signs[0]].keys()
        unpickled = {}

        for sign in signs:
            unpickled[sign] = {}
            for key in keys:
                if key == "CENTER" or key == "IMAGE":
                    unpickled[sign][key] = features[sign][key]
                else:
                    unpickled[sign][key] = {'kp': [], 'des': []}

        for sign in signs:
            for key in keys:
                if key != "CENTER" and key != "IMAGE":
                    kp = [cv2.KeyPoint(x=p[0][0], y=p[0][1], _size=p[1], _angle=p[2],
                                       _response=p[3], _octave=p[4], _class_id=p[5]) for p in features[sign][key]]
                    des = np.array([d[6] for d in features[sign][key]])
                    unpickled[sign][key]['kp'] = kp
                    unpickled[sign][key]['des'] = des

        return unpickled

    def get_corners_and_cat(self, bb):
        x = int(round(bb['x']))
        y = int(round(bb['y']))
        height = int(round(bb["height"]*-1))  # shitfix
        width = int(round(bb["width"]))
        category = self.categories[bb['category']]['name']

        top_left = (x, y)
        top_right = (x + width, y)
        bottom_left = (x, y - height)
        bottom_right = (x + width, y - height)
        center = (x + width/2, y - height/2)

        top_left = tuple(i if i > 0 else 0 for i in top_left)
        top_right = tuple(i if i > 0 else 0 for i in top_right)
        bottom_left = tuple(i if i > 0 else 0 for i in bottom_left)
        bottom_right = tuple(i if i > 0 else 0 for i in bottom_right)
        return top_left, top_right, bottom_left, bottom_right, center, category

    def draw_bb_and_cat(self, image, top_left, top_right, bottom_left, bottom_right, category):
        cv2.line(image, top_left, top_right, (0, 0, 255), 2)  # red
        cv2.line(image, bottom_left, top_left, (0, 0, 255), 2)  # green
        cv2.line(image, bottom_right, bottom_left, (0, 0, 255), 2)  # blue
        cv2.line(image, bottom_right, top_right, (0, 0, 255), 2)  # yellow
        cv2.putText(image, text=category, org=(bottom_left[0], bottom_left[1] + 20), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.5, color=(0, 0, 0), thickness=3, lineType=cv2.LINE_AA)
        cv2.putText(image, text=category, org=(bottom_left[0], bottom_left[1] + 20), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.5, color=(255, 255, 255), thickness=1, lineType=cv2.LINE_AA)
    # ----------------- Feature Detection -----------------

    def draw(self, img, bb_center_in_drone_img, imgpts):
        imgpts = imgpts.astype("int32")
        img = cv2.line(img, bb_center_in_drone_img, tuple(
            imgpts[0].ravel()), (255, 0, 0), 5)
        img = cv2.line(img, bb_center_in_drone_img, tuple(
            imgpts[1].ravel()), (0, 255, 0), 5)
        img = cv2.line(img, bb_center_in_drone_img, tuple(
            imgpts[2].ravel()), (0, 0, 255), 5)
        return img

    def detect_features(self, img, detector=DETECTOR):
        if detector == "SURF":
            # find the keypoints and descriptors with SIFT
            kp, des = self.feature_detector["SURF"].detectAndCompute(img, None)
        elif detector == "SIFT":
            # find the keypoints and descriptors with SIFT
            kp, des = self.feature_detector["SIFT"].detectAndCompute(img, None)
        elif detector == "ORB":
            # find the keypoints and descriptors with ORB
            kp, des = self.feature_detector["ORB"].detectAndCompute(img, None)
        elif detector == "BRIEF":
            # find the keypoints with STAR and compute the descriptors with BRIEF
            kp = self.feature_detector["STAR"].detect(img, None)
            kp, des = self.feature_detector["BRIEF"].compute(img, kp)
        return kp, des

    def get_matches(self, canon_des, drone_des, canon_kp, drone_kp, canon_img, drone_img, display_result=False):
        # BFMatcher with default params
        matches = self.feature_detector['MATCHER'].knnMatch(
            canon_des, drone_des, k=2)

        # Apply ratio test
        good = []
        for m, n in matches:
            if m.distance < 0.60 * n.distance:
                good.append([m])

        if display_result:
            img3 = cv2.drawMatchesKnn(canon_img, canon_kp, drone_img, drone_kp,
                                      good, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
            plt.imshow(img3), plt.ion(), plt.show(), plt.pause(0.001)
        return good

    def get_points(self, canon_kp, drone_kp, matches, canonical_center):
        canonical2D_kp = np.array(
            [canon_kp[item[0].queryIdx].pt for item in matches])
        image_points = np.array(
            [drone_kp[item[0].trainIdx].pt for item in matches], dtype=np.float32)
        object_points = np.zeros(
            (image_points.shape[0], image_points.shape[1] + 1), dtype=np.float64)
        object_points[:, :2] = (canonical2D_kp - canonical_center) / 10.0
        return object_points, image_points

    def callback(self, data):
        # Convert the image from OpenCV to ROS format
        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            print(e)

        torch_im = self.trans(cv_image)

        if self.device_type == "cuda":
            torch_im = torch.unsqueeze(torch_im, 0).to(self.device)
            torch.cuda.synchronize()
        else:
            torch_im = torch.unsqueeze(torch_im, 0)

        # set the detector to evaluation mode, speeds inference.
        self.detector.eval()

        with torch.no_grad():
            out = self.detector(torch_im)  # .cpu()

            # detect bounding box with threshold
            bbs = self.detector.decode_output(out, 0.7, multiple_bb=True)

            if bbs:
                for bb in bbs[0]:
                    top_left, top_right, bottom_left, bottom_right, center, category = self.get_corners_and_cat(
                        bb)
                    self.draw_bb_and_cat(
                        cv_image, top_left, top_right, bottom_left, bottom_right, category)

                    # ----------------------- REVISED FEATURE DETECTION ----------------------
                    # detect features
                    kp, des = self.detect_features(cv_image)
                    sign = category.replace(" ", "_")
                    # find matches
                    matches = self.get_matches(
                        self.features[sign][DETECTOR]['des'],
                        des,
                        self.features[sign][DETECTOR]['kp'],
                        kp,
                        self.features[sign]['IMAGE'],
                        cv_image,
                        display_result=False
                    )

                    if len(matches) >= 4:
                        # get points (maybe find centerpoint first?)
                        object_points, image_points = self.get_points(
                            self.features[sign][DETECTOR]['kp'],
                            kp,
                            matches,
                            self.features[sign]['CENTER']
                        )
                        # SolvePnPRansac
                        retval, rvec, tvec, inliers = cv2.solvePnPRansac(
                            object_points.reshape(-1, 1, 3), 
                            image_points.reshape(-1, 1, 2), 
                            self.camera_params["K"], 
                            self.camera_params["D"], 
                            useExtrinsicGuess=True)

                        # project axis with result from ransac
                        projected_axis, jacobian = cv2.projectPoints(
                            AXIS, rvec, tvec, self.camera_params["K"], self.camera_params["D"])

                        # draw axis on drone image
                        bb_center_in_drone_img = (
                            bb['x'] + bb['width']/2, bb['y'] + bb['height']/2)
                        bb_center_in_drone_img = tuple(
                            int(i.item()) for i in bb_center_in_drone_img)
                        cv_image = self.draw(
                            cv_image, bb_center_in_drone_img, projected_axis)
                        # ------------------------------------------------------------------------

                        # ---------------------- Pose Creation and publication  ----------------------
                        t = TransformStamped()
                        t.header.frame_id = 'cf1/camera_link'
                        t.header.stamp = rospy.Time.now()
                        t.child_frame_id = sign

                        rotation = quaternion_from_euler(math.radians(rvec.ravel()[0]),
                                                        math.radians(rvec.ravel()[1]),
                                                        math.radians(rvec.ravel()[2]))
                        
                        self.br.sendTransform(
                            tvec.ravel(), rotation, rospy.Time.now(), sign, 'cf1/camera_link') 
                        # ---------------------------------------------------------------------------------

        try:
            self.image_pub.publish(self.bridge.cv2_to_imgmsg(cv_image, "bgr8"))
        except CvBridgeError as e:
            print(e)


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
    device.add_argument("--cpu", dest="device",
                        action="store_const", const="cpu")
    device.add_argument("--gpu", dest="device",
                        action="store_const", const="cuda")
    args = parser.parse_args()
    main(args.device)