#!/usr/bin/env python
from __future__ import print_function

import math
import rospy
import pickle
import os.path
import cv2
import numpy as np
import matplotlib.pyplot as plt
import PIL.Image
import tf
from sensor_msgs.msg import Image
from std_msgs.msg import String
from geometry_msgs.msg import TransformStamped
from cv_bridge import CvBridge, CvBridgeError
from tf.transformations import quaternion_from_euler
import torchvision.transforms.functional as TF

import time
import torch
from torchvision import transforms

from dd2419_detector_baseline_OG import utils
from dd2419_detector_baseline_OG.detector import Detector

PX_PER_CM = 11.3

LABELS_PATH = "src/perception/scripts/dd2419_detector_baseline_OG/dd2419_coco/annotations/training.json"
CWD = os.path.abspath(os.path.dirname(__file__))
MODEL_PATH = os.path.join(
    CWD, "../models/det_2021-04-13_10-36-30-100473.pt")  # Current model
# MODEL_PATH = os.path.join(
#     CWD, "../models/det_2021-02-26_09-39-26-142893.pt")  # old model
AXIS = np.float32([[3, 0, 0], [0, 3, 0], [0, 0, 3]]).reshape(-1, 3)
RotX = np.float32([[1, 0, 0], [0, -1,0], [0, 0, -1]]) # Rotation matrix about x-axis

Ryz = np.array([[ 0.20076997, -0.40057632,  0.89399666],
       [-0.89399666, -0.44807362,  0.        ],
       [ 0.40057632, -0.79923003, -0.44807362]])


# CV_IMAGE = cv2.imread("/home/robot/dd2419_project/src/perception/scripts/debug_photos/stop_angle05.jpg")
class image_converter:

    def __init__(self, device="cpu"):
        # Publishers
        self.image_pub = rospy.Publisher("/myresult", Image, queue_size=2)
        self.sign_pub = rospy.Publisher("/sign_poses", String, queue_size=50)
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
            "SURF": cv2.xfeatures2d.SURF_create(),  # 500
            "ORB": cv2.ORB_create(),
            # "STAR": cv2.xfeatures2d.StarDetector_create(),
            # "BRIEF": cv2.xfeatures2d.BriefDescriptorExtractor_create(),
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
        with open(os.path.join(CWD, PICKLE_FILE), 'rb') as handle:
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

    def get_corners_and_cat(self, bb, img):
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

        cropped_img = img[top_left[1]: bottom_right[1], top_left[0]: bottom_right[0]]

        return top_left, top_right, bottom_left, bottom_right, center, category, cropped_img

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


    def detect_features(self, img, detector="SURF"):
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
            if m.distance < 0.75 * n.distance:
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
        object_points[:, :2] = (canonical2D_kp - canonical_center) / (PX_PER_CM * 100) # TODO this is random
        return object_points, image_points

    def callback(self, data):
        # Convert the image from OpenCV to ROS format
        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
            # cv_image = CV_IMAGE.copy()
            start_time = time.time()
            cv_image_nn = PIL.Image.fromarray(cv_image)#PIL.Image.open("/home/robot/dd2419_project/src/perception/scripts/debug_photos/stop_angle05.jpg") #CV_IMAGE.copy()
            print("took {}s to convert image to PIL".format(time.time()-start_time))
        except CvBridgeError as e:
            print(e)

        cv_image_nn = TF.to_tensor(cv_image_nn)

        # if self.device_type == "cuda":
        #     torch_im = torch.unsqueeze(torch_im, 0).to(self.device)
        #     torch.cuda.synchronize()
        # else:
        #     torch_im = torch.unsqueeze(torch_im, 0)

        # set the detector to evaluation mode, speeds inference.
        self.detector.eval()

        with torch.no_grad():
            cv_image_nn = cv_image_nn.to(self.device_type)
            out = self.detector(cv_image_nn[None, ...])

            # out = self.detector(torch_im)  # .cpu()

            # detect bounding box with thresholdr_bb
            bbs = self.detector.decode_output(out[0], NN_THRESHOLD, multiple_bb=True)
            if bbs:
                for bb in bbs[0]:
                    start_time = time.time()

                    # image pre-processing
                    top_left, top_right, bottom_left, bottom_right, center_in_og_img, category, cropped_img = self.get_corners_and_cat(
                        bb, cv_image)
                    self.draw_bb_and_cat(
                        cv_image, top_left, top_right, bottom_left, bottom_right, category)
                    # ----------------------- REVISED FEATURE DETECTION ----------------------
                    # detect features
                    kp, des = self.detect_features(cropped_img, detector=DETECTOR)
                    sign = category.replace(" ", "_")





                    # print("canon kp\n{}".format(len(self.features[sign][DETECTOR]['kp'])))
                    # kp1_og_hashes = [8735329470608, 8735330314887, 8735330314983, 8735330314986, 8735330316559, 8735330316562,
                    #  8735330316565, 8735330316568, 8735330316571, 8735330316574, 8735330316577, 8735330316580,
                    #  8735330316583, 8735330316586, 8735330316589, 8735330316592, 8735330316595, 8735330316598,
                    #  8735330316601, 8735330316604, 8735330316607, 8735330316610, 8735330316613, 8735330316616,
                    #  8735330316619, 8735330316622, 8735330316625, 8735330316628, 8735330316631, 8735330316634,
                    #  8735330316637, 8735330316640, 8735330316643, 8735330316646, 8735330316649, 8735330316652,
                    #  8735330316655, 8735330316658, 8735330316661, 8735330316664, 8735330316667, 8735330316670,
                    #  8735330316673, 8735330316676, 8735330316679, 8735330316682, 8735330316685, 8735330316688,
                    #  8735330316691, 8735330316694, 8735330316697, 8735330316700, 8735330316703, 8735330316706,
                    #  8735330316709, 8735330316712, 8735330316715, 8735330316718, 8735330316721, 8735330316724,
                    #  8735330316727, 8735330316730]
                    # kp1_hashes = [hash(kp_canon) for kp_canon in self.features[sign][DETECTOR]['kp']]
                    # if sorted(kp1_og_hashes) != sorted(kp1_hashes):
                    #     print("kp1 hashes differ")
                    #
                    # kp2_og_hashes = [8774104041399, 8774104041402, 8774104041405, 8774104041408, 8774104041411, 8774104041414, 8774104041417, 8774104041420, 8774104041423, 8774104041426, 8774104041429, 8774104041432, 8774104041435, 8774104041438, 8774104041441, 8774104041444, 8774104041447, 8774104041450, 8774104041453, 8774104041456, 8774104041459, 8774104041462, 8774104041465, 8774104041468, 8774104159491, 8774104159494, 8774104159497, 8774104159500, 8774104159503, 8774104159506, 8774104159509, 8774104159512, 8774104159515, 8774104159518, 8774104159521, 8774104159524, 8774104159527, 8774104159530, 8774104159533, 8774104159536, 8774104159539, 8774104159542, 8774104159545, 8774104159548, 8774104159551, 8774104159554, 8774104159557, 8774104159560, 8774104159563, 8774104159566, 8774104159569, 8774104159572, 8774104159575, 8774104159578, 8774104159581, 8774104159584, 8774104159587, 8774104159590, 8774104159593, 8774104159596, 8774104159599, 8774104159602, 8774104159605, 8774104159608, 8774104159611, 8774104159614, 8774104159617, 8774104159620, 8774104159623, 8774104159626, 8774104159629, 8774104159632, 8774104159635, 8774104159638, 8774104159641, 8774104159644, 8774104159647, 8774104159650, 8774104159653, 8774104159656, 8774104159659, 8774104159662, 8774104159665, 8774104159668, 8774104159671, 8774104159674, 8774104159677, 8774104159680, 8774104159683, 8774104159686, 8774104159689, 8774104159692, 8774104159695, 8774104159698, 8774104159701, 8774104159704, 8774104159707, 8774104159710, 8774104159713, 8774104159716, 8774104159719, 8774104159722, 8774104159725, 8774104159728, 8774104159731, 8774104159734, 8774104159737, 8774104159740, 8774104157443, 8774104157446, 8774104157449, 8774104157452, 8774104157455, 8774104157458, 8774104157461, 8774104157464, 8774104157467, 8774104157470, 8774104157473, 8774104157476, 8774104157479, 8774104157482, 8774104157485, 8774104157488, 8774104157491, 8774104157494, 8774104157497, 8774104157500, 8774104157503, 8774104157506, 8774104157509, 8774104157512, 8774104157515, 8774104157518, 8774104157521, 8774104157524, 8774104157527, 8774104157530, 8774104157533, 8774104157536, 8774104157539, 8774104157542, 8774104157545, 8774104157548, 8774104157551, 8774104157554, 8774104157557, 8774104157560, 8774104157563, 8774104157566, 8774104157569, 8774104157572, 8774104157575, 8774104157578, 8774104157581, 8774104157584, 8774104157587, 8774104157590, 8774104157593, 8774104157596, 8774104157599, 8774104157602, 8774104157605]
                    #
                    # # print("kp", kp)
                    # kp2_hashes = [hash(i) for i in kp]
                    # if sorted(kp2_og_hashes) != sorted(kp2_hashes):
                    #     print("kp2 hashes differ")
                    #
                    # print("kp2", len(list(set(kp2_og_hashes).symmetric_difference(set(kp2_hashes)))))

                    # find matches
                    matches = self.get_matches(
                        self.features[sign][DETECTOR]['des'],
                        des,
                        self.features[sign][DETECTOR]['kp'],
                        kp,
                        self.features[sign]['IMAGE'],
                        cropped_img,
                        display_result=False
                    )

                    # pretty_good = [(i[0].distance, i[0].imgIdx, i[0].queryIdx, i[0].trainIdx) for i in matches]
                    # print(len(pretty_good))
                    # for j in pretty_good:
                    #     print(j)


                    if len(matches) >= 4:
                        # get points (maybe find centerpoint first?)
                        object_points, image_points = self.get_points(
                            self.features[sign][DETECTOR]['kp'],
                            kp,
                            matches,
                            self.features[sign]['CENTER']
                        )
                        # print("object points:\n{}\nimage_points:\n{}".format(object_points, image_points))
                        # Convert image points from cropped to actual image
                        center_in_cropped_img = ((bottom_right[0] - top_left[0])/2, (bottom_right[1] - top_left[1])/2)
                        image_points = image_points + np.array(center_in_og_img) - np.array(center_in_cropped_img)
                        # print("do math correctly", np.array(center_in_og_img), np.array(center_in_cropped_img))


                        # SolvePnPRansac
                        retval, rvec, tvec, inliers = cv2.solvePnPRansac(
                            object_points.reshape(-1, 1, 3),
                            image_points.reshape(-1, 1, 2),
                            self.camera_params["K"],
                            self.camera_params["D"])

                        print(tvec)

                        norm = np.linalg.norm(tvec)
                        if norm > 30 or norm < 1e-10:
                            continue

                        # Python implementation of C++ code below, not currently working but maybe a good start?
                        # rodrigues, _ = cv2.Rodrigues(rvec)
                        # rvec_converted, _ = cv2.Rodrigues(rodrigues.T)
                        # rvec_converted = Ryz * rvec_converted
                        #
                        # tvec_converted = -rodrigues.T * tvec
                        # tvec_converted = Ryz * tvec_converted
                        #
                        # rvec[0], rvec[1], rvec[2] = rvec_converted[0][0], rvec_converted[1][1], rvec_converted[2][2]
                        # tvec[0], tvec[1], tvec[2] = tvec_converted[0][0], tvec_converted[1][1], tvec_converted[2][2]

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

                        print("took {}s to compute and draw 6d pose".format(time.time() - start_time))

                        # ------------------------------------------------------------------------
                        sign_msg = "{SIGN},{STAMP},{FRAME},{TVEC},{RVEC}".format(SIGN=sign,
                                                                                 STAMP=rospy.Time.now(),
                                                                                 FRAME="cf1/camera_link",
                                                                                 TVEC=tvec.tostring(),
                                                                                 RVEC=rvec.tostring())
                        self.sign_pub.publish(sign_msg)


                        # ---------------------- Pose Creation and publication  ----------------------
                        # t = TransformStamped()
                        # t.header.frame_id = 'cf1/camera_link'
                        # t.header.stamp = rospy.Time.now()
                        # t.child_frame_id = "landmark/detected_" + sign
                        #
                        # rotation = quaternion_from_euler(math.radians(rvec.ravel()[0]),
                        #                                 math.radians(rvec.ravel()[1]),
                        #                                 math.radians(rvec.ravel()[2]))
                        #
                        # self.br.sendTransform(
                        #     tvec.ravel(), rotation, rospy.Time.now(), "landmark/detected_" + sign, 'cf1/camera_link')
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

DETECTOR = rospy.get_param("~feature_detector", "SIFT")
HARDWARE = rospy.get_param('~inference_hardware', 'cpu')
PICKLE_FILE = rospy.get_param('~pickle_file', 'lucas-features.pickle')

NN_THRESHOLD = rospy.get_param("~nn_theshold", 0.5)

if __name__ == '__main__':
    main(HARDWARE)