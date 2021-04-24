import os.path
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import time
from dd2419_detector_baseline_OG.utils import run_model_singleimage

SCALING_FACTOR = 0.3333
DRONE_IMAGE_RATIO = (640, 640) # TODO we might change this depending on different signs.
PX_PER_CM = 11.3


class feature_detected:
    def __init__(self, kp1, kp2, good, canon_img, drone_img, center_in_og_img, center_in_cropped_img, canon_center, drone_img_og):
        self.key_points = {1: kp1, 2: kp2}
        self.good_matches = good
        self.images = {"canon_img": canon_img,
                       "drone_img": drone_img,
                       "drone_img_og": drone_img_og}
        self.centers = {"center_in_og_img": center_in_og_img,
                        "center_in_cropped_img": center_in_cropped_img,
                        "canon_center": canon_center}

def get_camera_values():
    D = np.array([0.061687, -0.049761, -0.008166, 0.004284, 0.0])
    K = np.array([231.250001, 0.0, 320.519378, 0.0, 231.065552, 240.631482, 0.0, 0.0, 1.0]).reshape(3, 3)
    P = np.array([231.25, 0.0, 322.360322, 0.0, 0.0, 231.06, 240.631, 0.0, 0.0, 0.0, 1.0, 0.0]).reshape(3, 4)
    R = np.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]).reshape(3, 3)

    return D, K, P, R

def image_preprocessing(canon_img_path, drone_img_path, bounding_box=None, crop=True):
    canon_img = cv.imread(canon_img_path)  # queryImage
    drone_img = cv.imread(drone_img_path)   # trainImage
    drone_img_og = drone_img
    # plt.imshow(drone_img), plt.show()
    canon_img = cv.cvtColor(canon_img, cv.COLOR_BGR2GRAY)
    drone_img = cv.cvtColor(drone_img, cv.COLOR_BGR2GRAY)

    dsize = (int(round(DRONE_IMAGE_RATIO[0] * SCALING_FACTOR)), int(round(DRONE_IMAGE_RATIO[1] * SCALING_FACTOR)))
    canon_img = cv.resize(canon_img, dsize, interpolation=cv.INTER_AREA)
    # plt.imshow(canon_img), plt.show()
    #
    # plt.imshow(drone_img), plt.show()


    if bounding_box is not None:
        height = bounding_box["height"].item() * -1
        width = bounding_box["width"].item()  # shitfix
        top_x = int(round(bounding_box['x'].item()))
        top_y = int(round(bounding_box['y'].item()))
        bottom_x = int(top_x + round(width))
        bottom_y = int(top_y - round(height))
        if crop:
            drone_img = drone_img[top_y: bottom_y, top_x: bottom_x]
        #
        # dsize = (int(round(bounding_box['width'].item())),
        #          int(round(bounding_box['height'].item())))  # cv_im_cropped.shape[0:2]#(640, 480)
        # canon_img = cv.resize(canon_img, dsize, interpolation=cv.INTER_AREA)

        center_in_og_img = ((top_x + bottom_x)/2, (top_y + bottom_y)/2)
        center_in_cropped_img = ((bottom_x - top_x)/2, (bottom_y - top_y)/2)
        canon_center = ((canon_img.shape[0]) / 2, (canon_img.shape[1]) / 2)
    else:
        top_x = 0
        top_y = 0
        bottom_x = canon_img.shape[0]
        bottom_y = canon_img.shape[1]

        center_in_og_img = ((drone_img.shape[0]) / 2, (drone_img.shape[1]) / 2)

        center_in_cropped_img = center_in_og_img

    return canon_img, drone_img, center_in_cropped_img,\
           center_in_og_img, canon_center, drone_img_og

def get_matches(des1, des2, canon_img, kp1, drone_img, kp2, ratio_factor=0.75, display_result=False):
    # BFMatcher with default params
    bf = cv.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)

    # Apply ratio test
    good = []
    for m, n in matches:
        if m.distance < ratio_factor * n.distance:
            good.append([m])

    if display_result:
        img3 = cv.drawMatchesKnn(canon_img, kp1, drone_img, kp2, good, None, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        plt.imshow(img3), plt.show()
    return good

def sift_feature_detection(canon_img, drone_img, ratio_factor=0.75, display_result=False):


    # dsize = (int(round(DRONE_IMAGE_RATIO[0] * SCALING_FACTOR)), int(round(DRONE_IMAGE_RATIO[1] * SCALING_FACTOR)))
    # canon_img = cv.resize(canon_img, dsize, interpolation=cv.INTER_AREA)

    # Initiate SIFT detector
    sift = cv.xfeatures2d.SIFT_create()

    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(canon_img, None)
    kp2, des2 = sift.detectAndCompute(drone_img, None)


    good = get_matches(des1, des2, canon_img, kp1, drone_img, kp2, ratio_factor=ratio_factor,
                       display_result=display_result)

    return kp1, kp2, good



def orb_feature_detection(canon_img, drone_img, ratio_factor=0.75, display_result=False):

    # Initiate SIFT detector
    orb = cv.ORB_create() # 500, 1.2, 8, 31, 0, 2

    # find the keypoints and descriptors with SIFT
    kp1, des1 = orb.detectAndCompute(canon_img, None)
    kp2, des2 = orb.detectAndCompute(drone_img, None)

    good = get_matches(des1, des2, canon_img, kp1, drone_img, kp2, ratio_factor=ratio_factor,
                       display_result=display_result)

    return kp1, kp2, good


def surf_feature_detection(canon_img, drone_img, ratio_factor=0.75, display_result=False):



    # dsize = (int(round(DRONE_IMAGE_RATIO[0] * SCALING_FACTOR)), int(round(DRONE_IMAGE_RATIO[1] * SCALING_FACTOR)))
    # canon_img = cv.resize(canon_img, dsize, interpolation=cv.INTER_AREA)
    # Initiate SIFT detector
    surf = cv.xfeatures2d.SURF_create(100, 12, 12, False, False)

    # find the keypoints and descriptors with SIFT
    kp1, des1 = surf.detectAndCompute(canon_img, None)
    kp2, des2 = surf.detectAndCompute(drone_img, None)

    good = get_matches(des1, des2, canon_img, kp1, drone_img, kp2, ratio_factor=ratio_factor,
                       display_result=display_result)

    return kp1, kp2, good

def BRIEF_feature_detection(canon_img, drone_img, ratio_factor=0.75, display_result=False):

    # Initiate SIFT detector
    star = cv.xfeatures2d.StarDetector_create()

    brief = cv.xfeatures2d.BriefDescriptorExtractor_create()

    # find the keypoints with STAR
    kp1 = star.detect(canon_img, None)
    kp2 = star.detect(drone_img, None)

    # compute the descriptors with BRIEF
    kp1, des1 = brief.compute(canon_img, kp1)
    kp2, des2 = brief.compute(drone_img, kp2)


    good = get_matches(des1, des2, canon_img, kp1, drone_img, kp2, ratio_factor=ratio_factor,
                       display_result=display_result)

    return kp1, kp2, good

def draw(img, corners, imgpts):
    img = cv.line(img, corners, tuple(imgpts[0].ravel()), (255, 0, 0), 5)
    img = cv.line(img, corners, tuple(imgpts[1].ravel()), (0, 255, 0), 5)
    img = cv.line(img, corners, tuple(imgpts[2].ravel()), (0, 0, 255), 5)
    return img

def get_points(kp1, kp2, good, object_center):
    canonical2D_kp = np.array([kp1[item[0].queryIdx].pt for item in good])
    image_points = np.array([kp2[item[0].trainIdx].pt for item in good], dtype=np.float32)
    object_points = np.zeros((image_points.shape[0], image_points.shape[1] + 1), dtype=np.float64)
    object_points[:, :2] = (canonical2D_kp - object_center) / (PX_PER_CM * 100) # scale in meters

    return object_points, image_points


def get_orientation(see_image_points=False):
    my_path = os.path.abspath(os.path.dirname(__file__))
    canon_img_path = os.path.join(my_path, "dd2419_traffic_sign_pdfs", "follow_right.jpg")
    # drone_img_path = os.path.join(my_path, "dd2419_detector_baseline_OG/performance_test/test_images",
    #                               "0000097.jpg")
    canon_img_path = "/home/robot/dd2419_project/src/perception/scripts/dd2419_traffic_sign_pdfs/stop.jpg"
    # drone_img_path = "/home/robot/dd2419_project/src/perception/scripts/debug_photos/stop13.jpg"

    drone_img_path = "/home/robot/dd2419_project/src/perception/scripts/debug_photos/stop_angle05.jpg"

    bounding_box = run_model_singleimage(drone_img_path, 0.5)[0][0]
    print(bounding_box)
    crop = True
    crop_canon = False

    canon_img, drone_img, center_in_cropped_img, center_in_og_img, canon_center,  drone_img_og =\
        image_preprocessing(canon_img_path, drone_img_path, bounding_box, crop)

    kp1, kp2, good = sift_feature_detection(canon_img, drone_img, ratio_factor=0.75, display_result=False)

    features_detected = feature_detected(kp1, kp2, good, canon_img, drone_img, center_in_og_img, center_in_cropped_img,
                            canon_center, drone_img_og)


    axis = np.float32([[3, 0, 0], [0, 3, 0], [0, 0, 3]]).reshape(-1, 3)
    D, K, P, R = get_camera_values()
    camera_matrix = K
    dist_coeffs = D



    # get feature object and image points (from cropped image)
    object_points, image_points = get_points(features_detected.key_points[1], features_detected.key_points[2],
                                             features_detected.good_matches, features_detected.centers["canon_center"])

    print("object points:\n{}\nimage_points:\n{}".format(object_points, image_points))

    # reconvert images points from cropped image to original image location
    image_points = image_points + np.array([features_detected.centers["center_in_og_img"]]) - np.array(features_detected.centers["center_in_cropped_img"])

    print("do math correctly", np.array([features_detected.centers["center_in_og_img"]]),np.array(features_detected.centers["center_in_cropped_img"]))

    if see_image_points:
        plt.imshow(features_detected.images["drone_img"])
        plt.scatter(image_points[:, 0], image_points[:, 1])
        plt.show()
        plt.imshow(features_detected.images["canon_img"])
        plt.scatter(object_points[:, 0] * 10.0 + features_detected.centers["canon_center"][0],
                    object_points[:, 1] * 10.0 + features_detected.centers["canon_center"][1])
        plt.show()

    # SolvePnPRansac

    retval, rvec, tvec, inliers = cv.solvePnPRansac(object_points.reshape(-1, 1, 3),
                                                    image_points.reshape(-1, 1, 2),
                                                    camera_matrix, dist_coeffs,
                                                    )
    print(tvec)

    rotation_matrix, _ = cv.Rodrigues(rvec)

    projected_axis, jacobian = cv.projectPoints(axis, rvec, tvec, camera_matrix, dist_coeffs)

    drone_sign_center_loc = features_detected.centers["center_in_og_img"]

    result_img = draw(features_detected.images["drone_img_og"], drone_sign_center_loc, projected_axis)

    plt.imshow(result_img), plt.show()



if __name__ == "__main__":
    get_orientation(False)



