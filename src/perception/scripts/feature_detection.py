import os.path
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import time
from dd2419_detector_baseline_OG.utils import run_model_singleimage

def rotate_image(image, angle):
  image_center = tuple(np.array(image.shape[1::-1]) / 2)
  rot_mat = cv.getRotationMatrix2D(image_center, angle, 1.0)
  result = cv.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv.INTER_LINEAR)
  return result

def image_preprocessing(query_img_path, train_img_path, crop):
    img1 = cv.imread(query_img_path, cv.IMREAD_GRAYSCALE)  # queryImage
    img2 = cv.imread(train_img_path, cv.IMREAD_GRAYSCALE)  # trainImage
    # img1 = img1[]
    # img2 = rotate_image(img2, 1)
    # plt.imshow(img1), plt.show()
    # plt.imshow(img2), plt.show()

    if crop:
        height = crop["height"].item() * -1
        width = crop["width"].item()  # shitfix
        top_x = int(round(crop['x'].item()))
        top_y = int(round(crop['y'].item()))
        bottom_x = int(top_x + round(width))
        bottom_y = int(top_y - round(height))
        img2 = img2[top_y: bottom_y, top_x: bottom_x]

    else:
        top_x = 0
        top_y = 0
        bottom_x = img1.shape[0]
        bottom_y = img1.shape[1]

    return img1, img2, ((bottom_x - top_x)/2, (bottom_y - top_y)/2)

def FLANN(query_img_path, train_img_path, crop):
    img1, img2, center = image_preprocessing(query_img_path, train_img_path, crop)
    start_time = time.time()

    # Initiate SIFT detector
    sift = cv.xfeatures2d.SIFT_create()
    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)
    # FLANN parameters
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    #
    # FLANN_INDEX_LSH = 6
    # index_params = dict(algorithm=FLANN_INDEX_LSH,
    #                     table_number=6,  # 12
    #                     key_size=12,  # 20
    #                     multi_probe_level=1)  # 2
    search_params = dict(checks=50)  # or pass empty dictionary
    flann = cv.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)
    # Need to draw only good matches, so create a mask
    matchesMask = [[0, 0] for i in range(len(matches))]
    # ratio test as per Lowe's paper
    for i, (m, n) in enumerate(matches):
        if m.distance < 0.7 * n.distance:
            matchesMask[i] = [1, 0]
    print("took {}s to run".format(time.time()-start_time))

    draw_params = dict(matchColor=(0, 255, 0),
                       singlePointColor=(255, 0, 0),
                       matchesMask=matchesMask,
                       flags=cv.DrawMatchesFlags_DEFAULT)
    img3 = cv.drawMatchesKnn(img1, kp1, img2, kp2, matches, None, **draw_params)
    plt.imshow(img3, ), plt.show()

def sift_feasture_detection(query_img_path, train_img_path, crop):
    img1, img2, center = image_preprocessing(query_img_path, train_img_path, crop)
    start_time = time.time()

    # Initiate SIFT detector
    sift = cv.xfeatures2d.SIFT_create()

    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)
    # BFMatcher with default params
    bf = cv.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)

    # Apply ratio test
    good = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good.append([m])
    # cv.drawMatchesKnn expects list of lists as matches.
    print("took {}s to run".format(time.time()-start_time))
    # img3 = cv.drawMatchesKnn(img1, kp1, img2, kp2, good, None, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    # plt.imshow(img3), plt.show()
    return kp1, kp2, good, img2, center


def draw(img, center, pts):
    # pts = imgpts.astype("int32")
    img = cv.line(img, center, tuple(pts[0].ravel()), (255, 0, 0), 5)
    img = cv.line(img, center, tuple(pts[1].ravel()), (0, 255, 0), 5)
    img = cv.line(img, center, tuple(pts[2].ravel()), (0, 0, 255), 5)
    return img

def draw_axis(img, R, t, K):
    # unit is mm
    rotV, _ = cv.Rodrigues(R)
    points = np.float32(
        [[55, 0, 0], [0, 55, 0], [0, 0, 55], [0, 0, 0]]).reshape(-1, 3)
    axisPoints, _ = cv.projectPoints(points, rotV, t, K, (0, 0, 0, 0))
    img = cv.line(img, tuple(axisPoints[3].ravel()), tuple(
        axisPoints[0].ravel()), (255, 0, 0), 3)
    img = cv.line(img, tuple(axisPoints[3].ravel()), tuple(
        axisPoints[1].ravel()), (0, 255, 0), 3)
    img = cv.line(img, tuple(axisPoints[3].ravel()), tuple(
        axisPoints[2].ravel()), (0, 0, 255), 3)
    return img
def get_object_point(kp1, kp2, good):
    canonical2D_kp = [kp1[item[0].queryIdx].pt for item in good]
    image_points = [kp2[item[0].trainIdx].pt for item in good]

    object_points = [item+(0.0,) for item in canonical2D_kp]
    # object_points = [(0.0, item[0], item[1]) for item in canonical2D_kp]

    return np.array(object_points), np.array(image_points)

def test_feature():
    my_path = os.path.abspath(os.path.dirname(__file__))
    query_img_path = os.path.join(my_path, "dd2419_traffic_sign_pdfs", "dangerous_right.jpg")
    train_img_path = os.path.join(my_path, "dd2419_detector_baseline_OG/performance_test/test_images",
                                  "0000068.jpg")
    model_run = run_model_singleimage(train_img_path)[0][0]
    print(model_run)

    # sift_feasture_detection(query_img_path, query_img_path, model_run)
    sift_feasture_detection(query_img_path, train_img_path, model_run)
    # FLANN(query_img_path, train_img_path, model_run)
    # FLANN(train_img_path, train_img_path, model_run)

def get_orientation(camera_matrix):
    # def get_unique(points, delta=0.1):
    #     results = []
    #     for row in points:
    #         for stored_result in results:
    #             if stored_result[0] * (1-delta) <= row[0] <= stored_result[0] * (1+delta) and stored_result[1] * (1-delta) <= row[1] <= stored_result[1] * (1+delta):# and stored_result[2] * (1-delta) <= row[2] <= stored_result[2] * (1+delta):
    #                 break
    #         results.append(row)
    #     return np.array(results)

    my_path = os.path.abspath(os.path.dirname(__file__))
    query_img_path = os.path.join(my_path, "dd2419_traffic_sign_pdfs", "dangerous_right.jpg")
    train_img_path = os.path.join(my_path, "dd2419_detector_baseline_OG/performance_test/test_images",
                                  "0000070.jpg")
    # train_img_path = query_img_path
    model_run = run_model_singleimage(train_img_path)[0][0]
    print(model_run)
    model_run = None
    # sift_feasture_detection(query_img_path, query_img_path, model_run)
    kp1, kp2, good, img2, image_center = sift_feasture_detection(query_img_path, query_img_path, model_run)
    result_img = img2
    plt.imshow(img2), plt.show()
    # Harded for now but will be read form camera matrix
    # I got these values from camera info
    D = np.array([0.061687, -0.049761, -0.008166, 0.004284, 0.0])
    K = np.array([231.250001, 0.0, 320.519378, 0.0, 231.065552, 240.631482, 0.0, 0.0, 1.0]).reshape(3, 3)
    P = np.array([231.25, 0.0, 322.360322, 0.0, 0.0, 231.06, 240.631, 0.0, 0.0, 0.0, 1.0, 0.0]).reshape(3, 4)
    R = np.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]).reshape(3, 3)
    # TODO: replace camera values with a camera
    # D = np.array([0.0, 0.0, 0.0, 0.0, 0.0])

    axis = np.float32([[-3, 0, 0], [0, 3, 0], [0, 0, 3]]).reshape(-1, 3)

    camera_matrix = K
    dist_coeffs = D

    object_points, image_points = get_object_point(kp1, kp2, good)
    # object_points = get_unique(object_points)
    # image_points = get_unique(image_points)
    objp = np.zeros((4 * 4, 3), np.float32)
    objp[:, :2] = np.mgrid[0:4, 0:4].T.reshape(-1, 2)
    # image_points = get_unique(image_points)



    retval, rvec, tvec, inliers = cv.solvePnPRansac(objp,
                                                    objp[:, :2], camera_matrix, dist_coeffs)

    image_points, jacobian = cv.projectPoints(axis, rvec, tvec, camera_matrix, dist_coeffs)
    # image_points = np.array([[ 33.95569611, 180.4183197 ],  [112.67071533, 144.4498291 ], [131.08009338, 114.48181152]])
    #
    result_img = draw(img2, image_center, image_points)
    # result_img = draw_axis(img2, R, tvec, K)
    cv.imshow('result_img', result_img)
    plt.imshow(result_img), plt.show()
if __name__ == "__main__":
    # test_feature()
    get_orientation(None)



