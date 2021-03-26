import os.path
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
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
    img2 = rotate_image(img2, 1)
    plt.imshow(img1), plt.show()
    plt.imshow(img2), plt.show()

    # height = crop["height"].item() * -1
    # width = crop["width"].item()  # shitfix
    # top_x = int(round(crop['x'].item()))
    # top_y = int(round(crop['y'].item()))
    # bottom_x = int(top_x + round(width))
    # bottom_y = int(top_y - round(height))
    #
    # img2 = img2[top_y: bottom_y, top_x: bottom_x]
    return img1, img2

def FLANN(query_img_path, train_img_path, crop):
    img1, img2 = image_preprocessing(query_img_path, train_img_path, crop)

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
    draw_params = dict(matchColor=(0, 255, 0),
                       singlePointColor=(255, 0, 0),
                       matchesMask=matchesMask,
                       flags=cv.DrawMatchesFlags_DEFAULT)
    img3 = cv.drawMatchesKnn(img1, kp1, img2, kp2, matches, None, **draw_params)
    plt.imshow(img3, ), plt.show()

def sift_feasture_detection(query_img_path, train_img_path, crop):
    img1, img2 = image_preprocessing(query_img_path, train_img_path, crop)

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
        if m.distance < 0.9 * n.distance:
            good.append([m])
    # cv.drawMatchesKnn expects list of lists as matches.
    img3 = cv.drawMatchesKnn(img1, kp1, img2, kp2, good, None, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    plt.imshow(img3), plt.show()


def test_feature():
    my_path = os.path.abspath(os.path.dirname(__file__))
    query_img_path = os.path.join(my_path, "dd2419_traffic_sign_pdfs", "dangerous_right_page-0001.jpg")
    train_img_path = os.path.join(my_path, "dd2419_detector_baseline_OG/performance_test/test_images",
                                  "0000070.jpg")
    model_run = run_model_singleimage(train_img_path)[0][0]
    print(model_run)

    sift_feasture_detection(query_img_path, query_img_path, model_run)
    # FLANN(query_img_path, train_img_path, model_run)

if __name__ == "__main__":
    test_feature()



