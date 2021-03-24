import cv2 as cv
import matplotlib.pyplot as plt
import os.path

my_path = os.path.abspath(os.path.dirname(__file__))
query_img_path = os.path.join(my_path, "dd2419_traffic_sign_pdfs", "dangerous_right_page-0001.jpg")
train_img_path = os.path.join(my_path, "scripts/dd2419_detector_baseline_OG/performance_test/test_images", "0000069.jpg")

img1 = cv.imread(query_img_path, cv.IMREAD_GRAYSCALE)  # queryImage
img2 = cv.imread(train_img_path, cv.IMREAD_GRAYSCALE)  # trainImage
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
img3 = cv.drawMatchesKnn(img1, kp1, img2, kp2, good, None, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
plt.imshow(img3), plt.show()
