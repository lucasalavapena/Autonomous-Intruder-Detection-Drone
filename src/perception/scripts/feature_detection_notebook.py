#%%
import os
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import time
from dd2419_detector_baseline_OG.utils import run_model_singleimage
%pylab inline

SCALING_FACTOR = 0.3333
DRONE_IMAGE_RATIO = (640, 480)
CWD = os.path.abspath(os.path.dirname(__file__))
axis = np.float32([[100, 0, 0], [0, 100, 0], [0, 0, 100]]).reshape(-1, 3)
RotX = np.float32([[1, 0, 0], [0, -1, 0], [0, 0, -1]]) # Rotation matrix about x-axis
D = np.array([0.061687, -0.049761, -0.008166, 0.004284, 0.0])
K = np.array([231.250001, 0.0, 320.519378, 0.0, 231.065552, 240.631482, 0.0, 0.0, 1.0]).reshape(3, 3)
P = np.array([231.25, 0.0, 322.360322, 0.0, 0.0, 231.06, 240.631, 0.0, 0.0, 0.0, 1.0, 0.0]).reshape(3, 4)
R = np.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]).reshape(3, 3)
# dsize = (int(round(DRONE_IMAGE_RATIO[0] * SCALING_FACTOR)), int(round(DRONE_IMAGE_RATIO[1] * SCALING_FACTOR)))
#     img = cv.resize(img, dsize, interpolation=cv.INTER_AREA)

def draw(img, corners, imgpts):
    img = cv.line(img, corners, tuple(imgpts[0].ravel()), (255, 0, 0), 5)
    img = cv.line(img, corners, tuple(imgpts[1].ravel()), (0, 255, 0), 5)
    img = cv.line(img, corners, tuple(imgpts[2].ravel()), (0, 0, 255), 5)
    return img

# %%
my_path = os.path.abspath(os.path.dirname(__file__))
canon_img_path = os.path.join(CWD, "dd2419_traffic_sign_pdfs/stop.jpg")
drone_img_path = os.path.join(CWD, "debug_photos/stop_angle05.jpg")
bounding_box = run_model_singleimage(drone_img_path, 0.5)[0][0]

bb_info = {
    "height": bounding_box["height"].item() * -1,  # shitfix
    "width": bounding_box["width"].item(),
    "top_x": int(round(bounding_box['x'].item())),
    "top_y": int(round(bounding_box['y'].item()))
}
bb_info["bottom_x"] = int(bb_info["top_x"] + round(bb_info["width"]))
bb_info["bottom_y"] = int(bb_info["top_y"] - round(bb_info["height"]))

# %%
canon_img = cv.imread(canon_img_path)  # queryImage
drone_img = cv.imread(drone_img_path)   # trainImage
drone_img_og = drone_img.copy()
# Convert to grayscsale
canon_img = cv.cvtColor(canon_img, cv.COLOR_BGR2GRAY)
drone_img = cv.cvtColor(drone_img, cv.COLOR_BGR2GRAY)


# # TODO not sure, take a look
center_in_drone_img = ((bb_info["top_x"] + bb_info["bottom_x"])/2,
                    (bb_info["top_y"] + bb_info["bottom_y"])/2)
# center_in_cropped_img = (
#     (bb_info["bottom_x"] - bb_info["top_x"])/2, (bb_info["bottom_y"] - bb_info["top_y"])/2)
# canon_center = ((canon_img.shape[0]) / 2, (canon_img.shape[1]) / 2)

# %%
# Resizeing canonical image
dsize = (int(round(DRONE_IMAGE_RATIO[0] * SCALING_FACTOR)),
         int(round(DRONE_IMAGE_RATIO[1] * SCALING_FACTOR)))
canon_img = cv.resize(canon_img, dsize, interpolation=cv.INTER_AREA)

# Initiate SIFT detector
surf = cv.xfeatures2d.SURF_create(100, 12, 12, False, False)

# find the keypoints and descriptors with SIFT
kp1, des1 = surf.detectAndCompute(canon_img, None)
kp2, des2 = surf.detectAndCompute(drone_img, None)

# BFMatcher with default params
bf = cv.BFMatcher()
matches = bf.knnMatch(des1, des2, k=2)

# Apply ratio test
good = []
for m, n in matches:
    if m.distance < 0.75 * n.distance:
        good.append([m])

display_result = True
if display_result:
    img3 = cv.drawMatchesKnn(canon_img, kp1, drone_img, kp2, good, None, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.imshow(img3)
    fig.savefig("matches.png", dpi=1000)

# %%
# Get points
canonical2D_kp = np.array([kp1[item[0].queryIdx].pt for item in good])
image_points = np.array([kp2[item[0].trainIdx].pt for item in good], dtype=np.float32)
object_points = np.zeros((image_points.shape[0], image_points.shape[1] + 1), dtype=np.float64)
object_points[:, :2] = canonical2D_kp
# TODO put this on hold
# object_points[:, :2] = (canonical2D_kp - object_center) / 10.0
object_points[:, :1] = object_points[:, :1] - canon_img.shape[1]/2
object_points[:, 1:2] = object_points[:, 1:2] - canon_img.shape[0]/2
a = None

# %%
#solve PnP
retval, rvec, tvec, inliers = cv.solvePnPRansac(object_points.reshape(-1, 1, 3),
                                                image_points.reshape(-1, 1, 2),
                                                K, D
                                                )

see_image_points = True
if see_image_points:
    plt.imshow(drone_img)
    plt.scatter(image_points[inliers, 0], image_points[inliers, 1])
    plt.show()
# %%

projected_axis, jacobian = cv.projectPoints(axis, rvec, tvec, K, D)

result_img = draw(drone_img_og, center_in_drone_img, projected_axis)
plt.imshow(result_img), plt.show()

# %%
# # Python implementation of C++ code below, not currently working but maybe a good start?
rodrigues, _ = cv.Rodrigues(rvec)
rvec_converted, _ = cv.Rodrigues(rodrigues.T)
rvec_converted = RotX * rvec_converted

tvec_converted = -rodrigues.T * tvec
tvec_converted = RotX * tvec_converted

rvec[0], rvec[1], rvec[2] = rvec_converted[0][0], rvec_converted[1][1], rvec_converted[2][2]
tvec[0], tvec[1], tvec[2] = tvec_converted[0][0], tvec_converted[1][1], tvec_converted[2][2]

result_img2 = draw(drone_img_og, center_in_drone_img, projected_axis)
plt.imshow(result_img2), plt.show()

# %%
