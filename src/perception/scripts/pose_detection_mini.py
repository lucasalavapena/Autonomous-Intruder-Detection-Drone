# import os.path
# import numpy as np
# import cv2 as cv
# import matplotlib.pyplot as plt
#
# def image_preprocessing(img_path):
#     img_og = cv.imread(img_path)  # queryImage
#     img = cv.cvtColor(img_og, cv.COLOR_BGR2GRAY)
#     image_center = (img.shape[0]/2, img.shape[1]/2)
#     return img, img_og, image_center
#
# def draw(img, starting_loc, imgpts):
#     """
#     draw axis on image
#     :param img:
#     :param starting_loc:
#     :param imgpts:
#     :return:
#     """
#     imgpts = imgpts.astype(np.int32)
#     img = cv.line(img, starting_loc, tuple(imgpts[0].ravel()), (255, 0, 0), 5)
#     img = cv.line(img, starting_loc, tuple(imgpts[1].ravel()), (0, 255, 0), 5)
#     img = cv.line(img, starting_loc, tuple(imgpts[2].ravel()), (0, 0, 255), 5)
#     return img
#
# def get_orientation(see_image_points=False):
#     my_path = os.path.abspath(os.path.dirname(__file__))
#     query_img_path = os.path.join(my_path, "dd2419_traffic_sign_pdfs", "G6_00138.jpg")
#     img, img_og, image_center = image_preprocessing(query_img_path)
#
#     plt.imshow(img), plt.show()
#
#     # Hard coded for now but will be read form camera matrix
#     # I got these values from camera info
#     D = np.array([0.061687, -0.049761, -0.008166, 0.004284, 0.0])
#     K = np.array([231.250001, 0.0, 320.519378, 0.0, 231.065552, 240.631482, 0.0, 0.0, 1.0]).reshape(3, 3)
#     P = np.array([231.25, 0.0, 322.360322, 0.0, 0.0, 231.06, 240.631, 0.0, 0.0, 0.0, 1.0, 0.0]).reshape(3, 4)
#     R = np.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]).reshape(3, 3)
#
#     # D = np.array([0.0, 0.0, 0.0, 0.0, 0.0]) # zero distortion
#
#     axis = np.float32([[3, 0, 0], [0, 3, 0], [0, 0, -3]]).reshape(-1, 3)
#
#     camera_matrix = K
#     dist_coeffs = D
#
#     image_points = np.array([[358, 132], [400, 203], [333, 257], [278, 210], [336, 202], [92, 69]], np.float32)
#
#     object_points = np.zeros((image_points.shape[0], 3), np.float32)
#     object_points[:, :2] = image_points# / 10.0
#
#     chess_3d = np.zeros((6 * 7, 3), np.float32)
#     chess_3d[:, :2] = np.mgrid[0:7, 0:6].T.reshape(-1, 2)
#
#     if see_image_points:
#         plt.imshow(img_og)
#         plt.scatter(image_points[:, 0], image_points[:, 1])
#         plt.show()
#
#     retval, rvec, tvec = cv.solvePnP(object_points, image_points, camera_matrix, dist_coeffs)
#
#     # Harded coded correct translation vector
#     # tvec = np.array([[0], [0], [0]], np.float64)
#     # tvec *= 0.
#
#     rotation_matrix, _ = cv.Rodrigues(rvec)
#     imgpts, jacobian = cv.projectPoints(axis, rvec, tvec, camera_matrix, dist_coeffs)
#
#     # image_center = (int(image_points[0][0]), int(image_points[0][1])) # for corner tutorial example
#     image_center = (355, 224) # hardcoded center
#     result_img = draw(img, image_center, imgpts)
#     plt.imshow(result_img), plt.show()
#
# if __name__ == "__main__":
#     get_orientation(see_image_points=True)
#


import os.path
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

def image_preprocessing(img_path):
    img_og = cv.imread(img_path)  # queryImage
    img = img_og#cv.cvtColor(img_og, cv.COLOR_BGR2GRAY)
    image_center = (img.shape[0]/2, img.shape[1]/2)
    return img, img_og, image_center

def draw(img, starting_loc, imgpts):
    """
    draw axis on image
    :param img:
    :param starting_loc:
    :param imgpts:
    :return:
    """
    imgpts = imgpts.astype(np.int32)
    img = cv.line(img, starting_loc, tuple(imgpts[0].ravel()), (255, 0, 0), 5)
    img = cv.line(img, starting_loc, tuple(imgpts[1].ravel()), (0, 255, 0), 5)
    img = cv.line(img, starting_loc, tuple(imgpts[2].ravel()), (0, 0, 255), 5)
    return img

def get_orientation(see_image_points=False):
    my_path = os.path.abspath(os.path.dirname(__file__))
    query_img_path = os.path.join(my_path, "dd2419_traffic_sign_pdfs",  "G6_00138.jpg")
    img, img_og, image_center = image_preprocessing(query_img_path)

    plt.imshow(img), plt.show()

    # Hard coded for now but will be read form camera matrix
    # I got these values from camera info
    D = np.array([0.061687, -0.049761, -0.008166, 0.004284, 0.0])
    K = np.array([231.250001, 0.0, 320.519378, 0.0, 231.065552, 240.631482, 0.0, 0.0, 1.0]).reshape(3, 3)
    P = np.array([231.25, 0.0, 322.360322, 0.0, 0.0, 231.06, 240.631, 0.0, 0.0, 0.0, 1.0, 0.0]).reshape(3, 4)
    R = np.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]).reshape(3, 3)

    # D = np.array([0.0, 0.0, 0.0, 0.0, 0.0]) # zero distortion

    axis = np.float32([[3, 0, 0], [0, 3, 0], [0, 0, -3]]).reshape(-1, 3)

    camera_matrix = K
    dist_coeffs = D

    image_points = np.array([[358, 132], [400, 203], [333, 257], [278, 210], [336, 202], [92, 69]], np.float32)

    object_points = np.zeros((image_points.shape[0], 3), np.float32)
    object_points[:, :2] = (image_points - image_points[4]) / 10.0

    print(object_points)

    chess_3d = np.zeros((6 * 7, 3), np.float32)
    chess_3d[:, :2] = np.mgrid[0:7, 0:6].T.reshape(-1, 2)

    if see_image_points:
        plt.imshow(img_og)
        plt.scatter(image_points[:, 0], image_points[:, 1])
        plt.show()

    retval, rvec, tvec, inliers = cv.solvePnPRansac(object_points.reshape(-1, 1, 3), image_points.reshape(-1, 1, 2), camera_matrix, dist_coeffs)

    print(rvec)
    print(tvec)

    # Harded coded correct translation vector
    # tvec = np.array([[0], [0], [0]], np.float64)
    # tvec *= 0.

    rotation_matrix, _ = cv.Rodrigues(rvec)
    imgpts, jacobian = cv.projectPoints(axis, rvec, tvec, camera_matrix, dist_coeffs)

    # image_center = (int(image_points[0][0]), int(image_points[0][1])) # for corner tutorial example
    image_center = (image_points[4][0], image_points[4][1]) # hardcoded center
    result_img = draw(img_og, image_center, imgpts)
    plt.imshow(result_img), plt.show()

if __name__ == "__main__":
    get_orientation(see_image_points=True)



