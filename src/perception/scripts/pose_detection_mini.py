import os.path
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

def image_preprocessing(img_path):
    img = cv.imread(img_path)  # queryImage
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    image_center = (img.shape[0]/2, img.shape[1]/2)
    return img, image_center

def draw(img, corners, imgpts):
    img = cv.line(img, corners, tuple(imgpts[0].ravel()), (255, 0, 0), 5)
    img = cv.line(img, corners, tuple(imgpts[1].ravel()), (0, 255, 0), 5)
    img = cv.line(img, corners, tuple(imgpts[2].ravel()), (0, 0, 255), 5)
    return img

def get_orientation():
    my_path = os.path.abspath(os.path.dirname(__file__))
    query_img_path = os.path.join(my_path, "dd2419_traffic_sign_pdfs", "dangerous_right.jpg")

    img, image_center = image_preprocessing(query_img_path)

    plt.imshow(img), plt.show()
    # Hard coded for now but will be read form camera matrix
    # I got these values from camera info
    D = np.array([0.061687, -0.049761, -0.008166, 0.004284, 0.0])
    K = np.array([231.250001, 0.0, 320.519378, 0.0, 231.065552, 240.631482, 0.0, 0.0, 1.0]).reshape(3, 3)
    P = np.array([231.25, 0.0, 322.360322, 0.0, 0.0, 231.06, 240.631, 0.0, 0.0, 0.0, 1.0, 0.0]).reshape(3, 4)
    R = np.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]).reshape(3, 3)
    # TODO: replace camera values with a camera
    # D = np.array([0.0, 0.0, 0.0, 0.0, 0.0])

    axis = np.float32([[3, 0, 0], [0, 3, 0], [0, 0, -3]]).reshape(-1, 3)

    camera_matrix = K
    dist_coeffs = D

    image_points = np.array([[280, 61], [69, 429], [494, 428], [315, 214], [232, 407], [198, 278], [280, 3]], np.float32)

    object_points = np.zeros((image_points.shape[0], 3), np.float32)
    object_points[:, :2] = image_points / 10.0

    retval, rvec, tvec = cv.solvePnP(object_points, image_points, camera_matrix, dist_coeffs)

    imgpts, jacobian = cv.projectPoints(axis, rvec, tvec, camera_matrix, dist_coeffs)

    result_img = draw(img, image_center, imgpts)
    plt.imshow(result_img), plt.show()

if __name__ == "__main__":
    get_orientation()



