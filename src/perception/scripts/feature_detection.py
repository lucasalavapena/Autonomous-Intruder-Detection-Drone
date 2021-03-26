import os.path

import cv2 as cv
import matplotlib.pyplot as plt
from dd2419_detector_baseline_OG.utils import run_model_singleimage

def sift_feasture_detection(query_img_path, train_img_path, crop):

    print(train_img_path)
    img1 = cv.imread(query_img_path, cv.IMREAD_GRAYSCALE)  # queryImage
    img2 = cv.imread(train_img_path, cv.IMREAD_GRAYSCALE)  # trainImage
    # img1 = img1[]
    plt.imshow(img1), plt.show()
    plt.imshow(img2), plt.show()

    height = crop["height"].item() * -1
    width = crop["width"].item()  # shitfix
    # bb[0][0]["width"] = torch.tensor(bb[0][0]["width"].item()*-1)
    # bb[0][0]["height"] = torch.tensor(bb[0][0]["height"].item()*-1) # shitfix
    top_x = int(round(crop['x'].item()))
    top_y = int(round(crop['y'].item()))
    bottom_x = int(top_x + round(width))
    bottom_y = int(top_y - round(height))

    # img2 = img2[top_y:bottom_y, top_x: bottom_x]
    # plt.imshow(img2)
    

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
                                  "0000069.jpg")
    model_run = run_model_singleimage(train_img_path)[0][0]
    print(model_run)

    sift_feasture_detection(query_img_path, train_img_path, model_run)

if __name__ == "__main__":
    test_feature()



