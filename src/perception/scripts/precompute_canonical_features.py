import os
import pickle
import cv2 as cv
import matplotlib.pyplot as plt

SCALING_FACTOR = 0.3333
DRONE_IMAGE_RATIO = (640, 480)
MY_PATH = os.path.abspath(os.path.dirname(__file__))

def image_preprocessing(img_path):
    img = cv.imread(img_path)
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # Resizeing images
    dsize = (int(round(DRONE_IMAGE_RATIO[0] * SCALING_FACTOR)), int(round(DRONE_IMAGE_RATIO[1] * SCALING_FACTOR)))
    img = cv.resize(img, dsize, interpolation=cv.INTER_AREA)

    # Filename
    label = img_path.split('/')[-1][:-4]

    return label, img

def feature_detection(img):
    # ---------- SIFT ----------
    # Initiate SIFT detector
    sift = cv.xfeatures2d.SIFT_create()

    # find the keypoints and descriptors with SIFT
    kp_sift, des_sift = sift.detectAndCompute(img, None)
    sift_data = [(p.pt, p.size, p.angle, p.response, p.octave,
                  p.class_id, d) for p, d in zip(kp_sift, des_sift)]

    # ---------- ORB ----------
    # Initiate ORB detector
    orb = cv.ORB_create() # 500, 1.2, 8, 31, 0, 2

    # find the keypoints and descriptors with SIFT
    kp_orb, des_orb = orb.detectAndCompute(img, None)
    orb_data = [(p.pt, p.size, p.angle, p.response, p.octave,
                p.class_id, d) for p, d in zip(kp_orb, des_orb)]


    # ---------- SURF ----------
    # Initiate SURF detector
    surf = cv.xfeatures2d.SURF_create()

    # find the keypoints and descriptors with SIFT
    kp_surf, des_surf = surf.detectAndCompute(img, None)
    surf_data = [(p.pt, p.size, p.angle, p.response, p.octave,
                p.class_id, d) for p, d in zip(kp_surf, des_surf)]


    # # ---------- BRIEF ----------
    # star = cv.xfeatures2d.StarDetector_create()
    # brief = cv.xfeatures2d.BriefDescriptorExtractor_create()

    # # find the keypoints with STAR
    # kp_brief = star.detect(img, None)

    # # compute the descriptors with BRIEF
    # kp_brief, des_brief = brief.compute(img, kp_brief)
    # if len(kp_brief) != 0: # BRIEF SOMETIMES BAD WATCH OUT WHEN USING
    #     brief_data = [(p.pt, p.size, p.angle, p.response, p.octave,
    #                 p.class_id, d) for p, d in zip(kp_brief, des_brief)]
    # else:
    #     brief_data = None

    keypoints = {
        'IMAGE': img,
        'CENTER': (img.shape[0] / 2, img.shape[1] / 2),
        'SIFT': sift_data, #{'kp': kp_sift, 'des': des_sift},
        'SURF': surf_data, #{'kp': kp_surf, 'des': des_surf},
        'ORB': orb_data #{'kp': kp_orb, 'des': des_orb},
        #'BRIEF': brief_data, #{'kp': kp_brief, 'des': des_brief}
    }

    return keypoints


def main():
    # Getting image paths and removing non-canonical images
    path_to_signs = MY_PATH + '/dd2419_traffic_sign_pdfs'
    _, _, filenames = next(os.walk(path_to_signs))
    filenames = [path_to_signs + '/' + x for x in filenames if not any(c.isdigit() for c in x)]

    signs = {}
    for filename in filenames:
        label, img = image_preprocessing(filename)
        signs[label] = img


    detected_features = {}
    for label, image in signs.items():
        detected_features[label] = feature_detection(image)

    # Pickling and dumping
    pickle_file = "features.pickle"
    with open(MY_PATH + '/' + pickle_file, 'wb') as handle:
        pickle.dump(detected_features, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return 0

if __name__ == "__main__":
    main()



