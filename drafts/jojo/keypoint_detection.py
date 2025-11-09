import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

def fast_detection(gray_img, threshold, nonmaxSuppression):

    # initialize FAST feature detector object
    fast = cv.FastFeatureDetector_create(threshold=threshold, nonmaxSuppression=nonmaxSuppression)

    # find the keypoints and descriptors with FAST
    kp = fast.detect(gray_img, None)

    return kp

def orb_detection(gray_img):

    # Initiate ORB detector
    orb = cv.ORB_create()
    
    # find the keypoints with ORB
    kp = orb.detect(gray_img,None)
    
    # compute the descriptors with ORB
    kp, des = orb.compute(gray_img, kp)

    return kp

def visualize_keypoint_detection(img, keypoints):

    # Draw keypoints on the image
    img_with_keypoints = cv.drawKeypoints(img, keypoints, None, color=(0, 255, 0))

    # Display the image with keypoints
    plt.imshow(cv.cvtColor(img_with_keypoints, cv.COLOR_BGR2RGB))
    plt.title('ORB Keypoints')
    plt.show()

    # Print the number of keypoints detected
    print(f'Number of keypoints detected: {len(keypoints)}')

if __name__ == "__main__":
    img_name = 'rafale.jpg'
    img = cv.imread(img_name)
    gray_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    kp = fast_detection(gray_img, 20, True)
    visualize_keypoint_detection(img, kp)