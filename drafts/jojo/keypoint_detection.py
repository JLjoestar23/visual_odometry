import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

def test_keypoint_detection(img_name, threshold, nonmaxSupression):
    img = cv.imread(img_name)

    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # Initialize ORB detector
    fast = cv.FastFeatureDetector_create(threshold=threshold, nonmaxSuppression=nonmaxSupression)

    # Find the keypoints and descriptors with FAST
    kp = fast.detect(img, None)

    # Draw keypoints on the image
    img_with_keypoints = cv.drawKeypoints(img, kp, None, color=(0, 255, 0))

    # Display the image with keypoints
    plt.imshow(cv.cvtColor(img_with_keypoints, cv.COLOR_BGR2RGB))
    plt.title('ORB Keypoints')
    plt.show()

    # Print the number of keypoints detected
    print(f'Number of keypoints detected: {len(kp)}')

if __name__ == "__main__":
    test_keypoint_detection('checkerboard_test.jpg', 10, True)