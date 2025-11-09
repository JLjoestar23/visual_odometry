import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
from keypoint_detection import fast_detection
from keypoint_tracking import feature_tracking

def main():
    prev_frame_name = '000000.png'
    next_frame_name = '000001.png'

    prev_frame = cv.imread(prev_frame_name)
    next_frame = cv.imread(next_frame_name)
    prev_frame_gray = cv.cvtColor(prev_frame, cv.COLOR_BGR2GRAY)
    next_frame_gray = cv.cvtColor(next_frame, cv.COLOR_BGR2GRAY)

    prev_points = fast_detection(prev_frame_gray, 25, True)

    prev_points = np.array([kp.pt for kp in prev_points], dtype=np.float32).reshape(-1, 1, 2)

    good_prev, good_next = feature_tracking(prev_frame_gray, next_frame_gray, prev_points)

    # K from calib.txt p0 (left grascale camera)
    K = np.array([[718.856, 0.0, 607.1928], [0.0, 718.856, 185.2157], [0.0, 0.0, 1.0]])

    E, mask = cv.findEssentialMat(good_next, good_prev, K, cv.RANSAC, prob=0.999, threshold=1.0)

    num_inliers, R, t, mask_pose = cv.recoverPose(E, good_next, good_prev, K)

    print("R:\n", R)
    print("t:\n", t)

if __name__ == "__main__":
    main()