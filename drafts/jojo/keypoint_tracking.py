import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
from keypoint_detection import fast_detection

def feature_tracking(prev_frame, next_frame, prev_points):

    # Parameters for lucas kanade optical flow
    lk_params = dict(winSize  = (21, 21),
                  maxLevel = 3,
                  criteria = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 30, 0.03),
                  flags = 0,
                  minEigThreshold = 1e-3)

    next_points, status, err = cv.calcOpticalFlowPyrLK(prev_frame, next_frame, prev_points, None, **lk_params)

    good_prev = prev_points[status == 1]
    good_next = next_points[status == 1]

    return good_prev, good_next

def visualize_feature_tracking(prev_frame, next_frame, good_prev, good_next):
    
    x_prev, y_prev = good_prev[:, 0], good_prev[:, 1]
    x_next, y_next = good_next[:, 0], good_next[:, 1]

    u = x_next - x_prev
    v = y_next - y_prev

    fig, axes = plt.subplots(1, 2, figsize=(12,6))

    # Left: previous frame with keypoints
    axes[0].imshow(prev_frame, cmap='gray')
    axes[0].scatter(x_prev, y_prev, s=2, c='lime', marker='o')
    axes[0].set_title("Previous Frame (Detected Features)")

    # Right: next frame with motion vectors
    axes[1].imshow(next_frame, cmap='gray')
    axes[1].quiver(x_prev, y_prev, u, v, color='r', angles='xy', scale_units='xy', scale=1)
    axes[1].set_title("Next Frame (Optical Flow Vectors)")

    for ax in axes:
        ax.axis('off')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":

    prev_frame_name = '000000.png'
    next_frame_name = '000001.png'
    prev_frame = cv.imread(prev_frame_name)
    next_frame = cv.imread(next_frame_name)
    prev_frame_gray = cv.cvtColor(prev_frame, cv.COLOR_BGR2GRAY)
    next_frame_gray = cv.cvtColor(next_frame, cv.COLOR_BGR2GRAY)

    prev_points = fast_detection(prev_frame_gray, 25, True)

    prev_points = np.array([kp.pt for kp in prev_points], dtype=np.float32).reshape(-1, 1, 2)

    good_prev, good_next = feature_tracking(prev_frame_gray, next_frame_gray, prev_points)

    visualize_feature_tracking(prev_frame, next_frame, good_prev, good_next)
