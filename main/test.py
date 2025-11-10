import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
from essential_matrix import findEssentialMat

def main():
    """Main function to estimate the essential matrix and visualize results."""
    # points in image 0 (3x15)
    pts0_data = np.array([
        [61.4195, 102.1798, 150.0000, 68.3768, 106.2098, 150.0000, 74.3208, 109.6134, 150.0000, 176.0870, 196.1538, 174.0000, 192.8571, 172.2222, 190.0000],
        [124.4290, 136.1955, 150.0000, 167.2490, 181.1490, 197.2377, 203.8325, 219.1146, 236.6025, 127.4080, 110.0296, 170.7846, 150.0000, 207.7350, 184.6410],
        [1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000]
    ])

    # points in image 1 (3x15)
    pts1_data = np.array([
        [45.5272, 63.4568, 86.9447, 61.6620, 80.2653, 104.1468, 75.7981, 94.7507, 118.6606, 135.5451, 176.3357, 147.5739, 184.5258, 157.8633, 191.6139],
        [126.5989, 136.7293, 150.0000, 165.9997, 180.2731, 198.5963, 200.5196, 217.7991, 239.5982, 125.7710, 105.4355, 172.3407, 150.0000, 212.1766, 188.5687],
        [1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000]
    ])

    # convert to proper format for findEssentialMat argument: (n, 2) arrays
    pts0 = pts0_data[:2, :].T  # take first two rows
    pts1 = pts1_data[:2, :].T  # same for second image

    # K taken from video lecture
    K = np.array([
        [300, 0, 150],
        [0, 300, 150],
        [0, 0, 1]
    ])

    # print("\nCamera matrix K:")
    # print(K)

    # don't print in sci-notation, 5 decimal points
    np.set_printoptions(suppress=True, precision=5)

    # ground truth
    E_true = np.array([[0.0, -1.0, 0.0], [-0.3615, 0.0, -3.1415], [0.0, 3.0, 0.0]])
    print("\nTrue essential matrix:")
    print(E_true)

    # test findEssentialMat function
    E_my = findEssentialMat(pts1, pts0, K)
    print("\nMy essential matrix:")
    print(E_my)

    # compare to OpenCV
    E_cv, mask = cv.findEssentialMat(pts1, pts0, K, method=cv.RANSAC, threshold=1.0)
    print("\nOpenCV essential matrix:")
    print(E_cv)

    # print("\nDifference:")
    # print(E_my - E_cv)
    
    # comparing transformation results between different E calculations
    # ground truth
    num_inliers, R, t, mask_pose = cv.recoverPose(E_true, pts1, pts0, K)
    print("\nTrue Transformation:")
    print(R)
    print(t)

    # our implementation
    num_inliers, R, t, mask_pose = cv.recoverPose(E_my, pts1, pts0, K)
    print("\nMy Transformation:")
    print(R)
    print(t)

    # openCV
    num_inliers, R, t, mask_pose = cv.recoverPose(E_cv, pts1, pts0, K)
    print("\nCV Transformation:")
    print(R)
    print(t)

    # visualize cube points
    plot_cube_points(pts0, pts1)

def plot_cube_points(pts0, pts1):
    """
    Plots the cube points from two images side by side.

    Args:
        pts0: 2xn numpy array of points from image 0
        pts1: 2xn numpy array of points from image 1
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    ax1.scatter(pts0[:, 0], pts0[:, 1], c='red', s=50)
    ax1.set_title('Image 0 - Cube Points')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.grid(True)
    ax1.invert_yaxis() # image coordinates: y increases downward

    ax2.scatter(pts1[:, 0], pts1[:, 1], c='blue', s=50)
    ax2.set_title('Image 1 - Cube Points')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.grid(True)
    ax2.invert_yaxis()
    
    plt.axis('equal')
    plt.show()

if __name__ == "__main__":
    main()