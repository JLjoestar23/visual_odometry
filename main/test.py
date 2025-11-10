import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
from essential_matrix import findEssentialMat

def main():
    # Points in image 1 (3x15 matrix)
    pts1_data = np.array([
        [61.4195, 102.1798, 150.0000, 68.3768, 106.2098, 150.0000, 74.3208, 109.6134, 150.0000, 176.0870, 196.1538, 174.0000, 192.8571, 172.2222, 190.0000],
        [124.4290, 136.1955, 150.0000, 167.2490, 181.1490, 197.2377, 203.8325, 219.1146, 236.6025, 127.4080, 110.0296, 170.7846, 150.0000, 207.7350, 184.6410],
        [1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000]
    ])

    # Points in image 2 (3x15 matrix)  
    pts2_data = np.array([
        [45.5272, 63.4568, 86.9447, 61.6620, 80.2653, 104.1468, 75.7981, 94.7507, 118.6606, 135.5451, 176.3357, 147.5739, 184.5258, 157.8633, 191.6139],
        [126.5989, 136.7293, 150.0000, 165.9997, 180.2731, 198.5963, 200.5196, 217.7991, 239.5982, 125.7710, 105.4355, 172.3407, 150.0000, 212.1766, 188.5687],
        [1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000]
    ])

    # Convert to proper format for findEssentialMat: (N, 2) arrays
    pts1 = pts1_data[:2, :]  # Take first two rows
    pts2 = pts2_data[:2, :]  # Same for second image

    # K taken from video lecture
    K = np.array([
        [300, 0, 150],
        [0, 300, 150],
        [0, 0, 1]
    ])

    # print("\nCamera matrix K:")
    # print(K)

    np.set_printoptions(suppress=True, precision=4)

    E_true = np.array([[0.0, -1.0, 0.0], [-0.3615, 0.0, -3.1415], [0.0, 3.0, 0.0]])
    print("\nTrue essential matrix:")
    print(E_true)

    # Now test your findEssentialMat function
    E_my = findEssentialMat(pts2, pts1, K)
    print("\nMy essential matrix:")
    print(E_my)

    # Compare with OpenCV
    E_cv, mask = cv.findEssentialMat(pts2, pts1, K, method=cv.RANSAC, threshold=1.0)
    print("\nOpenCV essential matrix:")
    print(E_cv)

    print("\nDifference:")
    print(E_my - E_cv)

    
    num_inliers, R, t, mask_pose = cv.recoverPose(E_true, pts2, pts1, K)
    print("\nTrue Transformation:")
    print(R)
    print(t)

    num_inliers, R, t, mask_pose = cv.recoverPose(E_my.T, pts2, pts1, K)
    print("\nMy Transformation:")
    print(R)
    print(t)

    num_inliers, R, t, mask_pose = cv.recoverPose(E_cv, pts2, pts1, K)
    print("\nCV Transformation:")
    print(R)
    print(t)

    #plot_cube_points(pts1, pts2)

    

def plot_cube_points(pts1, pts2):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    ax1.scatter(pts1[:, 0], pts1[:, 1], c='red', s=50)
    ax1.set_title('Image 1 - Cube Points')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.grid(True)
    ax1.invert_yaxis()  # Image coordinates: y increases downward
    
    ax2.scatter(pts2[:, 0], pts2[:, 1], c='blue', s=50)
    ax2.set_title('Image 2 - Cube Points')  
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.grid(True)
    ax2.invert_yaxis()
    
    plt.axis('equal')
    plt.show()

if __name__ == "__main__":
    main()