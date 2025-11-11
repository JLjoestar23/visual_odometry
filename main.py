"""
Main script to run visual odometry on the KITTI datset.
"""

import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
import os
from essential_matrix import find_essential_mat, test_epipolar_constraint


def main():
    """
    Main script function to run visual odometry on KITTI seqeunce.

    Set the path to the sequence camera with DATA_DIR.

    Loads one frame at a time and computes the camera difference between the
    current frame and previous frame using the essential matrix. See README.md
    for detailed explanation.

    Displays the current frame and trajectory while running. Afterwords, it
    plots the estimated trajectory with the ground truth trajectory.
    """
    # directory
    DATA_DIR = "dataset/sequences/00"

    # define list of images
    img_list = sorted(
        os.listdir(f"{DATA_DIR}/image_0"), key=lambda x: int(x.split(".")[0])
    )

    # print(img_list)
    # print(img_list[0])
    # print(len(img_list))

    # camera calibration matrix
    # gathered from calib.txt p0 (left grascale camera)
    K = np.array([[718.856, 0.0, 607.1928], [0.0, 718.856, 185.2157], [0.0, 0.0, 1.0]])

    # trigger a redetection whenever the total number of keypoints go below a
    # certain threshold
    kp_redetect_threshold = 2000

    # load ground truth data
    true_poses = load_poses(DATA_DIR)
    # print(true_poses)
    prev_true_pose = true_poses[0, 0:3, 3]
    # print(prev_true_pose.shape)

    # initialize FAST feature detector object
    fast = cv.FastFeatureDetector_create(threshold=20, nonmaxSuppression=True)

    # initialize the first frame of data
    prev_frame = cv.imread(f"{DATA_DIR}/image_0/{img_list[0]}", cv.IMREAD_GRAYSCALE)
    prev_kp = keypoint_detection(fast, prev_frame)

    # initialize positions and rotations
    positions = np.zeros((len(img_list), 3, 1))
    rotations = np.zeros((len(img_list), 3, 3))
    rotations[0] = np.eye(3)

    # Intitialize live location plotting
    plt.ion()
    _, ax = plt.subplots()
    (line,) = ax.plot([], [], "b-")
    ax.set_xlim(left=-300, right=200)
    ax.set_ylim(bottom=-50, top=400)
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Z (m)")
    plt.title("Estimated Camera trajectory (top-down view)")

    # Show first frame
    cv.imshow("camera_feed", prev_frame)

    # loop through 1550 images
    for i, img in enumerate(img_list[1:1501], start=1):
        if img.lower().endswith(".png"):
            print(f"Processing Image #: {i}")

            # process the next frame
            next_frame = cv.imread(f"{DATA_DIR}/image_0/{img}", cv.IMREAD_GRAYSCALE)

            # Visualize current frame
            cv.imshow("camera_feed", next_frame)
            if cv.waitKey(1) & 0xFF == ord("q"):  # press q to quit
                break

            # LK optical flow to track keypoints into the next frame
            prev_kp, next_kp = keypoint_tracking(prev_frame, next_frame, prev_kp)

            # Find the essential
            essential_mat, _ = cv.findEssentialMat(
                next_kp, prev_kp, K, cv.RANSAC, prob=0.999, threshold=1.0
            )

            # Extract the  with SVD from esse
            _, R, t, mask_pose = cv.recoverPose(essential_mat, next_kp, prev_kp, K)

            # print(R)
            # print(t)
            # print(mask_pose)

            # update to only consist of keypoints that are consistent with the
            # recovered pose
            inliers = mask_pose != 0
            next_kp = next_kp[inliers[:, 0]]
            prev_kp = prev_kp[inliers[:, 0]]

            # print(inliers)
            # print(next_kp)
            # print(prev_kp)

            next_true_pose = true_poses[i, 0:3, 3]
            scale = np.sqrt(np.sum(np.square(np.abs(next_true_pose - prev_true_pose))))

            # print(scale)

            if scale < 0.15:
                print(f"Frame {i}, scale {scale}")

            # if metric-scale movement between frames is significant enough,
            # check if z (forward/backwards motion) is greater than in other
            # directions
            if scale > 0.15 and abs(t[2]) > abs(t[1]) and abs(t[2]) > abs(t[0]):
                # update poses at current frame idx
                rotations[i] = R @ rotations[i - 1]
                positions[i] = positions[i - 1] + scale * (rotations[i - 1] @ t)

                # print(positions[i])
            else:
                rotations[i] = rotations[i - 1]
                positions[i] = positions[i - 1]

            # Plot current location estimate
            line.set_data(positions[:i, 0, 0], positions[:i, 2, 0])
            plt.draw()
            plt.pause(0.01)

            # If there are too few keypoints due to movement, detect new ones
            if next_kp.shape[0] < kp_redetect_threshold:
                prev_kp = keypoint_detection(fast, next_frame)
            # Otherwise reshape existing ones from filter
            else:
                prev_kp = next_kp.reshape(-1, 1, 2)

            # print(rotations[i])
            # print(positions[i])

            # Update variables for next loop iteration
            prev_frame = next_frame
            prev_true_pose = next_true_pose

    # Turn off interactive plotting
    plt.ioff()

    # Plot the real trajectory vs estimated trajectory
    gt_positions = true_poses[0:1550, 0:3, 3]
    plt.plot(positions[:, 0, 0], positions[:, 2, 0], label="Estimated")
    plt.plot(gt_positions[:, 0], gt_positions[:, 2], label="Ground Truth")
    plt.xlabel("X (m)")
    plt.ylabel("Z (m)")
    plt.title("Camera trajectory (top-down view)")
    plt.legend()
    plt.axis("equal")
    plt.show()


def keypoint_detection(detector, gray_img):
    """
    Detect keypoints from grayscale image using given detector

    Args:
        detector (cv2 feature detector object): Detector algorithm to run
        gray_img (np.array): Grayscale image as numpy array
    """

    # find the keypoints and descriptors with FAST
    kp = detector.detect(gray_img, None)

    # Extract each keypoint point into numpy array
    pts = np.array([k.pt for k in kp], dtype=np.float32).reshape(-1, 1, 2)

    return pts


def keypoint_tracking(prev_frame, next_frame, prev_points):
    """
    Track the keypoints between frames using Lucas-Kanade optical flow

    After tracking the keypoints, it filters to only keypoints that were
    tracked.

    Args:
        prev_frame (np.array): Previous frame image as grayscale
        next_frame (np.array): Next frame image as grayscale
        prev_points (np.array): Keypoints from previous frame
    """

    # Parameters for lucas kanade optical flow
    lk_params = dict(
        winSize=(21, 21),
        maxLevel=3,
        criteria=(cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 30, 0.03),
        flags=0,
        minEigThreshold=1e-3,
    )

    # Perform optical flow
    next_points, status, _ = cv.calcOpticalFlowPyrLK(
        prev_frame, next_frame, prev_points, None, **lk_params
    )

    # Filter to only tracked keypoints
    good_prev = prev_points[status == 1]
    good_next = next_points[status == 1]

    return good_prev, good_next


def load_poses(folder):
    """
    Load ground truth poses from KITTI file.

    Args:
        folder (str): Path to image sequence folder
    """
    pose_file = f"{folder}/truth.txt"

    # Read and parse the poses
    poses = []
    with open(pose_file, "r", encoding="utf-8") as f:
        lines = f.readlines()

        for line in lines:
            t_w_cam0 = np.fromstring(line, dtype=float, sep=" ")
            t_w_cam0 = t_w_cam0.reshape(3, 4)
            t_w_cam0 = np.vstack((t_w_cam0, [0, 0, 0, 1]))
            poses.append(t_w_cam0)

    return np.array(poses)


# Run the script if not imported
if __name__ == "__main__":
    main()
