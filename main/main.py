import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
import os
from essential_matrix import findEssentialMat, test_epipolar_constraint


def main():
    # directory
    data_dir = "dataset/sequences/00"

    # define list of images
    img_list = sorted(os.listdir(f"{data_dir}/image_0"), key=lambda x: int(x.split('.')[0]))

    #print(img_list)
    #print(img_list[0])
    #print(len(img_list))

    # camera calibration matrix
    # gathered from calib.txt p0 (left grascale camera)
    K = np.array([[718.856, 0.0, 607.1928], 
                  [0.0, 718.856, 185.2157], 
                  [0.0, 0.0, 1.0]])

    # trigger a redetection whenever the total number of keypoints go below a 
    # certain threshold
    kp_redetect_threshold = 2000

    # load ground truth data
    true_poses = load_poses(data_dir)
    #print(true_poses)
    #plot_2d_trajectory(true_poses)
    prev_true_pose = true_poses[0, 0:3, 3]
    #print(prev_true_pose.shape)

    # initialize FAST feature detector object
    fast = cv.FastFeatureDetector_create(threshold=20, nonmaxSuppression=True)

    # initialize the first frame of data
    prev_frame = cv.imread(f"{data_dir}/image_0/{img_list[0]}", cv.IMREAD_GRAYSCALE)
    prev_kp = keypoint_detection(fast, prev_frame)

    # initialize positions and rotations
    positions = np.zeros((len(img_list), 3, 1))
    rotations = np.zeros((len(img_list), 3, 3))
    rotations[0] = np.eye(3)

    # loop through 1550 images
    for i, img in enumerate(img_list[1:1501], start=1):
        if img.lower().endswith(".png"):
            print(f"Processing Image #: {i}")
            
            # process the next frame
            next_frame = cv.imread(f"{data_dir}/image_0/{img}", cv.IMREAD_GRAYSCALE)

            # LK optical flow to track keypoints into the next frame
            prev_kp, next_kp = keypoint_tracking(prev_frame, next_frame, prev_kp)

            E, mask = cv.findEssentialMat(next_kp, prev_kp, K, cv.RANSAC, prob=0.999, threshold=1.0)

            num_inliers, R, t, mask_pose = cv.recoverPose(E, next_kp, prev_kp, K)

            #print(R)
            #print(t)
            #print(mask_pose)

            # update to only consist of keypoints that are consistent with the
            # recovered pose
            inliers = (mask_pose != 0)
            next_kp = next_kp[inliers[:, 0]]
            prev_kp = prev_kp[inliers[:, 0]]

            #print(inliers)
            #print(next_kp)
            #print(prev_kp)

            next_true_pose = true_poses[i, 0:3, 3]
            scale = np.sqrt(np.sum(np.square(np.abs(next_true_pose - prev_true_pose))))

            #print(scale)

            if scale < 0.15:
                print(f"Frame {i}, scale {scale}")

            # if metric-scale movement between frames is significant enough,
            # check if z (forward/backwards motion) is greater than in other
            # directions
            if scale > 0.15 and abs(t[2]) > abs(t[1]) and abs(t[2]) > abs(t[0]):
                # update poses at current frame idx
                rotations[i] = R @ rotations[i - 1]
                positions[i] = positions[i - 1] + scale * (
                    rotations[i - 1] @ t
                )  # 1 should be scale

                #print(positions[i])

            else:
                rotations[i] = rotations[i - 1]
                positions[i] = positions[i - 1]

            if next_kp.shape[0] < kp_redetect_threshold:
                prev_kp = keypoint_detection(fast, next_frame)
            else:
                prev_kp = next_kp.reshape(-1, 1, 2)

            #print(rotations[i])
            #print(positions[i])

            prev_frame = next_frame
            prev_true_pose = next_true_pose

    gt_positions = true_poses[0:1550, 0:3, 3]
    plt.plot(positions[:, 0, 0], positions[:, 2, 0], label="Estimated")
    plt.plot(gt_positions[:, 0], gt_positions[:, 2], label="Ground Truth")
    plt.xlabel("X (m)")
    plt.ylabel("Z (m)")
    plt.title("Camera trajectory (top-down view)")
    plt.legend()
    plt.axis("equal")
    plt.show()

def keypoint_detection(method, gray_img):

    # find the keypoints and descriptors with FAST
    kp = method.detect(gray_img, None)

    # restructure
    pts = np.array([k.pt for k in kp], dtype=np.float32).reshape(-1, 1, 2)

    return pts

def keypoint_tracking(prev_frame, next_frame, prev_points):

    # Parameters for lucas kanade optical flow
    lk_params = dict(winSize  = (21, 21),
                  maxLevel = 3,
                  criteria = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 30, 0.03),
                  flags = 0,
                  minEigThreshold = 1e-3)

    next_points, status, err = cv.calcOpticalFlowPyrLK(prev_frame, next_frame,
                                                        prev_points, None, 
                                                        **lk_params)

    good_prev = prev_points[status == 1]
    good_next = next_points[status == 1]

    return good_prev, good_next

def load_poses(folder):
    """Load ground truth poses (T_w_cam0) from file."""
    pose_file = f"{folder}/truth.txt"

    # Read and parse the poses
    poses = []
    with open(pose_file, "r", encoding="utf-8") as f:
        lines = f.readlines()

        for line in lines:
            T_w_cam0 = np.fromstring(line, dtype=float, sep=" ")
            T_w_cam0 = T_w_cam0.reshape(3, 4)
            T_w_cam0 = np.vstack((T_w_cam0, [0, 0, 0, 1]))
            poses.append(T_w_cam0)

    return np.array(poses)

def plot_2d_trajectory(poses):
    # Extract the translation vectors (camera positions)
    positions = poses[:, :3, 3]  # shape: (N, 3)
    
    # X and Z for top-down view
    x = positions[0:1500, 0]
    z = positions[0:1500, 2]

    plt.figure(figsize=(8, 6))
    plt.plot(x, z, '-o', markersize=2, color='blue')
    plt.xlabel('X [m]')
    plt.ylabel('Z [m]')
    plt.title('Ground Truth Trajectory (Top-Down View)')
    plt.axis('equal')  # keep aspect ratio so trajectory isn’t distorted
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main()