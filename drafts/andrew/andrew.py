import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

data_folder = "dataset/sequences/00"

percent_to_use = 0.25

num_images = int(percent_to_use * len(os.listdir(f"{data_folder}/image_0")))

# K from calib.txt p0 (left grascale camera)
K = np.array([[718.856, 0.0, 607.1928], [0.0, 718.856, 185.2157], [0.0, 0.0, 1.0]])

num_kps = 2000
num_resample_kps = num_kps // 2

prev_img = cv2.imread(f"{data_folder}/image_0/000000.png", cv2.IMREAD_GRAYSCALE)


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


def get_keypoints(orb_detector, img):
    keypoints = orb_detector.detect(img, None)

    prevPts = np.zeros((len(keypoints), 1, 2))

    # Extract X/Y coords
    for i, kp in enumerate(keypoints):
        prevPts[i, 0, 0] = kp.pt[0]
        prevPts[i, 0, 1] = kp.pt[1]

    return prevPts.astype(np.float32)


true_poses = load_poses(data_folder)
prev_true_pose = true_poses[0, 0:3, 3]

orb = cv2.ORB_create(nfeatures=num_kps)
prev_kps = get_keypoints(orb, prev_img)

positions = np.zeros((num_images, 3, 1))
rotations = np.zeros((num_images, 3, 3))
rotations[0] = np.eye(3)

for i in range(1, num_images):

    if i % 10 == 0:
        print(f"On {i} / {num_images}")

    im_num = (6 - len(str(i))) * "0" + str(i)
    # Load consecutive grayscale frames
    next_img = cv2.imread(f"{data_folder}/image_0/{im_num}.png", cv2.IMREAD_GRAYSCALE)

    # Compute optical flow
    next_kps, status, err = cv2.calcOpticalFlowPyrLK(prev_img, next_img, prev_kps, None)

    # Keep only valid points
    prev_kps = prev_kps[status == 1]
    next_kps = next_kps[status == 1]

    E, mask = cv2.findEssentialMat(
        next_kps, prev_kps, K, cv2.RANSAC, prob=0.999, threshold=1.0
    )

    num_inliers, R, t, mask_pose = cv2.recoverPose(E, next_kps, prev_kps, K)

    inliers = mask_pose == 1
    next_kps = next_kps[inliers[:, 0]]
    prev_kps = prev_kps[inliers[:, 0]]

    next_true_pose = true_poses[i, 0:3, 3]
    scale = np.sqrt(np.sum(np.square(np.abs(next_true_pose - prev_true_pose))))

    if scale < 0.15:
        print(f"Frame {i}, scale {scale}")

    if scale > 0.15 and abs(t[2]) > abs(t[1]) and abs(t[2]) > abs(t[0]):
        # yaw = np.arctan2(R[0, 2], R[2, 2])

        # if abs(yaw) > 0.12:
        #     print(f"Frame {i}, Yaw too high {yaw}")

        # if scale > 0.15 and abs(yaw) < 0.08:  #  and
        rotations[i] = R @ rotations[i - 1]
        positions[i] = positions[i - 1] + scale * (
            rotations[i - 1] @ t
        )  # 1 should be scale
    else:
        rotations[i] = rotations[i - 1]
        positions[i] = positions[i - 1]

    if next_kps.shape[0] < num_resample_kps:
        prev_kps = get_keypoints(orb, next_img)
    else:
        prev_kps = next_kps.reshape(-1, 1, 2)

    prev_img = next_img
    prev_true_pose = next_true_pose


plt.plot(positions[:, 0, 0], positions[:, 2, 0])
plt.plot(true_poses[:num_images, 0, 3], true_poses[:num_images, 2, 3])
plt.xlabel("X (m)")
plt.ylabel("Z (m)")
plt.title("Camera trajectory (top-down view)")
plt.axis("equal")
plt.show()
plt.show()
# # Convert for display
# next_rgb = cv2.cvtColor(next, cv2.COLOR_GRAY2RGB)

# # Draw motion vectors
# for (x1, y1), (x2, y2) in zip(good_old, good_new):
#     cv2.arrowedLine(
#         next_rgb, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 1, tipLength=0.3
#     )

# # Plot side by side
# plt.figure(figsize=(12, 6))
# plt.subplot(1, 2, 1)
# plt.imshow(prev, cmap="gray")
# plt.title("Previous Frame")
# plt.axis("off")

# plt.subplot(1, 2, 2)
# plt.imshow(next_rgb)
# plt.title("Next Frame with Optical Flow")
# plt.axis("off")

# plt.tight_layout()
# plt.show()
