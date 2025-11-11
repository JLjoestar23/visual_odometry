"""
Module with functions to calculate the essential matrix
"""

import numpy as np


def find_essential_mat(pts0, pts1, K):
    """
    Calculate the Essential Matrix based on correspondences bewteen 2 images.

    Args:
        pts0: 2xn numpy array of keypoints from image 0
        pts1: 2xn numpy array of keypoints from image 1
        K: 3x3 numpy camera calibration matrix

    Returns:
        E_final: 3x3 numpy array of the essential matrix
    """
    # normalize image points according to camera calibration matrix K
    pts0_norm = normalize_pts(pts0, K)
    pts1_norm = normalize_pts(pts1, K)

    pts0_cond, T0 = precondition_pts(pts0_norm)
    pts1_cond, T1 = precondition_pts(pts1_norm)

    # construct the constraint matrix A
    A = construct_costraint_mat(pts1_cond, pts0_cond)

    # solve Ae = 0 using SVD
    _, _, Vt = np.linalg.svd(A)
    # utilize right-most column vector in V
    E_vec = Vt[-1]
    E = E_vec.reshape(3, 3)

    # enforce rank 2 constraint
    # use SVD to get U and Vt for reconstruction
    U, S, Vt = np.linalg.svd(E)

    # ensure proper rotation matrices (det=+1)
    if np.linalg.det(U) < 0:
        U[:, -1] *= -1
    if np.linalg.det(Vt) < 0:
        Vt[-1, :] *= -1

    # enforce rank 2 by setting smallest singular value to 0
    S[2] = 0
    # reconstruct E with rank 2 constraint
    e_rank2 = U @ np.diag(S) @ Vt

    # undo pre-conditioning
    e_final = T0.T @ e_rank2 @ T1

    # normalize
    e_final /= np.linalg.norm(e_final)

    # ensure consistent sign (make E[2,2] positive if possible)
    if e_final[2, 2] < 0:
        e_final = -e_final

    return e_final


def normalize_pts(pts, K):
    """
    Normalize pixel coordinates using camera calibration matrix.

    Args:
        pts: 2xn numpy array of image keypoints
        K: 3x3 numpy array camera matrix

    Returns:
        pts_norm: 3xn numpy array of normalized keypoints
    """
    # define pts as nx1: [x_n, y_n, 1]
    pts = np.hstack((pts, np.ones((pts.shape[0], 1))))

    # multiply by inv(K) to normalize projection
    pts_norm = (np.linalg.inv(K) @ pts.T).T

    return pts_norm


def precondition_pts(pts_norm):
    """
    Translate centroid of the keypoints to the camera plane origin and scale so
    that average distance from origin is sqrt(2).

    Args:
        pts_norm: 3xn numpy array of normalized image keypoints

    Returns:
        pts_cond: 3xn numpy array of pre-conditioned keypoints
        T: 3x3 numpy array of the corresponding transformation matrix
    """
    # use only x, y coordinates
    pts_xy = pts_norm[:, :2] if pts_norm.shape[1] > 2 else pts_norm

    centroid = np.mean(pts_xy, axis=0)
    scale = np.sqrt(2) / np.mean(np.linalg.norm(pts_xy - centroid, axis=1))

    T = np.array(
        [[scale, 0, -scale * centroid[0]], [0, scale, -scale * centroid[1]], [0, 0, 1]]
    )

    # apply transformation to homogeneous coordinates
    pts_homo = np.hstack((pts_xy, np.ones((len(pts_xy), 1))))
    pts_cond = (T @ pts_homo.T).T

    return pts_cond, T


def construct_costraint_mat(pts0, pts1):
    """
    Constructs the A matrix for the 8-point constraint problem.

    Args:
        pts0: 2xn numpy array of pre-conditioned keypoints from image 0
        pts1: 2xn numpy array of pre-conditioned keypoints from image 1

    Returns:
        A: nx9 numpy array of the A matrix
    """
    A = []
    # A should be an nx9
    for i in range(pts1.shape[0]):
        x1, y1, _ = pts1[i]
        x0, y0, _ = pts0[i]
        A.append([x0 * x1, x0 * y1, x0, y0 * x1, y0 * y1, y0, x1, y1, 1])

    return np.array(A)


def test_epipolar_constraint(E, pts0, pts1, K):
    """
    Test how well the essential matrix satisfies the epipolar constraint.

    Args:
        E: nx9 numpy array essential matrix calculated from pts0 and pts1
        pts0: 2xn numpy array of keypoints from image 0
        pts1: 2xn numpy array of keypoints from image 1
        K: 3x3 numpy array camera matrix
    """
    pts0_norm = normalize_pts(pts0, K)
    pts1_norm = normalize_pts(pts1, K)

    # convert to homogeneous
    pts0_homo = np.hstack((pts0_norm, np.ones((pts0_norm.shape[0], 1))))
    pts1_homo = np.hstack((pts1_norm, np.ones((pts1_norm.shape[0], 1))))

    errors = []
    for i in range(len(pts0_homo)):
        error = pts1_homo[i] @ E @ pts0_homo[i]
        errors.append(abs(error))

    print(f"Mean epipolar error: {np.mean(errors):.6f}")
    print(f"Max epipolar error: {np.max(errors):.6f}")
