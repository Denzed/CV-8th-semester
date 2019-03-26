from typing import List

import cv2
import numpy as np
import sortednp as snp
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix

from _camtrack import PointCloudBuilder, view_mat3x4_to_pose, to_homogeneous, \
    rodrigues_and_translation_to_view_mat3x4
from corners import FrameCorners


def bundle_adjustment_sparsity(frame_count: int, point_count: int, frame_ids, point_ids):
    m = frame_count
    n = frame_count * 9 + point_count * 3
    mat = lil_matrix((m, n), dtype=int)

    for i in range(frame_count):
        for s in range(9):
            mat[i, frame_ids * 9 + s] = 1

        for s in range(3):
            mat[i, frame_count * 9 + point_ids * 3 + s] = 1

    return mat


def view_mat3x4_to_array(view_mat: np.ndarray) -> np.ndarray:
    pose = view_mat3x4_to_pose(view_mat)
    r_vec, *_ = cv2.Rodrigues(pose.r_mat)

    return np.hstack((r_vec.ravel(), pose.t_vec.ravel())).ravel()


def to_x(intrinsic_params: np.ndarray,
         view_mats: List[np.ndarray],
         pc_builder: PointCloudBuilder) -> np.ndarray:
    res = np.hstack((
        np.array([
            np.hstack((view_mat3x4_to_array(view), intrinsic_params))
            for view in view_mats
        ]).ravel(),
        pc_builder.points.ravel()
    )).ravel()
    return res


def from_x(x, frame_cnt, point_cnt):
    camera_params = x[:frame_cnt * 9].reshape((frame_cnt, 9))
    points3d = x[frame_cnt * 9:].reshape((point_cnt, 3))

    views = []
    for cam in camera_params:
        r_vec = cam[:3].reshape(3, 1)
        t_vec = cam[3:6].reshape(3, 1)
        views.append(rodrigues_and_translation_to_view_mat3x4(r_vec, t_vec))

    return views, points3d


def make_fun(
        intrinsic_mat: np.ndarray,
        frame_cnt: int,
        point_cnt: int,
        frame_ids: np.ndarray,
        point_ids: np.ndarray,
        points2d: np.ndarray):
    def fun(x):
        views, points3d = from_x(x, frame_cnt, point_cnt)
        views = np.array(views)

        points3d = to_homogeneous(points3d)
        points2d_ = np.array([
            intrinsic_mat @ view @ point for view, point in zip(views[frame_ids], points3d[point_ids])
        ])
        points2d_ /= points2d_[[2]]

        return (points2d - points2d_.T[:2].T).ravel()

    return fun


def run_bundle_adjustment(intrinsic_mat: np.ndarray,
                          corner_list: List[FrameCorners],
                          max_inlier_reprojection_error: float,
                          view_mats: List[np.ndarray],
                          pc_builder: PointCloudBuilder) -> List[np.ndarray]:
    frame_cnt = len(view_mats)
    point_cnt = pc_builder.points.shape[0]

    intrinsic_params = np.array([intrinsic_mat[0][0], 0, 0])
    x0 = to_x(intrinsic_params, view_mats, pc_builder)

    frame_ids = []
    point_ids = []
    points2d = []

    for frame, corners in enumerate(corner_list):
        _, (indices_1, indices_2) = snp.intersect(corners.ids.flatten(), pc_builder.ids.flatten(), indices=True)

        frame_point_cnt = indices_1.shape[0]
        frame_ids += [frame] * frame_point_cnt
        point_ids += indices_2.tolist()
        points2d += corners.points[indices_1].tolist()

    frame_ids = np.array(frame_ids)
    point_ids = np.array(point_ids)
    points2d = np.array(points2d)

    fun = make_fun(intrinsic_mat, frame_cnt, point_cnt, frame_ids, point_ids, points2d)

    mat = bundle_adjustment_sparsity(frame_cnt, point_cnt, frame_ids, point_ids)

    print(x0.shape, fun(x0).shape, mat.shape)

    res_x = least_squares(fun, x0,
                          jac_sparsity=mat,
                          verbose=2,
                          x_scale='jac',
                          ftol=max_inlier_reprojection_error,
                          method='trf') # TODO: from point count

    res_view_mats, res_pc = from_x(res_x, frame_cnt, point_cnt)

    pc_builder.update_points(pc_builder.ids, res_pc)

    return res_view_mats
