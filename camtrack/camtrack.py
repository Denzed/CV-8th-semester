#! /usr/bin/env python3

__all__ = [
    'track_and_calc_colors'
]

from typing import List, Tuple

import cv2
import numpy as np
import sortednp as snp

import frameseq
from _camtrack import *
from corners import CornerStorage
from data3d import CameraParameters, PointCloud, Pose

initialization_frames = 25
frame_refining_window = 10
essential_homography_ratio_threshold = 0.7

findEssentialMat_params = dict(
    method=cv2.RANSAC,
    prob=0.99,
    threshold=1
)

triangulation_params = TriangulationParameters(
    max_reprojection_error=1,
    min_triangulation_angle_deg=3,
    min_depth=0.1
)

solvePnPRansac_params = dict(
    distCoeffs=None,
    iterationsCount=239,
    reprojectionError=1
)


def _initialize_cloud_by_two_frames(prev_corners, cur_corners, intrinsic_mat):
    correspondences = build_correspondences(prev_corners, cur_corners)

    failed = correspondences.points_1.shape[0] <= 5 # http://answers.opencv.org/question/67951/essential-matrix-6x3-expecting-3x3/

    E, mask = [None] * 2
    if not failed:
        E, mask = cv2.findEssentialMat(
            correspondences.points_1, correspondences.points_2, intrinsic_mat,
            **findEssentialMat_params
        )

        H, mask_H = cv2.findHomography(
            correspondences.points_1,
            correspondences.points_2,
            method=findEssentialMat_params['method'],
            ransacReprojThreshold=triangulation_params.max_reprojection_error,
            maxIters=solvePnPRansac_params['iterationsCount'],
            confidence=findEssentialMat_params['prob']
        )

        essential_inliers = np.sum(mask)
        homography_inliers = np.sum(mask_H)
        failed |= essential_inliers / homography_inliers < essential_homography_ratio_threshold

    if not failed:
        mask = np.ma.make_mask(mask).flatten()
        R1, R2, t12 = cv2.decomposeEssentialMat(E)

        points, ids, view = np.array([]), np.array([]), None
        for R in [R1, R2]:
            for t in [t12.reshape(-1), -t12.reshape(-1)]:
                view_ = pose_to_view_mat3x4(Pose(R, t))

                points_, ids_ = triangulate_correspondences(
                    correspondences,
                    eye3x4(),
                    view_,
                    intrinsic_mat,
                    triangulation_params,
                    mask=mask
                )

                if points_.size > points.size:
                    points, ids, view = points_, ids_, view_

        return points.shape[0], view, points, ids

    return -1, None, None, None


def _initialize_cloud(corner_storage: CornerStorage, intrinsic_mat: np.ndarray) -> (int, np.ndarray, PointCloudBuilder):
    res = (-1, -1, None, None, None, None)

    for frame, corners in enumerate(corner_storage[1:initialization_frames + 1], start=1):
        inliers, view, points, ids = _initialize_cloud_by_two_frames(corner_storage[0], corners, intrinsic_mat)
        res_ = (inliers, frame, view, points, ids)
        if res_[0] > res[0]:
            res = res_

    size, frame, view, points, ids = res
    print(f"Initial cloud size if {size}")

    cloud_builder = PointCloudBuilder()
    cloud_builder.add_points(ids, points)

    return (frame, view, cloud_builder)


def _track_camera(corner_storage: CornerStorage,
                  intrinsic_mat: np.ndarray) \
        -> Tuple[List[np.ndarray], PointCloudBuilder]:

    views = [None] * len(corner_storage)
    (calibration_frame, calibration_view, cloud_builder) = \
        _initialize_cloud(corner_storage, intrinsic_mat)

    for frame, corners in enumerate(corner_storage):
        if frame == 0:
            views[frame] = eye3x4()
            continue
        elif frame == calibration_frame:
            views[frame] = calibration_view
            continue

        _, (indices_1, indices_2) = snp.intersect(
            cloud_builder.ids.flatten(),
            corners.ids.flatten(),
            indices=True
        )

        inliers_provided, R, t, inliers = cv2.solvePnPRansac(
            cloud_builder.points[indices_1], corners.points[indices_2],
            intrinsic_mat,
            **solvePnPRansac_params
        )

        intersection_ids = cloud_builder.ids[indices_1]
        if inliers_provided:
            outlier_ids = np.delete(intersection_ids, inliers, axis=0)
        else:
            outlier_ids = np.array([], dtype=intersection_ids.dtype)

        views[frame] = rodrigues_and_translation_to_view_mat3x4(R, t)

        cloud_builder.remove_ids(outlier_ids)

        triangulated = 0
        if frame >= frame_refining_window:
            other_frame = frame - frame_refining_window

            correspondences = build_correspondences(
                corner_storage[other_frame],
                corners
            )

            new_points, new_ids = triangulate_correspondences(
                correspondences,
                views[other_frame],
                views[frame],
                intrinsic_mat,
                triangulation_params
            )

            triangulated = new_points.shape[0]
            cloud_builder.add_points(new_ids, new_points)

        print(f"Frame \t{frame}: in \t{inliers.shape[0] if inliers is not None else 0} | triangulated \t{triangulated} | total in cloud \t{cloud_builder.points.shape[0]}", end="\r")
    print()

    return views, cloud_builder


def track_and_calc_colors(camera_parameters: CameraParameters,
                          corner_storage: CornerStorage,
                          frame_sequence_path: str) \
        -> Tuple[List[Pose], PointCloud]:
    rgb_sequence = frameseq.read_rgb_f32(frame_sequence_path)
    intrinsic_mat = to_opencv_camera_mat3x3(
        camera_parameters,
        rgb_sequence[0].shape[0]
    )
    view_mats, point_cloud_builder = _track_camera(
        corner_storage,
        intrinsic_mat
    )
    calc_point_cloud_colors(
        point_cloud_builder,
        rgb_sequence,
        view_mats,
        intrinsic_mat,
        corner_storage,
        5.0
    )
    point_cloud = point_cloud_builder.build_point_cloud()
    poses = list(map(view_mat3x4_to_pose, view_mats))
    return poses, point_cloud


if __name__ == '__main__':
    create_cli(track_and_calc_colors)()
