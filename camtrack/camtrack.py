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

initialization_frames = 50
frame_refining_window = 12
essential_vs_homography_difference_threshold = 0.3

findEssentialMat_params = dict(
    method=cv2.RANSAC,
    prob=0.99,
    threshold=0.1
)

triangulation_params = TriangulationParameters(
    max_reprojection_error=3,
    min_triangulation_angle_deg=2,
    min_depth=0.1
)

solvePnPRansac_params = dict(
    distCoeffs=None,
    iterationsCount=300,
    reprojectionError=1
)


def _initialize_cloud_by_two_frames(prev_corners, cur_corners, intrinsic_mat):
    correspondences = build_correspondences(prev_corners, cur_corners)

    failed = correspondences.points_1.shape[0] < 6 # http://answers.opencv.org/question/67951/essential-matrix-6x3-expecting-3x3/

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

        essential_inlier_ratio = np.sum(mask) / correspondences.points_1.shape[0]
        homography_inlier_ratio = np.sum(mask_H) / correspondences.points_1.shape[0]
        essential_vs_homography_difference = abs(essential_inlier_ratio - homography_inlier_ratio)
        print(essential_inlier_ratio, homography_inlier_ratio, essential_vs_homography_difference)

        failed |= essential_vs_homography_difference >= essential_vs_homography_difference_threshold

    inlier_cnt, R, t, pose_mask = [None] * 4
    if not failed:
        inlier_cnt, R, t, pose_mask = cv2.recoverPose(
            E, correspondences.points_1, correspondences.points_2, intrinsic_mat,
            mask=mask
        )

        failed |= inlier_cnt == 0

    if not failed:
        pose_mask = np.ma.make_mask(pose_mask).flatten()

        points, ids = triangulate_correspondences(
            correspondences,
            eye3x4(),
            pose_to_view_mat3x4(Pose(R.T, -t)),
            intrinsic_mat,
            triangulation_params,
            mask=pose_mask
        )
        print(inlier_cnt, '->', points.shape[0])

        return points.shape[0], points, ids

    return -1, None, None


def _initialize_cloud(corner_storage: CornerStorage, intrinsic_mat: np.ndarray) -> PointCloudBuilder:
    cloud_size, cloud_points, cloud_ids = -1, None, None

    for corners in corner_storage[1:initialization_frames + 1]:
        inliers, points, ids = _initialize_cloud_by_two_frames(corner_storage[0], corners, intrinsic_mat)
        if inliers > cloud_size:
            cloud_size = inliers
            cloud_points = points
            cloud_ids = ids

    print(f"Initial cloud size if {cloud_size}")

    cloud_builder = PointCloudBuilder()
    cloud_builder.add_points(cloud_ids, cloud_points)

    return cloud_builder


def _track_camera(corner_storage: CornerStorage,
                  intrinsic_mat: np.ndarray) \
        -> Tuple[List[np.ndarray], PointCloudBuilder]:

    views = []
    cloud_builder = _initialize_cloud(corner_storage, intrinsic_mat)

    refinement_frame = frame_refining_window

    for frame, corners in enumerate(corner_storage):
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
            outlier_ids = np.array([])

        views.append(rodrigues_and_translation_to_view_mat3x4(R, t))

        cloud_builder.remove_ids(outlier_ids)

        triangulated = 0
        if frame == refinement_frame:
            correspondences = build_correspondences(
                corner_storage[frame - frame_refining_window],
                corners
            )

            new_points, new_ids = triangulate_correspondences(
                correspondences,
                views[frame - frame_refining_window],
                views[-1],
                intrinsic_mat,
                triangulation_params
            )

            cloud_builder.add_points(new_ids, new_points)
            refinement_frame = frame + frame_refining_window
            triangulated = new_points.shape[0]

        print(f"Frame \t{frame}: in \t{inliers.shape[0]} | triangulated \t{triangulated} | total in cloud \t{cloud_builder.points.shape[0]}", end="\r")
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
