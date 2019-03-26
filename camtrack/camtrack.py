#! /usr/bin/env python3

__all__ = [
    'track_and_calc_colors'
]

from typing import List, Tuple, Dict

import cv2
import numpy as np
import sortednp as snp

import frameseq
from _camtrack import *
from corners import CornerStorage
from data3d import CameraParameters, PointCloud, Pose


class TrackingMode:
    def __init__(self,
                 name: str,
                 triangulation_params: TriangulationParameters,
                 essential_homography_ratio_threshold: float,
                 essential_mat_params: Dict,
                 solve_pnp_ransac_params: Dict,
                 frame_refinement_window: int):
        self.name = name
        self.triangulation_params = triangulation_params
        self.essential_homography_ratio_threshold = essential_homography_ratio_threshold
        self.essential_mat_params = essential_mat_params
        self.solve_pnp_ransac_params = solve_pnp_ransac_params
        self.frame_refinement_window = frame_refinement_window


initialization_frames = 500

default_solve_pnp_ransac_params = dict(distCoeffs=None, iterationsCount=239, reprojectionError=3)

tracking_modes = [
    TrackingMode(
        "Strict",
        TriangulationParameters(max_reprojection_error=0.5, min_triangulation_angle_deg=5, min_depth=0.1),
        3,
        dict(method=cv2.RANSAC, prob=0.999, threshold=1),
        default_solve_pnp_ransac_params,
        10
    ),
    TrackingMode(
        "Mild",
        TriangulationParameters(max_reprojection_error=1, min_triangulation_angle_deg=3, min_depth=0.1),
        2,
        dict(method=cv2.RANSAC, prob=0.999, threshold=3),
        default_solve_pnp_ransac_params,
        5
    ),
    TrackingMode(
        "Very mild",
        TriangulationParameters(max_reprojection_error=3, min_triangulation_angle_deg=1, min_depth=0.1),
        0.95,
        dict(method=cv2.RANSAC, prob=0.999, threshold=5),
        default_solve_pnp_ransac_params,
        1
    ),
]


def _initialize_cloud_by_two_frames(
        prev_corners, cur_corners,
        intrinsic_mat, base_view,
        tracking_mode: TrackingMode
):
    correspondences = build_correspondences(prev_corners, cur_corners)

    failed = correspondences.points_1.shape[0] <= 5

    e, mask = [None] * 2
    if not failed:
        e, mask = cv2.findEssentialMat(
            correspondences.points_1, correspondences.points_2, intrinsic_mat,
            **tracking_mode.essential_mat_params
        )

        _, mask_h = cv2.findHomography(
            correspondences.points_1,
            correspondences.points_2,
            method=tracking_mode.essential_mat_params['method'],
            ransacReprojThreshold=tracking_mode.triangulation_params.max_reprojection_error,
            maxIters=tracking_mode.solve_pnp_ransac_params['iterationsCount'],
            confidence=tracking_mode.essential_mat_params['prob'],
            mask=np.copy(mask)
        )

        essential_inliers = np.sum(mask)
        homography_inliers = np.sum(mask_h)

        failed |= homography_inliers == 0 \
            or essential_inliers < tracking_mode.essential_homography_ratio_threshold * homography_inliers

    if not failed:
        mask = np.ma.make_mask(mask).flatten()
        r1, r2, t12 = cv2.decomposeEssentialMat(e)

        points, ids, view = np.array([]), np.array([]), None
        for R in [r1, r2]:
            for t in [t12, -t12]:
                view_ = np.hstack((R, t))

                points_, ids_ = triangulate_correspondences(
                    correspondences,
                    base_view,
                    view_,
                    intrinsic_mat,
                    tracking_mode.triangulation_params,
                    mask=mask
                )

                if points_.size > points.size:
                    points, ids, view = points_, ids_, view_

        return points.shape[0], view, points, ids

    return -1, None, None, None


def _initialize_cloud(
        corner_storage: CornerStorage,
        intrinsic_mat: np.ndarray,
        tracking_mode: TrackingMode
) -> (int, np.ndarray, PointCloudBuilder):
    res = (-1, -1, None, None, None)

    for frame, corners in enumerate(corner_storage[1:initialization_frames + 1], start=1):
        inliers, view, points, ids = _initialize_cloud_by_two_frames(
            corner_storage[0], corners,
            intrinsic_mat, eye3x4(),
            tracking_mode
        )
        res_ = (inliers, frame, view, points, ids)
        if res_[0] > res[0]:
            res = res_

    size, frame, view, points, ids = res

    if size <= 0:
        raise ValueError("initialization failed")
    print(f"Initial cloud size if {size}")

    cloud_builder = PointCloudBuilder()
    cloud_builder.add_points(ids, points)

    return frame, view, cloud_builder


def _update_cloud_by_two_frames(
        prev_corners, cur_corners,
        intrinsic_mat, prev_view, cur_view,
        tracking_mode: TrackingMode
):
    correspondences = build_correspondences(prev_corners, cur_corners)

    if correspondences.points_1.shape[0] <= 5:
        return -1, None, None

    points, ids = triangulate_correspondences(
        correspondences,
        prev_view,
        cur_view,
        intrinsic_mat,
        tracking_mode.triangulation_params,
    )

    return points.shape[0], points, ids


def _track_camera(corner_storage: CornerStorage,
                  intrinsic_mat: np.ndarray,
                  tracking_mode: TrackingMode) \
        -> Tuple[List[np.ndarray], PointCloudBuilder]:

    views = []
    (calibration_frame, calibration_view, cloud_builder) = \
        _initialize_cloud(corner_storage, intrinsic_mat, tracking_mode)

    refinement_frame = 0 + tracking_mode.frame_refinement_window
    for frame, corners in enumerate(corner_storage):
        if frame == 0:
            views.append(eye3x4())
            continue
        elif frame == calibration_frame:
            views.append(calibration_view)
            continue

        _, (indices_1, indices_2) = snp.intersect(
            cloud_builder.ids.flatten(),
            corners.ids.flatten(),
            indices=True
        )

        if indices_1.shape[0] < 4:
            print()
            raise ValueError("not enough points for solvePnPRansac")

        inliers_provided, r, t, inliers = cv2.solvePnPRansac(
            cloud_builder.points[indices_1],
            np.reshape(corners.points[indices_2], (-1, 1, 2)),
            intrinsic_mat,
            **tracking_mode.solve_pnp_ransac_params
        )

        intersection_ids = cloud_builder.ids[indices_1]
        if inliers_provided:
            outlier_ids = np.delete(intersection_ids, inliers, axis=0)
        else:
            outlier_ids = np.array([], dtype=intersection_ids.dtype)

        views.append(rodrigues_and_translation_to_view_mat3x4(r, t))

        cloud_builder.remove_ids(outlier_ids)

        triangulated = 0
        if frame >= refinement_frame:
            if tracking_mode.frame_refinement_window > 1:
                refinement_start = max(0, frame - 2 * tracking_mode.frame_refinement_window)
            else:
                refinement_start = max(0, frame - initialization_frames)
            for other_frame in range(refinement_start, frame):
                point_cnt, new_points, new_ids = _update_cloud_by_two_frames(
                    corner_storage[other_frame], corners,
                    intrinsic_mat, views[other_frame], views[frame],
                    tracking_mode
                )

                if point_cnt > 0:
                    _, (indices_1, indices_2) = snp.intersect(
                        cloud_builder.ids.flatten(),
                        new_ids.flatten(),
                        indices=True
                    )

                    triangulated += point_cnt - indices_2.shape[0]
                    cloud_builder.add_points(
                        np.delete(new_ids, indices_2, axis=0),
                        np.delete(new_points, indices_2, axis=0)
                    )

            refinement_frame = frame + tracking_mode.frame_refinement_window

        inlier_cnt = inliers.shape[0] if inliers is not None else 0
        in_cloud_cnt = cloud_builder.points.shape[0]
        print(
            f"Frame {frame: 4}: in {inlier_cnt: 4} of {intersection_ids.shape[0]} "
            f"| triangulated \t{triangulated: 4} "
            f"| total in cloud \t{in_cloud_cnt}"
        )

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

    for mode in tracking_modes:
        try:
            print(f"trying \"{mode.name}\" mode...")
            view_mats, point_cloud_builder = _track_camera(
                corner_storage,
                intrinsic_mat,
                mode
            )
            print(f"trying \"{mode.name}\" mode... Success!")
            break
        except ValueError as error:
            print(f"trying \"{mode.name}\" mode... Failed with {error.args}!")
    else:
        raise ValueError("all tracking modes failed")
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
