#! /usr/bin/env python3
from _corners import FrameCorners

__all__ = [
    'track_and_calc_colors'
]

from typing import List, Tuple

import numpy as np

from corners import CornerStorage
from data3d import CameraParameters, PointCloud, Pose
import sortednp as snp
import click
import frameseq
from _camtrack import *
import cv2

initialization_frames = 25
frame_refining_window = 25

findEssentialMat_params = dict(
    method=cv2.RANSAC,
    prob=0.9,
    threshold=0.1
)

triangulation_params = TriangulationParameters(
    max_reprojection_error=1,
    min_triangulation_angle_deg=1,
    min_depth=0.1
)

solvePnPRansac_params = dict(
    distCoeffs=None,
    iterationsCount=218
)

def _init_cloud(prev_corners, cur_corners, intrinsic_mat):
    correspondences = build_correspondences(prev_corners, cur_corners)

    E, mask = cv2.findEssentialMat(
        correspondences.points_1, correspondences.points_2, intrinsic_mat,
        **findEssentialMat_params
    )

    inlier_cnt, R, t, pose_mask = cv2.recoverPose(
        E, correspondences.points_1, correspondences.points_2, intrinsic_mat,
        mask=mask
    )

    if inlier_cnt == 0:
        return -1, None, None

    pose_mask = np.ma.make_mask(pose_mask).flatten()

    points, ids = triangulate_correspondences(
        correspondences,
        eye3x4(),
        pose_to_view_mat3x4(Pose(R.T, -t)),
        intrinsic_mat,
        triangulation_params,
        mask=pose_mask
    )

    return points.shape[0], points, ids

    # return inlier_cnt, points.T[return_mask], correspondences.ids[return_mask]

def _track_camera(corner_storage: CornerStorage,
                  intrinsic_mat: np.ndarray) \
        -> Tuple[List[np.ndarray], PointCloudBuilder]:

    cloud_size, cloud_points, cloud_ids = -1, None, None

    with click.progressbar(corner_storage[1:initialization_frames + 1],
                                         label='Initializing',
                                         length=min(len(corner_storage) - 1, initialization_frames)) as bar:

        for corners in bar:
            inliers, points, ids = _init_cloud(corner_storage[0], corners, intrinsic_mat)
            if inliers > cloud_size:
                cloud_size = points.size
                cloud_points = points
                cloud_ids = ids

    views = []
    cloud_builder = PointCloudBuilder()
    cloud_builder.add_points(cloud_ids, cloud_points)

    with click.progressbar(enumerate(corner_storage),
                           label='Tracking',
                           length=len(corner_storage)) as bar:
        refinement_frame = frame_refining_window

        for frame, corners in bar:
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

            if inliers_provided:
                intersection_ids = cloud_builder.ids[indices_1]
                outlier_ids = np.delete(intersection_ids, inliers, axis=0)

                cloud_builder.remove_ids(outlier_ids)

            views.append(rodrigues_and_translation_to_view_mat3x4(R, t))

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
