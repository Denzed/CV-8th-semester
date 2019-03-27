#! /usr/bin/env python3

__all__ = [
    'FrameCorners',
    'CornerStorage',
    'build',
    'dump',
    'load',
    'draw',
    'without_short_tracks'
]

import click
import cv2
import numpy as np
import pims

from _corners import FrameCorners, CornerStorage, StorageImpl
from _corners import dump, load, draw, without_short_tracks, create_cli


class _CornerStorageBuilder:
    corner_detection_args = dict(
        useHarrisDetector=False,
        maxCorners=1179,
        qualityLevel=0.03,
        minDistance=8
    )

    lucas_kanade_args = dict(
        winSize=(8, 8),
        maxLevel=1,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
    )

    corner_discovery_window = 1

    def __init__(self, progress_indicator=None):
        self._progress_indicator = progress_indicator
        self._corners = dict()

    def set_corners_at_frame(self, frame, corners):
        self._corners[frame] = corners
        if self._progress_indicator is not None:
            self._progress_indicator.update(1)

    def build_corner_storage(self):
        return StorageImpl(item[1] for item in sorted(self._corners.items()))


def are_near(a, b, diff_threshold):
    return abs(a - b).reshape(-1, 2).max(-1) < diff_threshold


def _build_impl(frame_sequence: pims.FramesSequence,
                builder: _CornerStorageBuilder) -> None:
    image_0 = frame_sequence[0]

    corner_positions_0 = cv2.goodFeaturesToTrack(
        image_0,
        **_CornerStorageBuilder.corner_detection_args
    )
    last_id = len(corner_positions_0)
    corners_0 = FrameCorners(
        np.array(range(0, last_id)),
        corner_positions_0,
        np.array([5] * last_id)
    )
    builder.set_corners_at_frame(0, corners_0)

    for frame, image_1 in enumerate(frame_sequence[1:], 1):
        image_0_conv, image_1_conv = np.uint8(image_0 * 255), np.uint8(image_1 * 255)
        corner_positions_1, status_1, _ = cv2.calcOpticalFlowPyrLK(
            image_0_conv, image_1_conv, corners_0.points, None,
            **_CornerStorageBuilder.lucas_kanade_args
        )
        corner_positions_0, status_0, _ = cv2.calcOpticalFlowPyrLK(
            image_1_conv, image_0_conv, corner_positions_1, None,
            **_CornerStorageBuilder.lucas_kanade_args
        )

        found = are_near(corners_0.points, corner_positions_0, 1) * (1 == np.concatenate(status_0 * status_1))

        corners_1 = FrameCorners(
            corners_0.ids[found],
            corner_positions_1[found],
            corners_0.sizes[found]
        )

        if frame % _CornerStorageBuilder.corner_discovery_window == 0:
            mask = np.zeros_like(image_1).fill(255)
            for x, y in (np.int32(corner) for corner in corners_1.points):
                cv2.circle(
                    mask, (x, y),
                    _CornerStorageBuilder.corner_detection_args["minDistance"],
                    0, -1
                )
            left_to_max = max(
                0,
                _CornerStorageBuilder.corner_detection_args["maxCorners"] - corners_1.points.shape[0]
            )
            discovered_corner_positions = cv2.goodFeaturesToTrack(
                image_1,
                **_CornerStorageBuilder.corner_detection_args,
                mask=mask
            )[:left_to_max]
            if discovered_corner_positions.size != 0:
                discovered_corner_positions = np.concatenate(discovered_corner_positions)
                new_last_id = last_id + len(discovered_corner_positions)
                corners_1 = FrameCorners(
                    np.concatenate((corners_1.ids, np.array([[i] for i in range(last_id, new_last_id)]))),
                    np.concatenate((corners_1.points, discovered_corner_positions)),
                    np.concatenate((corners_1.sizes, np.array([[5] for _ in range(last_id, new_last_id)])))
                )
                last_id = new_last_id

        builder.set_corners_at_frame(frame, corners_1)
        image_0 = image_1
        corners_0 = corners_1


def build(frame_sequence: pims.FramesSequence,
          progress: bool = True) -> CornerStorage:
    """
    Build corners for all frames of a frame sequence.

    :param frame_sequence: grayscale float32 frame sequence.
    :param progress: enable/disable building progress bar.
    :return: corners for all frames of given sequence.
    """
    if progress:
        with click.progressbar(length=len(frame_sequence),
                               label='Calculating corners') as progress_bar:
            builder = _CornerStorageBuilder(progress_bar)
            _build_impl(frame_sequence, builder)
    else:
        builder = _CornerStorageBuilder()
        _build_impl(frame_sequence, builder)
    return builder.build_corner_storage()


if __name__ == '__main__':
    create_cli(build)()  # pylint:disable=no-value-for-parameter
