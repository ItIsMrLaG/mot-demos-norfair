# https://github.com/tryolabs/norfair/tree/master/demos/detectron2
# https://tryolabs.github.io/norfair/2.2/reference/filter/#norfair.filter.OptimizedKalmanFilterFactory
from __future__ import annotations

import argparse
import os

from norfair import Tracker, Video, draw_tracked_objects, FilterPyKalmanFilterFactory
from constants import DETECTRON2_CFG, DETECTRON2_MODEL_W
from detectors import CircleDetector, BackgroundCircleDetector, DetectronCarDetector


def get_detector(name: str) -> CircleDetector | BackgroundCircleDetector | DetectronCarDetector:
    if name == "CircleDetector":
        return CircleDetector()
    if name == "BackgroundCircleDetector":
        return BackgroundCircleDetector()
    if name == "DetectronCarDetector":
        return DetectronCarDetector(DETECTRON2_CFG, DETECTRON2_MODEL_W)
    raise ValueError


def main(file: str, detector_nm: str):
    detector = get_detector(detector_nm)
    video = Video(input_path=file, output_path=f"{detector.__class__.__name__}_{os.path.basename(file)}")
    tracker = Tracker(distance_function="euclidean", distance_threshold=100,
                      filter_factory=FilterPyKalmanFilterFactory(R=100))
    for frame in video:
        detections = detector(frame)
        tracked_objects = tracker.update(detections=detections)
        draw_tracked_objects(frame, tracked_objects)
        video.write(frame)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("detector_nm", type=str, choices=[
        "CircleDetector",
        "BackgroundCircleDetector",
        "DetectronCarDetector"
    ], help="Input video file")

    parser.add_argument("video_fl", type=str, help="Input video file")
    args = parser.parse_args()
    main(args.video_fl, args.detector_nm)
