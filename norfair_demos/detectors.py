from pathlib import Path
from typing import Any

import numpy as np
import cv2
from cv2 import Mat
from detectron2.config import get_cfg, CfgNode
from detectron2.engine import DefaultPredictor

from norfair import Detection


class BackgroundCircleDetector:
    def __init__(self):
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2()

    def __call__(self, frame: Mat | np.ndarray[Any, np.dtype] | np.ndarray):
        fg_mask = self.bg_subtractor.apply(frame)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)

        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        detections = []
        for contour in contours:
            ((x, y), radius) = cv2.minEnclosingCircle(contour)

            if 10 < radius < 50:
                detections.append([int(x), int(y)])

        return [Detection(points) for points in np.array(detections)]


class CircleDetector:
    def __call__(self, frame: Mat | np.ndarray[Any, np.dtype] | np.ndarray):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        circles = cv2.HoughCircles(
            gray,
            cv2.HOUGH_GRADIENT,
            dp=1.2,
            minDist=50,
            param1=50,
            param2=30,
            minRadius=10,
            maxRadius=50
        )

        detections = []
        if circles is not None:
            circles = np.round(circles[0, :]).astype("int")
            for (x, y, r) in circles:
                detections.append([x, y])  # Добавляем координаты центра круга

        return [Detection(points) for points in np.array(detections)]


class DetectronCarDetector:
    cfg: CfgNode
    detector: DefaultPredictor

    def __init__(self, cfg_yaml: Path, model_weights: Path):
        cfg = get_cfg()
        cfg.merge_from_file(str(cfg_yaml))
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.4
        cfg.MODEL.WEIGHTS = str(model_weights)
        self.cfg = cfg
        self.detector = DefaultPredictor(cfg)

    def __call__(self, frame: Mat | np.ndarray[Any, np.dtype] | np.ndarray):
        _detections = self.detector(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        return [
            Detection(p)
            for p, c in zip(
                _detections["instances"].pred_boxes.get_centers().cpu().numpy(),
                _detections["instances"].pred_classes,
            )
        ]
