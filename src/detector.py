from ultralytics import YOLO
import numpy as np
import os


class Detector:
    def __init__(self, model_path, conf_threshold=0.25, iou_threshold=0.45, classes=None):

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"YOLO model not found: {model_path}")

        self.model = YOLO(model_path)
        self.conf = conf_threshold
        self.iou = iou_threshold
        self.classes = classes if classes else None

    def detect(self, frame):
        """
        Returns:
        [x1, y1, x2, y2, class_id, confidence]
        """

        results = self.model(
            frame,
            conf=self.conf,
            iou=self.iou,
            classes=self.classes,
            verbose=False
        )

        detections = []

        if not results or results[0].boxes is None:
            return detections

        boxes = results[0].boxes

        xyxy = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy()
        clss = boxes.cls.cpu().numpy().astype(int)

        for (x1, y1, x2, y2), conf, cls_id in zip(xyxy, confs, clss):
            detections.append([
                int(x1),
                int(y1),
                int(x2),
                int(y2),
                cls_id,
                float(conf)
            ])

        return detections
