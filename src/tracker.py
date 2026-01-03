import numpy as np
from filterpy.kalman import KalmanFilter
from scipy.optimize import linear_sum_assignment


class Track:
    def __init__(self, bbox, track_id):
        self.track_id = track_id
        self.kf = self._init_kf(bbox)
        self.bbox = bbox
        self.age = 0
        self.hits = 1
        self.time_since_update = 0

    def _init_kf(self, bbox):
        kf = KalmanFilter(dim_x=7, dim_z=4)
        kf.F = np.array([
            [1, 0, 0, 0, 1, 0, 0],
            [0, 1, 0, 0, 0, 1, 0],
            [0, 0, 1, 0, 0, 0, 1],
            [0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 1],
        ])
        kf.H = np.eye(4, 7)
        kf.P *= 10
        kf.R *= 1
        kf.Q *= 0.01

        x1, y1, x2, y2 = bbox
        kf.x[:4] = np.array([[x1], [y1], [x2], [y2]])
        return kf

    def predict(self):
        self.kf.predict()
        self.age += 1
        self.time_since_update += 1
        self.bbox = self.kf.x[:4].reshape(-1)

    def update(self, bbox):
        self.time_since_update = 0
        self.hits += 1
        self.kf.update(np.array(bbox))
        self.bbox = self.kf.x[:4].reshape(-1)


class ByteTracker:
    def __init__(self, max_age=30, iou_threshold=0.3):
        self.max_age = max_age
        self.iou_threshold = iou_threshold
        self.tracks = []
        self.next_id = 0

    @staticmethod
    def iou(boxA, boxB):
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])

        inter = max(0, xB - xA) * max(0, yB - yA)
        areaA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
        areaB = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
        return inter / (areaA + areaB - inter + 1e-6)

    def update(self, detections):
        results = []

        for track in self.tracks:
            track.predict()

        det_bboxes = [d[:4] for d in detections]

        if len(self.tracks) and len(det_bboxes):
            iou_matrix = np.zeros((len(self.tracks), len(det_bboxes)))
            for i, t in enumerate(self.tracks):
                for j, d in enumerate(det_bboxes):
                    iou_matrix[i, j] = 1 - self.iou(t.bbox, d)

            row_idx, col_idx = linear_sum_assignment(iou_matrix)

            assigned_tracks = set()
            assigned_dets = set()

            for r, c in zip(row_idx, col_idx):
                if iou_matrix[r, c] < 1 - self.iou_threshold:
                    self.tracks[r].update(det_bboxes[c])
                    assigned_tracks.add(r)
                    assigned_dets.add(c)

            for i, d in enumerate(det_bboxes):
                if i not in assigned_dets:
                    self.tracks.append(Track(d, self.next_id))
                    self.next_id += 1

        else:
            for d in det_bboxes:
                self.tracks.append(Track(d, self.next_id))
                self.next_id += 1

        self.tracks = [
            t for t in self.tracks if t.time_since_update <= self.max_age
        ]

        for track in self.tracks:
            for d in detections:
                if np.allclose(track.bbox, d[:4], atol=5):
                    x1, y1, x2, y2, cls, conf = d
                    results.append([
                        int(x1), int(y1), int(x2), int(y2),
                        track.track_id,
                        cls,
                        conf
                    ])

        return results
