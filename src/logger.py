import csv
import json
import os
from datetime import datetime


class ResultLogger:
    def __init__(self, output_dir, formats=("csv", "json")):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        self.formats = formats
        self.csv_file = None
        self.csv_writer = None

        if "csv" in formats:
            self._init_csv()

        if "json" in formats:
            self.json_path = os.path.join(output_dir, "results.jsonl")

    def _init_csv(self):
        csv_path = os.path.join(self.output_dir, "results.csv")
        self.csv_file = open(csv_path, mode="w", newline="")
        self.csv_writer = csv.writer(self.csv_file)

        self.csv_writer.writerow([
            "timestamp",
            "frame_id",
            "track_id",
            "class_id",
            "confidence",
            "x1", "y1", "x2", "y2"
        ])

    def log(self, frame_id, tracks):
        """
        tracks: [x1,y1,x2,y2,track_id,class_id,confidence]
        """
        timestamp = datetime.utcnow().isoformat()

        for t in tracks:
            x1, y1, x2, y2, track_id, cls_id, conf = t

            row = {
                "timestamp": str(timestamp),
                "frame_id": int(frame_id),
                "track_id": int(track_id),
                "class_id": int(cls_id),
                "confidence": float(conf),
                "x1": int(x1),
                "y1": int(y1),
                "x2": int(x2),
                "y2": int(y2),
            }


            if self.csv_writer:
                self.csv_writer.writerow(row.values())

            if "json" in self.formats:
                with open(self.json_path, "a") as f:
                    f.write(json.dumps(row) + "\n")

    def close(self):
        if self.csv_file:
            self.csv_file.close()
