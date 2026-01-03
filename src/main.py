import argparse
import yaml
import cv2
import os

from video_io import get_stream
from detector import Detector
from tracker import ByteTracker
from visualizer import draw_tracks
from logger import ResultLogger
from utils.fps import FPS


# ------------------------
# Config helpers
# ------------------------

def load_config(path="config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def parse_args():
    parser = argparse.ArgumentParser(description="Object Detection & Tracking")

    group = parser.add_mutually_exclusive_group()
    group.add_argument("--image")
    group.add_argument("--images")
    group.add_argument("--video")
    group.add_argument("--camera", type=int)

    return parser.parse_args()


def override_input_config(cfg, args):
    if args.image:
        cfg["input"]["type"] = "image"
        cfg["input"]["path"] = args.image
    elif args.images:
        cfg["input"]["type"] = "image_folder"
        cfg["input"]["path"] = args.images
    elif args.video:
        cfg["input"]["type"] = "video"
        cfg["input"]["path"] = args.video
    elif args.camera is not None:
        cfg["input"]["type"] = "camera"
        cfg["input"]["camera_id"] = args.camera
    return cfg


# ------------------------
# Main pipeline
# ------------------------

def main():
    args = parse_args()
    config = load_config()
    config = override_input_config(config, args)

    input_type = config["input"]["type"]
    is_single_image = input_type == "image"

    # ------------------------
    # Output setup for image
    # ------------------------
    if is_single_image:
        os.makedirs("data/output", exist_ok=True)
        image_output_path = "data/output/annotated_image.jpg"

    # ------------------------
    # Detector
    # ------------------------
    det_cfg = config["detection"]
    detector = Detector(
        model_path=det_cfg["model_path"],
        conf_threshold=det_cfg["confidence_threshold"],
        iou_threshold=det_cfg.get("iou_threshold", 0.45),
        classes=det_cfg.get("classes")
    )

    # 🔑 IMPORTANT: class names from YOLO
    CLASS_NAMES = detector.model.names

    # ------------------------
    # Tracker (video/camera only)
    # ------------------------
    tracker = None
    if not is_single_image:
        tracker = ByteTracker(
            max_age=config["tracking"]["max_age"],
            iou_threshold=config["tracking"]["iou_threshold"]
        )

    # ------------------------
    # Logger (video/camera only)
    # ------------------------
    logger = None
    if not is_single_image:
        logger = ResultLogger(
            output_dir=config["logging"]["output_dir"],
            formats=config["logging"]["formats"]
        )

    # ------------------------
    # FPS (video/camera only)
    # ------------------------
    fps = FPS()

    # ------------------------
    # Input stream
    # ------------------------
    stream = get_stream(config["input"])

    # ------------------------
    # Video writer (disabled for image)
    # ------------------------
    writer = None
    save_video = config["output"]["save_video"] and not is_single_image
    output_path = config["output"]["video_path"]

    # ------------------------
    # Main loop
    # ------------------------
    for frame_id, frame in enumerate(stream):

        detections = detector.detect(frame)

        # IMAGE → no tracking IDs
        if is_single_image:
            tracks = []
            for d in detections:
                x1, y1, x2, y2, cls, conf = d
                tracks.append([x1, y1, x2, y2, -1, cls, conf])
        else:
            tracks = tracker.update(detections)

        # Draw boxes + labels
        draw_tracks(
            frame,
            tracks,
            class_names=CLASS_NAMES,
            show_ids=not is_single_image
        )

        # Save annotated image
        if is_single_image:
            cv2.imwrite(image_output_path, frame)
            print(f"[INFO] Annotated image saved at {image_output_path}")

        # Logging (video/camera only)
        if logger:
            logger.log(frame_id, tracks)

        # FPS overlay (video/camera only)
        if not is_single_image:
            fps.update()
            if config["performance"]["display_fps"]:
                cv2.putText(
                    frame,
                    f"FPS: {fps.average_fps():.2f}",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (0, 255, 0),
                    2
                )

        # Video writer init
        if save_video and writer is None:
            h, w = frame.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(
                output_path,
                fourcc,
                config["performance"]["target_fps"],
                (w, h)
            )

        if writer:
            writer.write(frame)

        # Display
        cv2.imshow("Output", frame)

        if is_single_image:
            cv2.waitKey(500)
            break
        else:
            if cv2.waitKey(1) & 0xFF == 27:
                break


    # ------------------------
    # Cleanup
    # ------------------------
    if writer:
        writer.release()

    if logger:
        logger.close()

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
