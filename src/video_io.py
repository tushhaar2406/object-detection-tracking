import cv2
import os
import glob

VALID_IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp")
VALID_VIDEO_EXTS = (".mp4", ".avi", ".mov", ".mkv")


def get_stream(input_cfg):
    """
    Unified frame generator.
    Supports:
      - single image
      - image folder
      - video file
      - live camera
    """
    input_type = input_cfg["type"]

    if input_type == "image":
        return _image_stream(input_cfg["path"])

    elif input_type == "image_folder":
        return _image_folder_stream(input_cfg["path"])

    elif input_type == "video":
        return _video_stream(input_cfg["path"])

    elif input_type == "camera":
        return _camera_stream(input_cfg["camera_id"])

    else:
        raise ValueError(f"Unsupported input type: {input_type}")


# ==========================
# Internal stream functions
# ==========================

def _image_stream(path):
    if not os.path.exists(path):
        raise FileNotFoundError(path)

    frame = cv2.imread(path)
    if frame is None:
        raise ValueError(f"Cannot read image: {path}")

    yield frame


def _image_folder_stream(folder):
    if not os.path.isdir(folder):
        raise NotADirectoryError(folder)

    images = []
    for ext in VALID_IMAGE_EXTS:
        images.extend(glob.glob(os.path.join(folder, f"*{ext}")))

    images = sorted(images)

    if not images:
        raise ValueError("No images found in folder")

    for img_path in images:
        frame = cv2.imread(img_path)
        if frame is not None:
            yield frame


def _video_stream(path):
    if not os.path.exists(path):
        raise FileNotFoundError(path)

    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {path}")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        yield frame

    cap.release()


def _camera_stream(camera_id):
    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera: {camera_id}")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        yield frame

    cap.release()
