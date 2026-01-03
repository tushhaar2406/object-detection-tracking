import cv2


def draw_tracks(
    frame,
    tracks,
    class_names=None,
    box_color=(0, 255, 0),
    id_color=(0, 0, 255),
    thickness=2,
    font_scale=0.6,
    show_ids=True
):
    """
    Draw bounding boxes and labels on frame.

    Args:
        frame (np.ndarray)
        tracks (list):
            [x1,y1,x2,y2,track_id,class_id,confidence]
        class_names (list or None)
        show_ids (bool): show tracking IDs or not
    """

    for t in tracks:
        x1, y1, x2, y2, track_id, cls_id, conf = t

        # bounding box
        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            box_color,
            thickness
        )

        # build label
        label_parts = []

        if show_ids and track_id != -1:
            label_parts.append(f"ID {track_id}")

        if class_names and cls_id < len(class_names):
            label_parts.append(class_names[cls_id])
        else:
            label_parts.append(f"class {cls_id}")

        label_parts.append(f"{conf:.2f}")

        label = " | ".join(label_parts)

        # label background
        (w, h), _ = cv2.getTextSize(
            label,
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            thickness
        )

        cv2.rectangle(
            frame,
            (x1, y1 - h - 8),
            (x1 + w + 4, y1),
            box_color,
            -1
        )

        # label text
        cv2.putText(
            frame,
            label,
            (x1 + 2, y1 - 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (255, 255, 255),
            thickness=1
        )

    return frame
