import cv2
import numpy as np
from ultralytics import YOLO
from ensemble_boxes import weighted_boxes_fusion

# ===============================
# LOAD MODELS
# ===============================
model1 = YOLO("model1_best.pt")
model2 = YOLO("runs/detect/train2/weights/best.pt")

CONF = 0.25
IOU = 0.45

# ===============================
# ENSEMBLE PREDICTION FUNCTION
# ===============================
def ensemble_predict(img_path):
    img = cv2.imread(str(img_path))
    if img is None:
        return [], [], []

    all_boxes, all_scores, all_labels = [], [], []

    for model in [model1, model2]:
        for flip in [False, True]:

            img2 = cv2.flip(img, 1) if flip else img

            result = model.predict(
                img2,
                imgsz=768,
                conf=CONF,
                iou=IOU,
                verbose=False
            )[0]

            if result.boxes is None or len(result.boxes) == 0:
                continue

            boxes = result.boxes.xyxyn.cpu().numpy()
            scores = result.boxes.conf.cpu().numpy()
            labels = result.boxes.cls.cpu().numpy().astype(int)

            if flip:
                boxes[:, [0, 2]] = 1 - boxes[:, [2, 0]]

            all_boxes.append(boxes)
            all_scores.append(scores)
            all_labels.append(labels)

    if not all_boxes:
        return [], [], []

    boxes, scores, labels = weighted_boxes_fusion(
    all_boxes,
    all_scores,
    all_labels,
    weights=[0.6, 0.4],   # one per model
    iou_thr=0.5,
    skip_box_thr=0.01
)


    return boxes, scores, labels
if __name__ == "__main__":
    test_img = r"C:\Users\woebe\College\shalini_hackathon\randomized_dataset\test\images\000019.jpg"   # <-- put an actual image path
    boxes, scores, labels = ensemble_predict(test_img)

    print("Boxes:", boxes)
    print("Scores:", scores)
    print("Labels:", labels)
