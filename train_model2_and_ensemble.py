"""
Road Damage Detection
Model-2 Training + Ensemble with Model-1
Windows / VS Code SAFE version
"""

import os
import shutil
import torch
import gc
import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO
from ensemble_boxes import weighted_boxes_fusion
import multiprocessing


def main():

    torch.cuda.empty_cache()
    gc.collect()

    DATA_YAML = "./rdd2022.yaml"
    TEST_IMAGES = r"C:/Users/woebe/College/shalini_hackathon/randomized_dataset/test/images"
""" 
    print("\n===== TRAINING MODEL 2 =====")

    TRAINING_CONFIG_2 = {
        "epochs": 30,
        "imgsz": 768,
        "batch": 10,
        "patience": 6,

        "optimizer": "AdamW",
        "lr0": 0.0008,
        "lrf": 0.01,

        "hsv_v": 0.6,
        "scale": 0.7,
        "mixup": 0.15,
        "copy_paste": 0.15,

        "workers": 8,
        "device": 0,
        "amp": True,

        "project": "runs/detect",
        "name": "train2",
        "exist_ok": True,
    }

    model2 = YOLO("yolov8m.pt")

    results2 = model2.train(
        data=DATA_YAML,
        **TRAINING_CONFIG_2
    )

    shutil.copy("runs/detect/train2/weights/best.pt")
    print("Model 2 saved as model2_best.pt") """

    model1 = YOLO("model1_best.pt")
    model2 = YOLO("best.pt")

    CONF = 0.25
    IOU = 0.45

    def ensemble_predict(img_path):
        img = cv2.imread(str(img_path))

        all_boxes, all_scores, all_labels = [], [], []

        for model in [model1, model2]:
            for flip in [False, True]:

                img2 = img.copy()
                if flip:
                    img2 = cv2.flip(img2, 1)

                r = model.predict(img2, imgsz=768, conf=CONF, iou=IOU, verbose=False)[0]

                if len(r.boxes) > 0:
                    boxes = r.boxes.xyxyn.cpu().numpy()
                    scores = r.boxes.conf.cpu().numpy()
                    labels = r.boxes.cls.cpu().numpy().astype(int)

                    if flip:
                        boxes[:, [0,2]] = 1 - boxes[:, [2,0]]

                    all_boxes.append(boxes)
                    all_scores.append(scores)
                    all_labels.append(labels)

        if len(all_boxes)==0:
            return [],[],[]

        boxes, scores, labels = weighted_boxes_fusion(
            all_boxes,
            all_scores,
            all_labels,
            weights=[0.6,0.4]*2,
            iou_thr=0.5,
            skip_box_thr=0.01
        )

        return boxes, scores, labels

    print("\n===== GENERATING ENSEMBLE PREDICTIONS =====")

    pred_dir = Path("predictions")
    pred_dir.mkdir(exist_ok=True)

    test_images = list(Path(TEST_IMAGES).glob("*"))

    for i,img in enumerate(test_images):

        if i % 50 == 0:
            print(f"Processing {i}/{len(test_images)}")

        boxes, scores, labels = ensemble_predict(img)

        lines = []
        for b,s,l in zip(boxes,scores,labels):
            x1,y1,x2,y2=b
            xc=(x1+x2)/2
            yc=(y1+y2)/2
            w=x2-x1
            h=y2-y1
            lines.append(f"{l} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f} {s:.6f}")

        with open(pred_dir/f"{img.stem}.txt","w") as f:
            f.write("\n".join(lines))

    shutil.make_archive("submission","zip","predictions")
    print("\nsubmission.zip created successfully!")

    print("\n===== PIPELINE COMPLETE =====")


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()


