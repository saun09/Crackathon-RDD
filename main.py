"""
Road Damage Detection - Optimized Training Pipeline for Kaggle GPU
Estimated total training time: 2-3 hours for competitive mAP
"""

import os
import yaml
import shutil
from pathlib import Path
import numpy as np
from collections import Counter

def main():
  print("=" * 70)
  print("ROAD DAMAGE DETECTION - OPTIMIZED TRAINING PIPELINE")
  print("=" * 70)


  from ultralytics import YOLO
  import torch

  import gc
  torch.cuda.empty_cache()
  gc.collect()
  print("\n✓ GPU memory cleared")

  print(f"\n✓ GPU Available: {torch.cuda.is_available()}")
  if torch.cuda.is_available():
      print(f"✓ GPU Name: {torch.cuda.get_device_name(0)}")
      print(f"✓ GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
      print(f"✓ GPU Memory Free: {torch.cuda.mem_get_info()[0] / 1e9:.2f} GB")

 
  # Class 0 (Longitudinal): 21900 
  # Class 1 (Transverse): 9963
  # Class 3 (Other): 9018
  # Class 2 (Alligator): 8922
  # Class 4 (Pothole): 5535

  data_yaml = r"""
  path: C:/Users/woebe/College/shalini_hackathon/randomized_dataset
  train: train/images
  val: val/images
  test: test/images

  nc: 5
  names:
    0: Longitudinal_Crack
    1: Transverse_Crack
    2: Alligator_Crack
    3: Other_Corruption
    4: Pothole
  """


  with open('rdd2022.yaml', 'w') as f:
      f.write(data_yaml)

  print("\n✓ Data configuration created: rdd2022.yaml")

  import os
  base_path = r'C:\Users\woebe\College\shalini_hackathon\randomized_dataset'
  train_path = os.path.join(base_path, 'train/images')
  val_path = os.path.join(base_path, 'val/images')
  test_path = os.path.join(base_path, 'test/images')

  print(f"✓ Checking dataset paths:")
  print(f"  Train exists: {os.path.exists(train_path)}")
  print(f"  Val exists: {os.path.exists(val_path)}")
  print(f"  Test exists: {os.path.exists(test_path)}")



  TRAINING_CONFIG = {
    
      'model_name': 'yolov8m.pt',  
      
      'epochs': 30,  
      'imgsz': 768,
      'batch': 10,

      'patience': 6,  # 
      
      'optimizer': 'AdamW',
      'lr0': 0.001,  # Initial learning rate
      'lrf': 0.01,   # Final learning rate (lr0 * lrf)
      'momentum': 0.937,
      'weight_decay': 0.0005,
      'warmup_epochs': 3,
      'warmup_momentum': 0.8,
      'warmup_bias_lr': 0.1,
      
      'hsv_h': 0.015,  
      'hsv_s': 0.7,   
      'hsv_v': 0.4,    
      'degrees': 0.0,  
      'translate': 0.1,
      'scale': 0.5, 
      'shear': 0.0,   
      'perspective': 0.0,
      'flipud': 0.0,  
      'fliplr': 0.5,  
      'mosaic': 1.0,   
      'mixup': 0.1,   
      'copy_paste': 0.1,  
      'close_mosaic': 5, 
      
      'box': 7.5,
      'cls': 0.5,
      'dfl': 1.5,
      
      'workers': 8,
      'device': 0, 
      'amp': True, 
      'cache': False,  
      'rect': False, 
      
      'val': True,
      'plots': True,
      'save': True,
      'save_period': -1,  
      'exist_ok': True,
      'verbose': True,
  }

  print("\n✓ Training configuration set (P100 GPU - ULTRA MEMORY SAFE)")
  print(f"  Model: {TRAINING_CONFIG['model_name']} (Medium - Best balance)")
  print(f"  Epochs: {TRAINING_CONFIG['epochs']} (early stopping at {TRAINING_CONFIG['patience']})")
  print(f"  Image size: {TRAINING_CONFIG['imgsz']}px (Memory-safe)")
  print(f"  Batch size: {TRAINING_CONFIG['batch']} (Conservative for stability)")



  print("\n" + "=" * 70)
  print("TRAINING MODEL 1 (Primary Model)")
  print("=" * 70)

  import gc
  torch.cuda.empty_cache()
  gc.collect()
  print("✓ GPU memory cleared before training")

  model_name = TRAINING_CONFIG.pop('model_name') 
  model1 = YOLO(model_name)

  results1 = model1.train(
      data='./rdd2022.yaml',
      **TRAINING_CONFIG
  )

  TRAINING_CONFIG['model_name'] = model_name

  print("\n✓ Model 1 training complete!")
  print(f"  Best weights: runs/detect/train/weights/best.pt")
  print(f"  Results: {results1}")

  best_model1_path = 'runs/detect/train/weights/best.pt'
  shutil.copy(best_model1_path, './model1_best.pt')
  print("✓ Model 1 saved as: model1_best.pt")


  print("\n" + "=" * 70)
  print("TRAINING MODEL 2 (Ensemble Model)")
  print("=" * 70)
  print("Training with different augmentation strategy for diversity...")

  import gc
  torch.cuda.empty_cache()
  gc.collect()
  print("✓ GPU memory cleared before Model 2 training")

  TRAINING_CONFIG_2 = TRAINING_CONFIG.copy()
  TRAINING_CONFIG_2['hsv_v'] = 0.6 
  TRAINING_CONFIG_2['scale'] = 0.7  
  TRAINING_CONFIG_2['mixup'] = 0.15 
  TRAINING_CONFIG_2['copy_paste'] = 0.15
  TRAINING_CONFIG_2['lr0'] = 0.0008  
  TRAINING_CONFIG_2['project'] = 'runs/detect'
  TRAINING_CONFIG_2['name'] = 'train2'

  model_name = TRAINING_CONFIG_2.pop('model_name')  
  model2 = YOLO(model_name)

  results2 = model2.train(
      data='./rdd2022.yaml',
      **TRAINING_CONFIG_2
  )

  print("\n✓ Model 2 training complete!")

  best_model2_path = 'runs/detect/train2/weights/best.pt'
  shutil.copy(best_model2_path, './model2_best.pt')
  print("✓ Model 2 saved as: model2_best.pt")

  print("\n" + "=" * 70)
  print("VALIDATION & THRESHOLD OPTIMIZATION")
  print("=" * 70)

  model1_best = YOLO('./model1_best.pt')
  model2_best = YOLO('./model2_best.pt')

  print("\nValidating Model 1...")
  metrics1 = model1_best.val(data='./rdd2022.yaml', imgsz=768, batch=10)
  print(f"Model 1 mAP@0.5: {metrics1.box.map50:.4f}")
  print(f"Model 1 mAP@0.5:0.95: {metrics1.box.map:.4f}")

  print("\nValidating Model 2...")
  metrics2 = model2_best.val(data='./rdd2022.yaml', imgsz=768, batch=10)
  print(f"Model 2 mAP@0.5: {metrics2.box.map50:.4f}")
  print(f"Model 2 mAP@0.5:0.95: {metrics2.box.map:.4f}")

  print("\nPer-class mAP (Model 1):")
  class_names = ['Longitudinal', 'Transverse', 'Alligator', 'Other', 'Pothole']
  for i, (name, map_val) in enumerate(zip(class_names, metrics1.box.maps)):
      print(f"  {name}: {map_val:.4f}")


  print("\n" + "=" * 70)
  print("GENERATING TEST PREDICTIONS (TTA + Ensemble)")
  print("=" * 70)

  from ensemble_boxes import weighted_boxes_fusion
  import cv2

  predictions_dir = Path('./predictions')

  os.makedirs(predictions_dir, exist_ok=True)

  test_images_dir = r'C:\Users\woebe\College\shalini_hackathon\randomized_dataset\test\images'
  test_images = sorted(list(Path(test_images_dir).glob('*.jpg')) + 
                      list(Path(test_images_dir).glob('*.png')))

  print(f"Found {len(test_images)} test images")

  CONF_THRESHOLD = 0.25 
  IOU_THRESHOLD = 0.45  

  CLASS_CONF_THRESHOLDS = {
      0: 0.30, 
      1: 0.25, 
      2: 0.20,  
      3: 0.25,  
      4: 0.20, 
  }

  def predict_with_tta_and_ensemble(image_path, model1, model2):
      """Predict with Test-Time Augmentation and Model Ensemble"""
      
      # Read image
      img = cv2.imread(str(image_path))
      h, w = img.shape[:2]
      
      all_boxes = []
      all_scores = []
      all_labels = []
      
      # TTA configurations: [augment_flag, flip]
      tta_configs = [
          (False, False),  # Original
          (True, False),   # Augmented
          (False, True),   # Horizontal flip
      ]
      
      for model_idx, model in enumerate([model1, model2]):
          for aug, flip in tta_configs:
              # Prepare image
              img_pred = img.copy()
              if flip:
                  img_pred = cv2.flip(img_pred, 1)
              
              # Predict
              results = model.predict(
                  img_pred,
                  imgsz=768,  # Match training size
                  conf=CONF_THRESHOLD,
                  iou=IOU_THRESHOLD,
                  augment=aug,
                  verbose=False
              )[0]
              
              # Extract predictions
              if len(results.boxes) > 0:
                  boxes = results.boxes.xyxyn.cpu().numpy()  # Normalized coordinates
                  scores = results.boxes.conf.cpu().numpy()
                  labels = results.boxes.cls.cpu().numpy().astype(int)
                  
                  # Flip boxes back if we flipped the image
                  if flip:
                      boxes[:, [0, 2]] = 1 - boxes[:, [2, 0]]
                  
                  all_boxes.append(boxes)
                  all_scores.append(scores)
                  all_labels.append(labels)
      
      # Weighted Box Fusion (WBF)
      if len(all_boxes) > 0:
          boxes_list = [boxes for boxes in all_boxes]
          scores_list = [scores for scores in all_scores]
          labels_list = [labels for labels in all_labels]
          
          # Weights: give more weight to model1 (assuming it's better)
          weights = [0.6, 0.4] * len(tta_configs) 
          
          fused_boxes, fused_scores, fused_labels = weighted_boxes_fusion(
              boxes_list,
              scores_list,
              labels_list,
              weights=weights,
              iou_thr=0.5,
              skip_box_thr=0.01
          )
          
          return fused_boxes, fused_scores, fused_labels
      
      return np.array([]), np.array([]), np.array([])

  # Generate predictions
  print("\nGenerating predictions with TTA + WBF ensemble...")
  print(f"Processing {len(test_images)} images with 6x predictions each (2 models × 3 TTA variants)")
  print("Estimated time: ~15 minutes on P100")

  import time
  start_time = time.time()

  for idx, img_path in enumerate(test_images):
      if idx % 50 == 0:
          elapsed = (time.time() - start_time) / 60
          if idx > 0:
              rate = idx / elapsed
              remaining = (len(test_images) - idx) / rate
              print(f"  Progress: {idx}/{len(test_images)} | {elapsed:.1f} min elapsed | ~{remaining:.1f} min remaining")
      
      # Predict
      boxes, scores, labels = predict_with_tta_and_ensemble(
          img_path, model1_best, model2_best
      )
      
      output_lines = []
      for box, score, label in zip(boxes, scores, labels):
          # Apply class-specific confidence threshold
          if score < CLASS_CONF_THRESHOLDS[int(label)]:
              continue
          
          x1, y1, x2, y2 = box
          x_center = (x1 + x2) / 2
          y_center = (y1 + y2) / 2
          width = x2 - x1
          height = y2 - y1
          
          output_lines.append(
              f"{int(label)} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f} {score:.6f}"
          )
      
      # Save prediction file
      pred_file = predictions_dir / f"{img_path.stem}.txt"
      with open(pred_file, 'w') as f:
          f.write('\n'.join(output_lines))

  total_time = (time.time() - start_time) / 60
  print(f"\n✓ Predictions completed in {total_time:.1f} minutes")
  print(f"✓ Predictions saved to: {predictions_dir}")


  print("\n" + "=" * 70)
  print("CREATING SUBMISSION FILE")
  print("=" * 70)

  shutil.make_archive('./submission', 'zip', predictions_dir)
  print("✓ Submission created: submission.zip")

  print("\n" + "=" * 70)
  print("PERFORMANCE SUMMARY")
  print("=" * 70)



if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()

    main()
