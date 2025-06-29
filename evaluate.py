import os
import cv2
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, jaccard_score
from tqdm import tqdm

# Directories
gt_dir = r'C:\Users\Anunanda\id_card_project\id_\eval\ground_truth_masks'       # PNG files
pred_dir = r'C:\Users\Anunanda\id_card_project\id_\eval\predicted_masks'         # JPG files

# Get base filenames from ground truth folder (without extension)
gt_files = [f for f in os.listdir(gt_dir) if f.endswith('.png')]
gt_basenames = [os.path.splitext(f)[0] for f in gt_files]

# Initialize metric lists
precisions, recalls, f1s, ious = [], [], [], []

for base in tqdm(gt_basenames, desc="Evaluating"):
    gt_path = os.path.join(gt_dir, base + ".png")
    pred_path = os.path.join(pred_dir, base + ".jpg")  # Adjust to your actual predicted extension

    if not os.path.exists(gt_path) or not os.path.exists(pred_path):
        print(f"Skipping {base} (missing file)")
        continue

    # Load masks
    gt = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
    pred = cv2.imread(pred_path, cv2.IMREAD_GRAYSCALE)

    if gt is None or pred is None:
        print(f"Skipping {base} (unreadable file)")
        continue

    # Resize predicted to match ground truth if needed
    if pred.shape != gt.shape:
        pred = cv2.resize(pred, (gt.shape[1], gt.shape[0]))

    # Binarize masks
    _, gt_bin = cv2.threshold(gt, 127, 1, cv2.THRESH_BINARY)
    _, pred_bin = cv2.threshold(pred, 127, 1, cv2.THRESH_BINARY)

    # Flatten for metric calculation
    gt_flat = gt_bin.flatten()
    pred_flat = pred_bin.flatten()

    # Metrics
    precisions.append(precision_score(gt_flat, pred_flat, zero_division=0))
    recalls.append(recall_score(gt_flat, pred_flat, zero_division=0))
    f1s.append(f1_score(gt_flat, pred_flat, zero_division=0))
    ious.append(jaccard_score(gt_flat, pred_flat, zero_division=0))

# --- Final Results ---
print("\n Evaluation Results:")
print(f"Precision: {np.mean(precisions):.4f}")
print(f"Recall:    {np.mean(recalls):.4f}")
print(f"F1-Score:  {np.mean(f1s):.4f}")
print(f"IoU:       {np.mean(ious):.4f}")
