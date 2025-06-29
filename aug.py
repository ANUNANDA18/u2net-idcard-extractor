import os
import cv2
import albumentations as A
from PIL import Image
from tqdm import tqdm

# Paths
input_img_dir = r'C:\Users\Anunanda\id_card_project\id_\dataset\train\images'
input_mask_dir = r'C:\Users\Anunanda\id_card_project\id_\dataset\train\masks'
output_img_dir = r'C:\Users\Anunanda\id_card_project\id_\augmented\images'
output_mask_dir = r'C:\Users\Anunanda\id_card_project\id_\augmented\mask'

# Create output folders
os.makedirs(output_img_dir, exist_ok=True)
os.makedirs(output_mask_dir, exist_ok=True)

# Augmentation pipeline
transform = A.Compose([
    A.Resize(320, 320),  # optional
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.5),
    A.Rotate(limit=15, p=0.5),
    A.GaussNoise(p=0.2),
    A.RandomScale(scale_limit=0.2, p=0.3),
], additional_targets={'mask': 'mask'})

# Number of augmented versions per image
N = 10

# Loop through each image-mask pair
for filename in tqdm(os.listdir(input_img_dir)):
    if not filename.lower().endswith(('.jpg', '.jpeg', '.png')):
        continue

    img_path = os.path.join(input_img_dir, filename)
    name_wo_ext = os.path.splitext(filename)[0]
    mask_path = os.path.join(input_mask_dir, name_wo_ext + '_mask.png')

    image = cv2.imread(img_path)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    if image is None or mask is None:
        print(f" Skipping {filename}: image or mask not found.")
        continue

    # Save original
    cv2.imwrite(os.path.join(output_img_dir, filename), image)
    cv2.imwrite(os.path.join(output_mask_dir, name_wo_ext + '.png'), mask)

    # Do augmentation...

    # Generate N augmented images
    for i in range(N):
        augmented = transform(image=image, mask=mask)
        aug_image = augmented['image']
        aug_mask = augmented['mask']

        name = filename.split('.')[0]
        aug_img_name = f"{name}_aug{i}.jpg"
        aug_mask_name = f"{name}_aug{i}.png"

        cv2.imwrite(os.path.join(output_img_dir, aug_img_name), aug_image)
        cv2.imwrite(os.path.join(output_mask_dir, aug_mask_name), aug_mask)
