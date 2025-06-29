from torch.utils.data import Dataset
from PIL import Image
import os
import numpy as np
import torchvision.transforms as transforms

class IDCardDataset(Dataset):
    def __init__(self, image_dir, mask_dir):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.image_list = sorted(os.listdir(image_dir))
        self.mask_list = sorted(os.listdir(mask_dir))
        self.transform = transforms.Compose([
            transforms.Resize((320, 320)),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        image = Image.open(os.path.join(self.image_dir, self.image_list[idx])).convert("RGB")
        mask = Image.open(os.path.join(self.mask_dir, self.mask_list[idx])).convert("L")
        return self.transform(image), self.transform(mask)
'''from torch.utils.data import Dataset
from PIL import Image
import os
import numpy as np
import torchvision.transforms as transforms

class IDCardDataset_Torchvision(Dataset):
    def __init__(self, image_dir, mask_dir, target_size=(320, 320)):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.image_list = sorted(os.listdir(image_dir))
        self.mask_list = sorted(os.listdir(mask_dir))
        self.target_size = target_size

        # Image transformations
        self.image_transform = transforms.Compose([
            transforms.Resize(self.target_size, interpolation=Image.BILINEAR), # Good for images
            transforms.ToTensor(), # Converts to tensor and normalizes to [0,1]
            # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # Mask transformations
        self.mask_transform = transforms.Compose([
            transforms.Resize(self.target_size, interpolation=Image.NEAREST), # CRITICAL for masks
            transforms.ToTensor() # Converts to tensor. For masks, it will be 0.0 or 1.0 (float)
        ])

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        image_path = os.path.join(self.image_dir, self.image_list[idx])
        mask_path = os.path.join(self.mask_dir, self.mask_list[idx])

        image = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        # Apply separate transforms
        transformed_image = self.image_transform(image)
        transformed_mask = self.mask_transform(mask)

        # Convert mask tensor to long type (essential for segmentation loss functions)
        transformed_mask = transformed_mask.squeeze(0).long() # ToTensor adds a channel dim, remove it for mask (C,H,W) -> (H,W)

        return transformed_image, transformed_mask

# --- How to use ---
if __name__ == '__main__':
    # (Use the same dummy_images and dummy_masks setup as above)
    os.makedirs('dummy_images', exist_ok=True)
    os.makedirs('dummy_masks', exist_ok=True)

    for i in range(5):
        img_h, img_w = np.random.randint(200, 600, size=2)
        mask_h, mask_w = np.random.randint(200, 600, size=2)

        dummy_img = Image.fromarray(np.random.randint(0, 255, (img_h, img_w, 3), dtype=np.uint8))
        dummy_mask = Image.fromarray(np.random.randint(0, 1, (mask_h, mask_w), dtype=np.uint8) * 255)

        dummy_img.save(os.path.join('dummy_images', f'image_{i}.png'))
        dummy_mask.save(os.path.join('dummy_masks', f'mask_{i}.png'))


    dataset = IDCardDataset_Torchvision(image_dir='dummy_images', mask_dir='dummy_masks', target_size=(320, 320))

    print(f"Dataset size: {len(dataset)}")
    img_tensor, mask_tensor = dataset[0]
    print(f"Image tensor shape: {img_tensor.shape}") # Should be torch.Size([3, 320, 320])
    print(f"Mask tensor shape: {mask_tensor.shape}") # Should be torch.Size([320, 320])
    print(f"Mask tensor dtype: {mask_tensor.dtype}") # Should be torch.int64

    # Clean up dummy directories
    import shutil
    shutil.rmtree('dummy_images')
    shutil.rmtree('dummy_masks')'''