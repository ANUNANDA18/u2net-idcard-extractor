import torch
import os
from PIL import Image
import torchvision.transforms as transforms
import numpy as np
from id_.scripts.u2net.model.u2net import U2NET


# --- Paths ---
model_path = r"C:\Users\Anunanda\id_card_project\id_\models\u2net.pth"
test_image_dir = r"C:\Users\Anunanda\id_card_project\id_\test\images"
output_mask_dir = r"C:\Users\Anunanda\id_card_project\id_\eval\predicted_masks"
output_extract_dir = r"C:\Users\Anunanda\id_card_project\id_\outputs\extracted"

os.makedirs(output_mask_dir, exist_ok=True)
os.makedirs(output_extract_dir, exist_ok=True)

# --- Load model ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = U2NET(3, 1)
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()

# --- Preprocess ---
transform = transforms.Compose([
    transforms.Resize((320, 320)),
    transforms.ToTensor()
])

# --- Mask post-process ---
def post_process_mask(pred):
    pred = pred.squeeze().cpu().numpy()
    pred = (pred * 255).astype(np.uint8)
    pred[pred > 100] = 255
    pred[pred <= 100] = 0
    return Image.fromarray(pred)

# --- Transparent extraction ---
def apply_mask(original, mask):
    mask = mask.resize(original.size).convert("L")
    original = original.convert("RGBA")
    result = Image.new("RGBA", original.size)
    result.paste(original, (0, 0), mask)
    return result


# --- Inference loop ---
for filename in os.listdir(test_image_dir):
    if filename.lower().endswith((".jpg", ".jpeg", ".png")):
        print(f" Processing: {filename}")
        img_path = os.path.join(test_image_dir, filename)
        image = Image.open(img_path).convert("RGB")
        input_tensor = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            pred = model(input_tensor)[0]

        # Save mask
        mask = post_process_mask(pred)
        mask.save(os.path.join(output_mask_dir, f"mask_{filename}"))

        # Save extracted ID card
        extracted = apply_mask(image, mask)
        extracted.save(os.path.join(output_extract_dir, f"extracted_{filename}.png"))

print("Done! Check outputs/masks and outputs/extracted.")

