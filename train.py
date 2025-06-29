import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from .dataset import IDCardDataset
import sys
import os
from .u2net.model import U2NET

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load dataset
train_dataset = IDCardDataset(r"C:\Users\Anunanda\id_card_project\id_\augmented\images", r"C:\Users\Anunanda\id_card_project\id_\augmented\mask")
train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)

# Load model
model = U2NET(3, 1)
model.to(device)

# Optimizer & Loss
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.BCELoss()

# Training loop
num_epochs = 25
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for images, masks in train_loader:
        images, masks = images.to(device), masks.to(device)
        outputs = model(images)[0]
        loss = criterion(outputs, masks)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss:.4f}")

# Save model
torch.save(model.state_dict(), r"C:\Users\Anunanda\id_card_project\id_\models\u2net.pth")
print("Training complete. Model saved.")
