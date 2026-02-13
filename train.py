import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from utils.dataset2d_sequence import LiverSequenceDataset
from models.cunet_clstm import CUNet_CLSTM

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Using device:", device)

# -------------------
# Load Dataset
# -------------------
dataset = LiverSequenceDataset("data", seq_len=16)
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

print("Dataset size:", len(dataset))

# -------------------
# Load Model
# -------------------
model = CUNet_CLSTM().to(device)

# Loss functions
seg_loss_fn = nn.BCELoss()
cls_loss_fn = nn.BCELoss()

optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

epochs = 10

# -------------------
# Training Loop
# -------------------
for epoch in range(epochs):
    model.train()
    total_loss = 0

    for volume_seq, mask_seq, label in dataloader:
        volume_seq = volume_seq.to(device)
        mask_seq = mask_seq.to(device)
        label = label.to(device).unsqueeze(1)

        optimizer.zero_grad()

        seg_out, cls_out = model(volume_seq)

        seg_loss = seg_loss_fn(seg_out, mask_seq)
        cls_loss = cls_loss_fn(cls_out, label)

        loss = seg_loss + 2.0 * cls_loss


        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch [{epoch+1}/{epochs}] Loss: {total_loss:.4f}")

# Save model
torch.save(model.state_dict(), "cunet_clstm_model.pth")

print("Training Complete. Model Saved!")
