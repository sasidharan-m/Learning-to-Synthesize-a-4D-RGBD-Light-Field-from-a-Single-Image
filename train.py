# Python script that starts the training of the defined neural network pipeline that generates the lightfield given the central view
# Author: Sasidharan Mahalingam
# Date Created: May 7 2025

# Import the required packages
import torch
from data.dataloader import getDataloader
from models.model import ForwardModel
from losses.losses import tvLoss, depthConsistencyLoss
import torch.optim as optim
import os
from tqdm import tqdm

training_data_path              = "/home/sasidharan/Projects/Plenoptic Camera/Datasets/Flower Dataset/Sub-Aperture Images/Train"
weights_save_path               = "/home/sasidharan/Projects/Plenoptic Camera/Code/Learning-to-Synthesize-a-4D-RGBD-Light-Field-from-a-Single-Image/weights"
checkpoint_path                 = "/home/sasidharan/Projects/Plenoptic Camera/Code/Learning-to-Synthesize-a-4D-RGBD-Light-Field-from-a-Single-Image/weights/checkpoint_epoch100_restore.pth"
grid                            = (13, 13)                # U×V views
crop_size                       = (256, 256)              # spatial-crop height & width
batch_size                      = 2
epochs                          = 1000
learning_rate                   = 1e-4
disp_mult                       = 4.0
lam_tv                          = 0.01
lam_dc                          = 0.005
lfsize                          = (432,622,9,9)

loader = getDataloader(
    training_data_path, grid, spatial_crop=crop_size,
    batch_size=batch_size,
    resize=None,               
    num_workers=2
)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = ForwardModel(lfsize=lfsize, disp_mult=disp_mult).to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

checkpoint = None

if checkpoint_path != "" and os.path.exists(checkpoint_path):
    print("Loading model state from checkpoint...")
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint["model_state"])
    optimizer.load_state_dict(checkpoint["optim_state"])

for epoch in range(epochs):
    model.train()
    total_loss = 0.0
    loop = tqdm(enumerate(loader), total=len(loader), ncols=80, desc=f"Epoch [{epoch+1}/{epochs}]", leave=True, dynamic_ncols=False)
    for i, (aif_batch, lf_batch) in loop:
        # move to device
        aif_batch = aif_batch.to(device)          # [B,3,h,w]
        lf_batch  = lf_batch.to(device)           # [B,h,w,V,U,3]

        # forward pass
        ray_depths, lf_shear, y = model(aif_batch)
        # ray_depths: [B,h,w,V,U]
        # lf_shear:   [B,h,w,V,U,3]
        # y:          [B,h,w,V,U,3]

        # 1) shear reconstruction loss
        shear_loss = torch.mean(torch.abs(lf_shear - lf_batch))

        # 2) occlusion‐corrected output loss
        output_loss = torch.mean(torch.abs(y - lf_batch))

        # 3) TV on depth
        tv = tvLoss(ray_depths, lfsize)

        # 4) depth consistency
        dc = depthConsistencyLoss(ray_depths, lfsize)

        # total
        loss = shear_loss + output_loss + lam_tv * tv + lam_dc * dc

        # backward + step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * aif_batch.size(0)

    avg_loss = total_loss / len(loader.dataset)
    
    if(((epoch + 1) % 1 == 0) or epoch == 0):
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")

    if(((epoch + 1) % 100 == 0) and epoch > 1):
        ckpt = {
            "epoch": epoch + 1,
            "model_state": model.state_dict(),
            "optim_state": optimizer.state_dict(),
            "loss": avg_loss,
        }
        weights_name = f"checkpoint_epoch{epoch+1}.pth"
        torch.save(ckpt, os.path.join(weights_save_path, weights_name))
torch.save(model.state_dict(), os.path.join(weights_save_path, "forward_model_final.pth"))