# Python script that run the training model
# Author: Sasidhran Mahalingam
# Date Created: May 7 2025

# Import the required packages
import os
import torch
from models.model import ForwardModel
from PIL import Image
import torchvision.transforms.functional as TF
from torchvision.utils import save_image

CHECKPOINT = "/home/sasidharan/Projects/Plenoptic Camera/Notebooks/forward_model_final.pth"
OUTPUT_DIR = "/home/sasidharan/Projects/Plenoptic Camera/Notebooks/Output Viewpoints"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Configuration
lfsize    = (432, 620, 9, 9)        # replace V, U with your angular dims
disp_mult = 4.0                  # same as training
device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1) Load model
model = ForwardModel(lfsize=lfsize, disp_mult=disp_mult).to(device)
checkpoint = torch.load(CHECKPOINT, map_location=device)
model.load_state_dict(checkpoint)
model.eval()

# 2) Prepare a batch of center views (aif_batch), e.g. from a DataLoader or single image
img = Image.open("car_1.png").convert("RGB")
img_t = TF.to_tensor(img).float().unsqueeze(0)    # [1,3,H,W], range [0,1]
# if model expects [-1,1] range:
img_t = img_t * 2.0 - 1.0                
aif_batch = img_t.to(device)

# 3) Infer
with torch.no_grad():
    ray_depths, lf_shear, y = model(aif_batch)
    # lf_shear: [1, H, W, V, U, 3], in [-1,1] if model output normalized

# 4) Save each sub-aperture view
_, H, W, V, U, C = y.shape
# bring to [V*U, C, H, W] and [0,1] range
views = y.squeeze(0)        # [H,W,V,U,3]
views = views.permute(2, 3, 4, 0, 1)  # [V,U,3,H,W]
views = views.reshape(V*U, C, H, W)

# if lf_shear is in [-1,1], map back to [0,1]
views = (views + 1.0) * 0.5
views = views.clamp(0,1)

# save grid or individual files
for idx in range(V*U):
    v = idx // U
    u = idx % U
    out_path = os.path.join(OUTPUT_DIR, f"view_{v}_{u}.png")
    save_image(views[idx], out_path)

print("Saved", V*U, "views to", OUTPUT_DIR)