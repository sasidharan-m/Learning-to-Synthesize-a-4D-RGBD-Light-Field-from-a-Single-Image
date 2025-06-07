# Python script that runs the trained model
# Author: Sasidhran Mahalingam
# Date Created: May 7 2025

# Import the required packages
import os
import torch
from models.model import ForwardModel
from PIL import Image
import torchvision.transforms.functional as TF
from torchvision.utils import save_image
import numpy as np
import matplotlib.pyplot as plt
import argparse

def parse_args():
    """
    Function that parses the input arguments

    Parameters:
    -----------
    None

    Returns:
    --------
    Returns the parsed argmuent values
    """
    parser = argparse.ArgumentParser(
        description=(
        "Python script for tesing the CNN framework to synthesize Light Field data from a single RGB image\n"
        "The script needs the path to the pre-trained weights, path to the input image and the path to the output directory for saving the synthesized light fields\n"
        "All other parameters are optional.\n\n"
        
        "Example:\n"
        "    python test.py --pretrained_weights <path to the pre-trained weights> --input_image <path to the input iamge> --output_path <path to the output directory>\n"
        ),
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument('--pretrained_weights', type=str, required=True,
                        help='Path to the pre-trained weights.')
    parser.add_argument('--input_image', type=str, required=True,
                        help='Path to the input image for which the light field has to be synthesized.')
    parser.add_argument('--output_path', type=str, required=True,
                        help='Path to the save the synthesized light fields.')
    parser.add_argument('--save_ray_depths', type=bool,required=False, default=False,
                        help='Flag to enable saving of the ray depths.')
    parser.add_argument('--save_lambertian', type=bool,required=False, default=False,
                        help='Flag to enable saving of the lambertian light fields.')
    parser.add_argument('--lfsize', type=tuple,required=False, default=(432, 620, 9, 9),
                        help='Tuple that defines the lize of the generated light field.')
    parser.add_argument('--disp_mult', type=float, required=False, default=(4.0),
                        help='Float disp_mult that defines the disparity range (-disp_mult, disp_mult).')
    return parser.parse_args()


def test(weights, input_image, output_dir, save_ray_depths=False, save_lambertian=False,
         lfsize=(432, 620, 9, 9), disp_mult=4.0):
    """
    Function that does the inference of the model to generate the light field data
    
    Parameters:
    -----------
    weights         - String that holds the path to the pre-trained weights
    input_image     - String that holds the path to the input image
    output_dir      - String that holds the path to save the generate light fields
    save_ray_depths - Flag to enable/diable the saving of the ray depths
    save_lambertian - Flag to enable/disable the saving of the lambertian light fields
    lfsize          - Tuple that defines the dimensions of the light field
    disp_mult       - Floating point value that defines the range of valid disparities

    Returns:
    --------
    Nothing
    """
    os.makedirs(output_dir, exist_ok=True)

    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    model = ForwardModel(lfsize=lfsize, disp_mult=disp_mult).to(device)
    checkpoint = torch.load(weights, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    # Prepare a batch of center views (aif_batch), e.g. from a DataLoader or single image
    img = Image.open(input_image).convert("RGB")
    img_t = TF.to_tensor(img).float().unsqueeze(0)    # [1,3,H,W], range [0,1]
    # if model expects [-1,1] range:
    img_t = img_t * 2.0 - 1.0                
    aif_batch = img_t.to(device)

    # Infer
    with torch.no_grad():
        ray_depths, lf_shear, y = model(aif_batch)
        # lf_shear: [1, H, W, V, U, 3], in [-1,1] if model output normalized

    # Save each sub-aperture view
    _, H, W, V, U, C = y.shape
    # Bring to [V*U, C, H, W] and [0,1] range
    views = y.squeeze(0)        # [H,W,V,U,3]
    views = views.permute(2, 3, 4, 0, 1)  # [V,U,3,H,W]
    views = views.reshape(V*U, C, H, W)

    # images are in [-1,1], map back to [0,1]
    views = (views + 1.0) * 0.5
    views = views.clamp(0,1)

    # save grid or individual files
    for idx in range(V*U):
        v = idx // U
        u = idx % U
        out_path = os.path.join(output_dir, 'sub-aperture_images')
        os.makedirs(out_path, exist_ok=True)
        out_path = os.path.join(output_dir, 'sub-aperture_images',f"view_{v}_{u}.png")
        save_image(views[idx], out_path)

    if save_lambertian:
        # Save the lambertian light fields
        lambertian_views = lf_shear.squeeze(0)        # [H,W,V,U,3]
        lambertian_views = lambertian_views.permute(2, 3, 4, 0, 1)  # [V,U,3,H,W]
        lambertian_views = lambertian_views.reshape(V*U, C, H, W)

        # images are in [-1,1], map back to [0,1]
        lambertian_views = (views + 1.0) * 0.5
        lambertian_views = views.clamp(0,1)

        # save grid or individual files
        for idx in range(V*U):
            v = idx // U
            u = idx % U
            out_path = os.path.join(output_dir, 'lambertian_sub-aperture_images')
            os.makedirs(out_path, exist_ok=True)
            out_path = os.path.join(output_dir, 'lambertian_sub-aperture_images',f"view_{v}_{u}.png")
            save_image(lambertian_views[idx], out_path)

    if save_ray_depths:
        # Get the disaprities, convert them into depths and normalize them
        depths = ray_depths.squeeze(0)
        depths = depths.permute(2, 3, 0, 1)
        depths = depths.reshape(V*U, H, W)
        depths = depths.cpu().detach().numpy()
        disp_min, disp_max = depths.min(), depths.max()
        # Shift the results to avoid negative values
        depths = depths - disp_min
        # Avoiding divide by zero
        depths[depths == 0] = 1e-6
        disp_min, disp_max = depths.min(), depths.max()
        # Calculate the depths from the disparity
        depths = (3000 * 0.05) / depths
        # Clip the values from 18.75 to 50. Might have to change the values for better visuzlization
        vmin = 18.75
        vmax = 50
        depths = np.clip(depths, vmin, vmax)
        # Taking log of the values for better visualization
        depth_log = np.log10(depths)
        depth_log_norm = (depth_log - depth_log.min()) / (depth_log.max() - depth_log.min())
        # save estimated depths
        for idx in range(V*U):
            v = idx // U
            u = idx % U
            out_path = os.path.join(output_dir, 'estimated_disparity')
            os.makedirs(out_path, exist_ok=True)
            out_path = os.path.join(output_dir, 'estimated_disparity', f"view_{v}_{u}.png")
            colormap = plt.get_cmap('plasma')
            colored_disp = colormap(depth_log_norm[idx])
            colored_disp = (colored_disp[:, :, :3] * 255).astype(np.uint8)
            depth_image = Image.fromarray(colored_disp)
            depth_image.save(out_path)

def main():
    """
    Driver function for training the model

    Parameters:
    -----------
    None

    Returns:
    --------
    Nothing
    """
    args = parse_args()
    print('Starting Inference...')
    test(args.pretrained_weights, args.input_image, args.output_path, 
         save_ray_depths=args.save_ray_depths, save_lambertian=args.save_lambertian,
         lfsize=args.lfsize, disp_mult=args.disp_mult)
    print('Inference done. Synthesized Light Field data saved.')

# Run the driver function
if __name__ == "__main__":
    main()
