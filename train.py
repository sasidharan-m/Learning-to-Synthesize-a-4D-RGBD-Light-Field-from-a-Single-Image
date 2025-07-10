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
        "Python script for training the CNN framework to synthesize Light Field data from a single RGB image\n"
        "The script needs the path to the training data and the path for saving the weights\n"
        "All other parameters are optional.\n\n"
        
        "Example:\n"
        "    python train.py --training_data_path <path to directory containing the light data for training> --weights_save_path <path to directory where the weights have to be saved>\n"
        ),
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument('--training_data_path', type=str, required=True,
                        help='Path to the training data.')
    parser.add_argument('--weights_save_path', type=str, required=True,
                        help='Path where model weights should be saved.')
    parser.add_argument('--checkpoint_path', type=str, required=False, default="",
                        help='Path where model weights read from to resume training.')
    parser.add_argument('--grid', type=tuple, required=False, default=(13,13),
                        help='Tuple (U,V) that define the angular dimensions of the Light Field data.')
    parser.add_argument('--crop_size', type=tuple, required=False, default=(256,256),
                        help='Tuple (H,W) setting for the size of the random crop used for training.')   
    parser.add_argument('--batch_size', type=int, required=False, default=2,
                        help='Batch size used for training.')
    parser.add_argument('--epochs', type=int, required=False, default=1000,
                        help='Number of epochs to train for.')
    parser.add_argument('--learning_rate', type=float, required=False, default=1e-4,
                        help='Learning rate used for training.')
    parser.add_argument('--disp_mult', type=float, required=False, default=(4.0),
                        help='Float disp_mult that defines the disparity range (-disp_mult, disp_mult).')
    parser.add_argument('--lam_tv', type=float, required=False, default=0.01,
                        help='Weight for the TV loss used for training.')
    parser.add_argument('--lam_dc', type=float, required=False, default=0.005,
                        help='Weight for the DC loss used for training.')
    parser.add_argument('--lfsize', type=tuple, required=False, default=(432,622,9,9),
                        help='Tuple (H,W,U,V) that defines the deimensions of the captured Light Field data.')
    return parser.parse_args()




def train(training_data_path, weights_save_path, checkpoint_path="", grid=(13,13), 
          crop_size=(256,256), batch_size=2, epochs=1000, learning_rate=1e-4, disp_mult=4.0,
          lam_tv=0.01, lam_dc=0.005, lfsize=(432,622,9,9)):
    """
    Function that does the CNN framework training

    Arguments:
    ----------
    training_data_path  - String that holds the path to the directory containing the sub-aperture views
    weights_save_path   - String that holds the path to the directory where the weights should be saved
    checkpoint_path     - String that holds the path to the stored weights if training has to be resumed
    grid                - Tuple that defines the U,V angular dimensions of the light field
    crop_size           - Tuple that defines the size of the random crop used for training
    batch_size          - Integer values that defines the batch size of the training pipeline
    epochs              - Integer that defines the number of epochs
    learning_rate       - Floating point value that defines the learning rate used for training
    disp_mult           - Floating point value that defines the range of valid disparities
    lam_tv              - Floating point value that defines the Total Variational Loss weightage used for training
    lam_dc              - Floating value that defines the Consistency Regularization Loss weightage used for training
    lfsize              - Tuple that defines the dimensions of the 4D Light Field

    """

    loader = getDataloader(
        training_data_path, grid, spatial_crop=crop_size,
        batch_size=batch_size,
        resize=None,               
        num_workers=4
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
    print("Starting training...")
    train(args.training_data_path, args.weights_save_path, checkpoint_path=args.checkpoint_path, 
          grid=args.grid, crop_size=args.crop_size, batch_size=args.batch_size,
          epochs=args.epochs, learning_rate=args.learning_rate, disp_mult=args.disp_mult,
          lam_tv=args.lam_tv, lam_dc=args.lam_dc, lfsize=args.lfsize)
    print('Training Done.')

# Run the driver function
if __name__ == "__main__":
    main()
