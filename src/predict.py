import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib.patches import Rectangle
import time

######################################
# Configuration
######################################
MODEL_WEIGHTS_PATH = "../results/20250314/postech/model_weights_postech.pth"
FEATURES = 32
NUM_GROUPS = 16

IMAGE_ID = "2024-12-31_188" # Try also "2024-12-31_191"
FOLDER_ID = IMAGE_ID.split("_")[0].replace("-","")
ATOM_IMAGE_PATH   = os.path.join(f"../{FOLDER_ID}_full/npy_{FOLDER_ID}/atom_npy/", f"{IMAGE_ID}_atom.npy")
BRIGHT_IMAGE_PATH = os.path.join(f"../{FOLDER_ID}_full/npy_{FOLDER_ID}/bright_npy/", f"{IMAGE_ID}_bright.npy")
DARK_IMAGE_PATH   = os.path.join(f"../{FOLDER_ID}_full/npy_{FOLDER_ID}/dark_npy/", f"{IMAGE_ID}_dark.npy")
OD_IMAGE_PATH     = os.path.join(f"../{FOLDER_ID}_full/npy_{FOLDER_ID}/OD_npy/", f"{IMAGE_ID}.npy")

SAVE_FIG_PATH      = "../results_predict/prediction_result.png"
SAVE_PRED_OD_PATH  = "../results_predict/predicted_OD.npy"

# --- Disk Mask + Crop Parameters ---
CENTER_X = 600  
CENTER_Y = 400  
RADIUS = 384

# Define ROI for calcoeff in predicted image
ROI_PRED = [115, 600, 165, 650]

# Define ROI for calcoeff in original image 
ROI_ORIG = [CENTER_X-RADIUS+ROI_PRED[0], CENTER_Y-RADIUS+ROI_PRED[1], CENTER_X-RADIUS+ROI_PRED[2], CENTER_Y-RADIUS+ROI_PRED[3]]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

######################################
# U-Net Architecture
######################################

class DownBlock(nn.Module):
    """
    Encoder
    """
    def __init__(self, in_ch, out_ch, num_groups):
        super(DownBlock, self).__init__()
        layers = [
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(num_groups=num_groups, num_channels=out_ch),
            nn.ReLU(True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(num_groups=num_groups, num_channels=out_ch),
            nn.ReLU(True),
        ]
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        return self.conv(x)

class UpBlock(nn.Module):
    """
    Decoder
    """
    def __init__(self, in_ch, out_ch, num_groups):
        super(UpBlock, self).__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2, padding=0)

        self.conv = nn.Sequential(
            nn.Conv2d(out_ch*2, out_ch, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(num_groups=num_groups, num_channels=out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(num_groups=num_groups, num_channels=out_ch),
            nn.ReLU(inplace=True),
        )
        
    def forward(self, x, skip):
        x = self.up(x)
        x = torch.cat((skip, x), dim=1)
        x = self.conv(x)
        return x

class UNet(nn.Module):
    def __init__(self, in_ch=1, out_ch=1, features=32, num_groups=16):
        super(UNet, self).__init__()

        # ------------------ Encoder (7 downsamples) ------------------
        self.down1 = DownBlock(in_ch, features, num_groups)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.down2 = DownBlock(features, features*2, num_groups)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.down3 = DownBlock(features*2, features*4, num_groups)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.down4 = DownBlock(features*4, features*8, num_groups)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.down5 = DownBlock(features*8, features*16, num_groups)
        self.pool5 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.down6 = DownBlock(features*16, features*32, num_groups)
        self.pool6 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.down7 = DownBlock(features*32, features*64, num_groups)
        self.pool7 = nn.MaxPool2d(kernel_size=2, stride=2)

        # ------------------ Bottleneck ------------------
        self.bottleneck = DownBlock(features*64, features*128, num_groups)

        # ------------------ Decoder (7 upsamples) ------------------
        self.up7 = UpBlock(features*128, features*64, num_groups)
        self.up6 = UpBlock(features*64, features*32, num_groups)
        self.up5 = UpBlock(features*32, features*16, num_groups)
        self.up4 = UpBlock(features*16, features*8, num_groups)
        self.up3 = UpBlock(features*8,  features*4, num_groups)
        self.up2 = UpBlock(features*4,  features*2, num_groups)
        self.up1 = UpBlock(features*2,  features, num_groups)

        # ------------------ Final Output ------------------
        self.final = nn.Sequential(
            nn.Conv2d(features, out_ch, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # ------------------ Encoder Forward Pass ------------------
        d1 = self.down1(x)  # shape: (Batch, features, 1024, 1024)
        d1_pool = self.pool1(d1) # shape: (Batch, features, 512, 512)

        d2 = self.down2(d1_pool)  # shape: (Batch, features*2, 512, 512)
        d2_pool = self.pool2(d2) # shape: (Batch, features*2, 256, 256)

        d3 = self.down3(d2_pool)  # shape: (Batch, features*4, 256, 256)
        d3_pool = self.pool3(d3) # shape: (Batch, features*4, 128, 128)

        d4 = self.down4(d3_pool)  # shape: (Batch, features*8, 128, 128)
        d4_pool = self.pool4(d4) # shape: (Batch, features*8, 64, 64)

        d5 = self.down5(d4_pool)  # shape: (Batch, features*16, 64, 64)
        d5_pool = self.pool5(d5) # shape: (Batch, features*16, 32, 32)

        d6 = self.down6(d5_pool)  # shape: (Batch, features*32, 32, 32)
        d6_pool = self.pool6(d6) # shape: (Batch, features*32, 16, 16)

        d7 = self.down7(d6_pool)  # shape: (Batch, features*64, 16, 16)
        d7_pool = self.pool7(d7) # shape: (Batch, features*64, 8, 8)

        # ------------------ Bottleneck ------------------
        bottleneck = self.bottleneck(d7_pool) # shape: (Batch, features*128, 8, 8)        

        # ------------------ Decoder Forward Pass ------------------
        u7 = self.up7(bottleneck, d7) # shape: (Batch, features*64, 16, 16)
        u6 = self.up6(u7, d6)   # shape: (Batch, features*32, 32, 32)
        u5 = self.up5(u6, d5)   # shape: (Batch, features*16, 64, 64)
        u4 = self.up4(u5, d4)   # shape: (Batch, features*8, 128, 128)
        u3 = self.up3(u4, d3)   # shape: (Batch, features*4, 256, 256)
        u2 = self.up2(u3, d2)   # shape: (Batch, features*2, 512, 512)
        u1 = self.up1(u2, d1)   # shape: (Batch, features, 1024, 1024)

        out = self.final(u1)    
        return out

######################################
# Helper: Compute RMSE of an image compared to target (nn.mseloss)
######################################

def compute_rmse(image, target):
    return np.sqrt(np.mean(np.square(image-target)))

######################################
# Main prediction
######################################

def main():

    ######################################
    # Load model
    ######################################

    model = UNet(in_ch=1, out_ch=1, features=FEATURES, num_groups=NUM_GROUPS).to(DEVICE)
    if os.path.exists(MODEL_WEIGHTS_PATH):
        model.load_state_dict(torch.load(MODEL_WEIGHTS_PATH, map_location=DEVICE, weights_only=True))
        model.eval()
        print("Model weights loaded successfully.")
    else:
        raise FileNotFoundError(f"No model weights found at {MODEL_WEIGHTS_PATH}.")

    start_time = time.time()    

    ######################################
    # Load and Preprocess the Images
    ######################################
    for path in [OD_IMAGE_PATH, ATOM_IMAGE_PATH, DARK_IMAGE_PATH]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"No image found at {path}.")
        
    atom_img = np.load(ATOM_IMAGE_PATH).astype(np.float32)
    bright_img = np.load(BRIGHT_IMAGE_PATH).astype(np.float32)
    dark_img = np.load(DARK_IMAGE_PATH).astype(np.float32)

    # Normalize images w.r.t. atom image max or bright image max
    atom_max = np.max(atom_img)
    atom_img_norm = atom_img / atom_max
    dark_img_norm = dark_img / atom_max  
    bright_max = np.max(bright_img)
    bright_img_norm = bright_img / bright_max
 
    H, W = atom_img_norm.shape

    # Create partial: zero out the circular region around (CENTER_X, CENTER_Y)
    center_y = CENTER_Y if CENTER_Y is not None else (H // 2)
    center_x = CENTER_X if CENTER_X is not None else (W // 2)
    r = RADIUS
    partial = np.copy(atom_img_norm)
    yy, xx = np.ogrid[:H, :W]
    dist_sq = (yy - center_y)**2 + (xx - center_x)**2
    disk_mask = dist_sq <= (r*r)
    partial[disk_mask] = 0.0

    y_min = max(0, center_y - r)
    y_max = min(H, center_y + r)
    x_min = max(0, center_x - r)
    x_max = min(W, center_x + r)
    
    # Crop relevant images
    partial_cropped = partial[y_min:y_max, x_min:x_max]
    atom_img_cropped, atom_img_norm_cropped = atom_img[y_min:y_max, x_min:x_max], atom_img_norm[y_min:y_max, x_min:x_max]
    bright_img_cropped, bright_img_norm_cropped = bright_img[y_min:y_max, x_min:x_max], bright_img_norm[y_min:y_max, x_min:x_max]
    dark_img_cropped, dark_img_norm_cropped = dark_img[y_min:y_max, x_min:x_max], dark_img_norm[y_min:y_max, x_min:x_max]
    
    # Calculate calcoeff between raw images
    calcoeff_raw = np.sum(bright_img_cropped[ROI_ORIG[0]:ROI_ORIG[2], ROI_ORIG[1]:ROI_ORIG[3]])/np.sum(atom_img_cropped[ROI_ORIG[0]:ROI_ORIG[2], ROI_ORIG[1]:ROI_ORIG[3]])
    
    # Calculate original (raw) OD
    epsilon = 1e-6
    ratio_raw = (atom_img_cropped * calcoeff_raw - dark_img_cropped + epsilon) / (bright_img_cropped - dark_img_cropped + epsilon)
    OD_raw = -np.log(ratio_raw)
    cutoff = 3.0
    OD_raw = np.where(OD_raw > cutoff, cutoff, OD_raw)

    # Predict bright image from masked atom image
    partial_tensor = torch.from_numpy(partial_cropped).unsqueeze(0).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        pred_bright_cropped_tsr = model(partial_tensor)
    pred_bright_cropped = pred_bright_cropped_tsr[0,0].cpu().numpy()

    # Calculate calcoeff between predicted and raw image 
    calcoeff_pred = np.sum(pred_bright_cropped[ROI_PRED[0]:ROI_PRED[2], ROI_PRED[1]:ROI_PRED[3]])/np.sum(atom_img_norm_cropped[ROI_PRED[0]:ROI_PRED[2], ROI_PRED[1]:ROI_PRED[3]])
    
    # Print calcoeffs
    print(f"calcoeff_raw: {calcoeff_raw:.6f}")
    print(f"calcoeff_pred: {calcoeff_pred:.6f}")

    # Compute the predicted OD
    epsilon = 1e-6
    ratio_pred = (atom_img_norm_cropped * calcoeff_pred - dark_img_norm_cropped + epsilon) / (pred_bright_cropped - dark_img_norm_cropped + epsilon)
    ratio_pred[ratio_pred<=0] = 1e-6 
    OD_pred = -np.log(ratio_pred)
    cutoff = 3.0
    OD_pred = np.where(OD_pred > cutoff, cutoff, OD_pred)

    end_time = time.time()-start_time
    print(f"OD reconstruction time: {end_time} s")

    # Save predicted OD image
    np.save(SAVE_PRED_OD_PATH, OD_pred)
    print(f"OD prediction .npy saved to {SAVE_PRED_OD_PATH}")

    ######################################
    # Compute and print RMSE values for OD images
    ###################################### 
    rmse_pred = compute_rmse(atom_img_norm_cropped * calcoeff_pred, pred_bright_cropped)
    print(f"RMSE of Predicted OD image: {rmse_pred:.6f}")

    ######################################
    # Display Results
    ######################################
    fig_OD, axes = plt.subplots(2, 3, figsize=(15, 10))

    axes[0,0].imshow(partial_cropped, cmap='gray', vmin=0, vmax=1)
    axes[0,0].set_title("Masked atom image", size=20)
    axes[0,0].axis('off')

    axes[0,1].imshow(atom_img_norm_cropped, cmap='gray', vmin=0, vmax=1)
    axes[0,1].set_title("Atom image", size=20)
    axes[0,1].axis('off')      

    axes[0,2].imshow(pred_bright_cropped, cmap='gray', vmin=0, vmax=1)
    axes[0,2].set_title("Predicted background image", size=20)
    axes[0,2].axis('off')

    sum_OD_pred = np.sum(OD_pred)
    print(f"Predicted OD (sum={sum_OD_pred:.2f})")

    sum_OD_raw = np.sum(OD_raw)
    print(f"Raw OD (sum={sum_OD_raw:.2f})")

    im_predOD = axes[1,0].imshow(OD_pred, cmap='viridis', vmin=0, vmax=0.5)
    axes[1,0].set_title(f"Predicted OD", size=20)
    axes[1,0].axis('off')
    cbar1 = fig_OD.colorbar(im_predOD, ax=axes[1,0], orientation='vertical')
    cbar1.set_label("OD value")

    im_rawOD = axes[1,1].imshow(OD_raw, cmap='viridis', vmin=0, vmax=0.5)
    axes[1,1].set_title(f"Raw OD", size=20)
    axes[1,1].axis('off')
    cbar2 = fig_OD.colorbar(im_rawOD, ax=axes[1,1], orientation='vertical')
    cbar2.set_label("OD value")        

    axes[1,2].axis('off')
    plt.tight_layout()
    plt.savefig(SAVE_FIG_PATH, dpi=300)
    plt.show()
    print(f"Figure saved to {SAVE_FIG_PATH}")

if __name__ == "__main__":
    main()
