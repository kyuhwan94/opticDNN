import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import Dataset, DataLoader, Subset, ConcatDataset, random_split
import numpy as np
import os
import argparse
import json
import random
import time
import matplotlib.pyplot as plt
import csv

######################################
# Shell script run example

# for config in configs/*.json; do
#     python3 train.py --config $config
# done

######################################

######################################
# Configuration
######################################

def load_config(config_path):
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config

def save_configuration_txt(config, config_file="configuration.txt"):
    """
    Save config dict as .txt 
    Args:
        config (dict): Dictionary containing configuration parameters.
        config_file (str): Name of the file to save the configuration.
    """
    # Get the script name if available
    script_name = os.path.basename(__file__) if "__file__" in globals() else "UNKNOWN_SCRIPT"

    # Ensure the save path exists (it is assumed that config contains a 'SAVE_PATH' key)
    os.makedirs(config["SAVE_PATH"], exist_ok=True)
    config_filename = os.path.join(config["SAVE_PATH"], config_file)

    # You can either dump the dict as formatted text or use json.dumps.
    with open(config_filename, "w") as f:
        f.write(f"Script name: {script_name}\n\n")
        f.write("=== Configuration ===\n\n")
        for key, value in config.items():
            # Optionally, pretty-print lists or nested dictionaries:
            if isinstance(value, (list, dict)):
                value_str = json.dumps(value, indent=4)
            else:
                value_str = str(value)
            f.write(f"{key} = {value_str}\n")
    
    print(f"Configuration saved to {config_filename}")

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
# BrightNPYDataset 
######################################

class BrightNPYDataset(Dataset):
    def __init__(
        self, 
        folder_path, 
        center_y=None, 
        center_x=None, 
        radius=100,
        transform=None
    ):
        self.folder_path = folder_path
        self.files = [f for f in os.listdir(folder_path) if f.endswith('.npy')]
        self.center_y = center_y
        self.center_x = center_x
        self.radius = radius
        self.transform = transform


    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file_path = os.path.join(self.folder_path, self.files[idx])
        img = np.load(file_path).astype(np.float32)

        # Normalize if no transform is given
        if self.transform:
            img = self.transform(img)
        else:
            if np.max(img) > 0:
                img = img / np.max(img)

        H, W = img.shape

        # Determine circle center and radius
        cy = self.center_y if self.center_y is not None else (H // 2)
        cx = self.center_x if self.center_x is not None else (W // 2)
        r = self.radius

        # Create partial copy (will zero-out inside the circle)
        partial = np.copy(img)
        y_coords, x_coords = np.ogrid[:H, :W]
        dist_sq = (y_coords - cy)**2 + (x_coords - cx)**2
        mask = dist_sq <= (r*r)
        partial[mask] = 0.0

        y_min = max(0, cy - r)
        y_max = min(H, cy + r)
        x_min = max(0, cx - r)
        x_max = min(W, cx + r)

        # Crop both partial and the original "full" image
        cropped_partial = partial[y_min:y_max, x_min:x_max]
        cropped_full = img[y_min:y_max, x_min:x_max]

        partial_tensor = torch.from_numpy(cropped_partial).unsqueeze(0)
        full_tensor = torch.from_numpy(cropped_full).unsqueeze(0)

        return partial_tensor, full_tensor

######################################
# RMSE Loss
######################################

class RMSELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(self, pred, target):
        mse_val = self.mse(pred, target)
        rmse_val = torch.sqrt(mse_val + 1e-8) # small epsilon for numerical stability
        return rmse_val
    
######################################
# Train
######################################

def train_model(
    model,
    dataloader,
    optimizer,
    criterion,
    device,
    val_loader=None,
    early_stop_patience=50,
    improvement_threshold=0.01,
    max_iterations=50
):
    """  
    Args:
        model: DNN model.
        dataloader: Training DataLoader.
        optimizer: Optimizer(ADAMW in our case).
        criterion: Loss function(RMSE in our case).
        device: torch.device.
        val_loader: Validation DataLoader (optional).
        early_stop_patience: Number of consecutive checks with no sufficient improvement to trigger early stopping.
        improvement_threshold: Required relative improvement in validation loss.
        max_iterations: Safety limit for training iterations.
  
    Returns:
        train_losses, val_losses: Lists of losses recorded during training.
    """
    model.train()
    train_losses = []
    val_losses = []

    best_val_loss = float("inf")
    no_improve_count = 0

    iteration = 0
    while True:
        iteration += 1
        start_time = time.time()

        total_loss = 0.0
        for partial, full in dataloader:
            partial, full = partial.to(device), full.to(device)
            optimizer.zero_grad()
            pred = model(partial)
            loss = criterion(pred, full)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        train_losses.append(avg_loss)

        epoch_time = time.time() - start_time
        
        print(f"Iteration {iteration}, Train RMSE: {avg_loss:.6f}, Time: {epoch_time:.2f} s")

        # -- Validation & Early Stopping check
        if val_loader is not None:
            val_loss = evaluate_model(model, val_loader, criterion, device)
            val_losses.append(val_loss)
            print(f"Validation RMSE: {val_loss:.6f}")

            # Check for improvement
            if val_loss < best_val_loss * (1 - improvement_threshold):
                best_val_loss = val_loss
                no_improve_count = 0
            else:
                no_improve_count += 1

            if no_improve_count >= early_stop_patience:
                print(f"Early stopping triggered at iteration {iteration}. "
                      f"Validation RMSE has not improved by {improvement_threshold*100}% "
                      f"for {early_stop_patience} consecutive checks.")
                break

        # -- Safety check to prevent infinite loops
        if iteration >= max_iterations:
            print(f"Reached max_iterations={max_iterations}. Stopping.")
            break

    return train_losses, val_losses

######################################
# Evaluate Model on a Dataset
######################################

def evaluate_model(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for partial, full in dataloader:
            partial, full = partial.to(device), full.to(device)
            pred = model(partial)
            loss = criterion(pred, full)
            total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    return avg_loss

######################################
# Visualization (to check things are on the right track)
######################################

def compute_psd2d(image_2d):
    F = np.fft.fft2(image_2d)
    F_shifted = np.fft.fftshift(F)
    return np.abs(F_shifted)**2

def bin_1d(data, bin_size=5):
    length = len(data)
    nbins = length // bin_size
    truncated_length = nbins * bin_size
    data_truncated = data[:truncated_length].reshape(nbins, bin_size)
    return data_truncated.mean(axis=1)

def visualize_prediction(model, dataloader, device, 
                         save_path="prediction.png", 
                         psd_save_path="psd_comparison.png",
                         line_cut_save_path="horizontal_cut.png"):
    model.eval()
    partial_batch, full_batch = next(iter(dataloader))
    partial_batch, full_batch = partial_batch.to(device), full_batch.to(device)

    with torch.no_grad():
        predicted_batch = model(partial_batch)

    partial_img = partial_batch[0, 0].cpu().numpy()
    full_img = full_batch[0, 0].cpu().numpy()
    pred_img = predicted_batch[0, 0].cpu().numpy()
    diff_img = np.abs(full_img - pred_img)

    # (1) Partial, Ground Truth, Predicted, Difference
    fig, axes = plt.subplots(1, 4, figsize=(16, 5))
    axes[0].imshow(partial_img, cmap='gray', vmin=0, vmax=1)
    axes[0].set_title("Partial Input")
    axes[0].axis('off')

    axes[1].imshow(full_img, cmap='gray', vmin=0, vmax=1)
    axes[1].set_title("Full Ground Truth")
    axes[1].axis('off')

    axes[2].imshow(pred_img, cmap='gray', vmin=0, vmax=1)
    axes[2].set_title("Predicted")
    axes[2].axis('off')

    axes[3].imshow(diff_img, cmap='gray', vmin=0, vmax=1)
    axes[3].set_title("Diff")
    axes[3].axis('off')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.show()

    # (2) PSD
    full_psd = compute_psd2d(full_img)
    pred_psd = compute_psd2d(pred_img)
    full_psd_log = np.log10(full_psd + 1e-12)
    pred_psd_log = np.log10(pred_psd + 1e-12)
    diff_psd_log = full_psd_log - pred_psd_log

    vmin = min(full_psd_log.min(), pred_psd_log.min())
    vmax = max(full_psd_log.max(), pred_psd_log.max())

    fig_psd, ax_psd = plt.subplots(1, 3, figsize=(15, 6))
    im0 = ax_psd[0].imshow(full_psd_log, cmap='viridis', vmin=vmin, vmax=vmax)
    ax_psd[0].axis('off')
    ax_psd[0].set_title("2D PSD: GT")
    cbar0 = fig_psd.colorbar(im0, ax=ax_psd[0])
    cbar0.set_label("log10(PSD)")

    im1 = ax_psd[1].imshow(pred_psd_log, cmap='viridis', vmin=vmin, vmax=vmax)
    ax_psd[1].axis('off')
    ax_psd[1].set_title("2D PSD: Pred")
    cbar1 = fig_psd.colorbar(im1, ax=ax_psd[1])
    cbar1.set_label("log10(PSD)")

    im2 = ax_psd[2].imshow(diff_psd_log, cmap='viridis', vmin=-vmax/3, vmax=vmax/3)
    ax_psd[2].axis('off')
    ax_psd[2].set_title("2D PSD: log10 diff")
    cbar2 = fig_psd.colorbar(im2, ax=ax_psd[2])
    cbar2.set_label("log10(PSD)")

    plt.tight_layout()
    plt.savefig(psd_save_path, dpi=300)
    plt.show()

    # (3) Horizontal line cuts
    center_row = full_psd_log.shape[0] // 2 + 1
    line_cut_gt = full_psd_log[center_row, :]
    line_cut_pred = pred_psd_log[center_row, :]
    line_cut_diff = diff_psd_log[center_row, :]

    bin_size = 5
    line_cut_gt_binned   = bin_1d(line_cut_gt,   bin_size=bin_size)
    line_cut_pred_binned = bin_1d(line_cut_pred, bin_size=bin_size)
    line_cut_diff_binned = bin_1d(line_cut_diff, bin_size=bin_size)

    x_axis = np.arange(len(line_cut_gt))
    x_axis_binned = np.arange(len(line_cut_gt_binned)) * bin_size + (bin_size / 2)

    fig_cut, ax_cut = plt.subplots(1, 3, figsize=(18, 5))
    ax_cut[0].plot(x_axis, line_cut_gt, label="GT PSD (raw)")
    ax_cut[0].plot(x_axis_binned, line_cut_gt_binned, 'o-', label=f"GT PSD binned({bin_size})")
    ax_cut[0].set_title("Horiz. Cut (GT PSD)")
    ax_cut[0].legend()

    ax_cut[1].plot(x_axis, line_cut_pred, color='orange', label="Pred PSD (raw)")
    ax_cut[1].plot(x_axis_binned, line_cut_pred_binned, 'o-', color='red', label=f"Pred binned({bin_size})")
    ax_cut[1].set_title("Horiz. Cut (Pred PSD)")
    ax_cut[1].legend()

    ax_cut[2].plot(x_axis, line_cut_diff, label="log10 diff (raw)")
    ax_cut[2].plot(x_axis_binned, line_cut_diff_binned, 'o-', label=f"log10 diff binned({bin_size})")
    ax_cut[2].set_title("Horiz. Cut (diff)")
    ax_cut[2].legend()

    plt.tight_layout()
    plt.savefig(line_cut_save_path, dpi=300)
    plt.show()

######################################
# Main
######################################

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True,
                        help="Path to the JSON configuration file")
    args = parser.parse_args()

    config = load_config(args.config)

    # Update config
    PRETRAIN_ENABLED = config["PRETRAIN_ENABLED"]
    REPLAY_ENABLED = config["REPLAY_ENABLED"]
    TRANSFER_LEARNING_ENABLED = config["TRANSFER_LEARNING_ENABLED"]
    VAL_RATIO = config["VAL_RATIO"]
    REPLAY_RATIO = config["REPLAY_RATIO"]
    SAVE_PATH = config["SAVE_PATH"]
    SAVE_WEIGHTS = config["SAVE_WEIGHTS"]
    LOAD_WEIGHTS = config["LOAD_WEIGHTS"]
    PATIENCE = config["PATIENCE"]
    IMPROVEMENT_THRESHOLD = config["IMPROVEMENT_THRESHOLD"]
    FEATURES = config["FEATURES"]
    NUM_GROUPS = config["NUM_GROUPS"]
    BATCH_SIZE = config["BATCH_SIZE"]
    BATCH_MOMENTUM = config["BATCH_MOMENTUM"]
    PRETRAIN_LR = config["PRETRAIN_LR"]
    TRANSFER_LR = config["TRANSFER_LR"]
    WEIGHT_DECAY = config["WEIGHT_DECAY"]
    CENTER_X = config["CENTER_X"]
    CENTER_Y = config["CENTER_Y"]
    RADIUS = config["RADIUS"]
    PRETRAIN_FOLDER_PATHS = config["PRETRAIN_FOLDER_PATHS"]
    EVAL_FOLDER_PATHS = config["EVAL_FOLDER_PATHS"]
    NEW_FOLDER_PATHS = config["NEW_FOLDER_PATHS"]

    # Create save directory and save config

    save_configuration_txt(config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # ---------------------------
    # 1) Prepare PRETRAIN data
    # ---------------------------

    print("=== Preparing PRETRAIN data from multiple folders ===")
    pretrain_datasets = []
    for folder_path in PRETRAIN_FOLDER_PATHS:
        if not os.path.isdir(folder_path):
            raise NotADirectoryError(f"Folder path '{folder_path}' does not exist.")
        dataset = BrightNPYDataset(
            folder_path=folder_path,
            center_y=CENTER_Y,
            center_x=CENTER_X,
            radius=RADIUS
        )
        pretrain_datasets.append(dataset)
        print(f"Loaded {len(dataset)} samples from '{folder_path}'")
    combined_pretrain_dataset = ConcatDataset(pretrain_datasets)
    print(f"Total PRETRAIN dataset size: {len(combined_pretrain_dataset)}")
    total_size = len(combined_pretrain_dataset)
    val_size = int(total_size * VAL_RATIO)
    train_size = total_size - val_size
       
    train_pretrain_dataset, val_pretrain_dataset = random_split(combined_pretrain_dataset, [train_size, val_size])
    combined_pretrain_loader = DataLoader(combined_pretrain_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=1)
    pretrain_loader = DataLoader(train_pretrain_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=1)
    val_pretrain_loader = DataLoader(val_pretrain_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=1)

    # ---------------------------
    # 2) Initialize model and optimizer
    # ---------------------------
    model = UNet(in_ch=1, out_ch=1, features=FEATURES, num_groups=NUM_GROUPS).to(device)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of trainable parameters: {num_params}")
    optimizer = optim.AdamW(model.parameters(), lr=PRETRAIN_LR, weight_decay=WEIGHT_DECAY)
    criterion = RMSELoss()

    # ---------------------------
    # 3) Train on PRETRAIN data
    # ---------------------------
    if PRETRAIN_ENABLED:

        training_start_time = time.time()

        print("=== Training on PRETRAIN data from scratch ===")
        train_losses_pretrain, val_losses_pretrain = train_model(
            model=model,
            dataloader=pretrain_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            val_loader=val_pretrain_loader,
            early_stop_patience=PATIENCE,
            improvement_threshold=IMPROVEMENT_THRESHOLD,
        )
        
        # Record overall training end time and report total training duration
        training_end_time = time.time()
        total_training_time = training_end_time - training_start_time
        print(f"Total training time: {total_training_time:.2f} seconds")

        # Save the total training time to a text file
        time_file = os.path.join(SAVE_PATH, "total_pretrain_training_time.txt")
        with open(time_file, "w") as f:
            f.write(f"Total training time: {total_training_time:.2f} seconds\n")

        # Save epoch vs loss
        with open(os.path.join(SAVE_PATH, "epoch_vs_loss_old.csv"), "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["epoch", "train_loss"])
            for ep, loss_val in enumerate(train_losses_pretrain, start=1):
                writer.writerow([ep, loss_val])
        with open(os.path.join(SAVE_PATH, "epoch_vs_val_loss_old.csv"), "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["epoch", "val_loss"])
            for ep, loss_val in enumerate(val_losses_pretrain, start=1):
                writer.writerow([ep, loss_val])

        # Save evaluation
        old_loss = evaluate_model(model, combined_pretrain_loader, criterion, device)
        print(f"Evaluation on OLD data: RMSE: {old_loss:.6f}")
        with open(os.path.join(SAVE_PATH, f"old_loss.csv"), "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Old_Data_RMSE_After_Training"])
            writer.writerow([old_loss])

        # Save model
        model_save_path_pretrain = os.path.join(SAVE_PATH, SAVE_WEIGHTS)
        torch.save(model.state_dict(), model_save_path_pretrain)
        print(f"Pretrain model weights saved to {model_save_path_pretrain}")

    else:
        model_load_path_pretrain = LOAD_WEIGHTS
        print("=== Skipping pretraining. Loading saved weights ===")
        if not os.path.isfile(model_load_path_pretrain):
            raise FileNotFoundError(
                f"Cannot find '{model_load_path_pretrain}'. Set TRAIN_PRETRAIN_DATA = True first to create it, or fix the path."
            )
        model.load_state_dict(torch.load(model_load_path_pretrain, map_location=device))
    
    # ---------------------------
    # 4) Train on NEW data
    # ---------------------------

    if TRANSFER_LEARNING_ENABLED:

        print("TRANSFER learning is enabled")
               
        # Smaller learning rate when replay
        optimizer = optim.AdamW(model.parameters(), lr=TRANSFER_LR, weight_decay=WEIGHT_DECAY)
        
        # ---------------------------
        # 5) Dynamic Replay Pool Setup:
        # Initialize two dynamic pools:
        #   old_pool_dynamic: starts as all folders in PREV_FOLDER_PATHS.
        #   replay_pool_dynamic: initially empty.
        # ---------------------------
        pretrain_pool_dynamic = list(PRETRAIN_FOLDER_PATHS)
        replay_pool_dynamic = []
        
        # ---------------------------
        # 6) Process each folder in NEW_FOLDER_PATHS individually for transfer learning
        # ---------------------------

        for new_folder in NEW_FOLDER_PATHS:
            training_start_time = time.time()
            print(f"=== Processing new data from folder: {new_folder} ===")
            # Load new data from this folder
            new_data = BrightNPYDataset(
                folder_path=new_folder,
                center_y=CENTER_Y,
                center_x=CENTER_X,
                radius=RADIUS
            )
            if REPLAY_ENABLED:

                print("Replay is ENABLED. Gathering pretrain & replay samples.")
                # Sample REPLAY_RATIO from each folder in pretrain_pool_dynamic:
                prev_samples = []
                for folder in pretrain_pool_dynamic:
                    ds = BrightNPYDataset(
                        folder_path=folder,
                        center_y=CENTER_Y,
                        center_x=CENTER_X,
                        radius=RADIUS
                    )
                    num_old = len(ds)
                    if num_old > 0:
                        indices = list(range(num_old))
                        random.shuffle(indices)
                        subset_size = max(1, int(REPLAY_RATIO * num_old))
                        chosen_indices = indices[:subset_size]
                        ds_subset = Subset(ds, chosen_indices)
                        prev_samples.append(ds_subset)
                
                # Sample REPLAY_RATIO from each folder in replay_pool_dynamic:
                replay_samples = []
                for folder in replay_pool_dynamic:
                    ds = BrightNPYDataset(
                        folder_path=folder,
                        center_y=CENTER_Y,
                        center_x=CENTER_X,
                        radius=RADIUS
                    )
                    num_rep = len(ds)
                    if num_rep > 0:
                        indices = list(range(num_rep))
                        random.shuffle(indices)
                        subset_size = max(1, int(REPLAY_RATIO * num_rep))
                        chosen_indices = indices[:subset_size]
                        ds_subset = Subset(ds, chosen_indices)
                        replay_samples.append(ds_subset)

                # Combine new data with samples from pretrain and replay pools:
                datasets_to_concat = [new_data]
                if prev_samples:
                    datasets_to_concat.append(ConcatDataset(prev_samples))
                if replay_samples:
                    datasets_to_concat.append(ConcatDataset(replay_samples))
                replay_dataset = ConcatDataset(datasets_to_concat)
            else:
                print("Replay is DISABLED. Training only on new data.")
                replay_dataset = new_data

            print(f"Final combined replay dataset size: {len(replay_dataset)}")

            # Split replay dataset into training and validation portions
            total_size_new = len(replay_dataset)
            val_size_new = int(total_size_new * VAL_RATIO)
            train_size_new = total_size_new - val_size_new

            train_replay_dataset, val_replay_dataset = random_split(replay_dataset, [train_size_new, val_size_new])
            replay_train_loader = DataLoader(train_replay_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=1)
            replay_val_loader = DataLoader(val_replay_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=1)

            # Train on replay dataset (new data from this folder + replay samples)
            print("=== Training on new data + replay for current folder ===")
            train_losses_new, val_losses_new = train_model(
                model=model,
                dataloader=replay_train_loader,
                optimizer=optimizer,
                criterion=criterion,
                device=device,
                val_loader=replay_val_loader,
                early_stop_patience=PATIENCE,
                improvement_threshold=IMPROVEMENT_THRESHOLD,
            )

            # Use a folder tag; here we use the base name of the replay folder.
            folder_tag = os.path.basename(os.path.dirname(os.path.dirname(os.path.normpath(new_folder))))
            
            # Record overall training end time and report total training duration       
            training_end_time = time.time()
            total_training_time = training_end_time - training_start_time
            print(f"Total training time: {total_training_time:.2f} seconds")

            # Save the total training time to a text file
            time_file = os.path.join(SAVE_PATH, f"{folder_tag}_training_time.txt")
            with open(time_file, "w") as f:
                f.write(f"Total training time: {total_training_time:.2f} seconds\n")

            # Save train/validation losses for this folder
            with open(os.path.join(SAVE_PATH, f"epoch_vs_loss_new_data_{folder_tag}.csv"), "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["epoch", "train_loss"])
                for ep, loss_val in enumerate(train_losses_new, start=1):
                    writer.writerow([ep, loss_val])
            with open(os.path.join(SAVE_PATH, f"epoch_vs_val_loss_new_data_{folder_tag}.csv"), "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["epoch", "val_loss"])
                for ep, loss_val in enumerate(val_losses_new, start=1):
                    writer.writerow([ep, loss_val])
            
            # Save the model weights for this replay training iteration
            model_save_path = os.path.join(SAVE_PATH, f"model_weights_replay_{folder_tag}.pth")
            torch.save(model.state_dict(), model_save_path)
            print(f"Final model weights saved to {model_save_path}")

            # ---------------------------
            # 7) Evaluate on both EVALUATION FOLDER data and on the current new folder
            # ---------------------------
            eval_datasets = []
        
            for folder_path in EVAL_FOLDER_PATHS:
                if not os.path.isdir(folder_path):
                    raise NotADirectoryError(f"Folder path '{folder_path}' does not exist.")
                eval_dataset = BrightNPYDataset(
                    folder_path=folder_path,
                    center_y=CENTER_Y,
                    center_x=CENTER_X,
                    radius=RADIUS
                )
                eval_datasets.append(eval_dataset)
                print(f"Loaded {len(eval_dataset)} samples from '{folder_path}'")
            combined_eval_dataset = ConcatDataset(eval_datasets)
            print(f"Total EVAL dataset size: {len(combined_eval_dataset)}")
            combined_eval_loader = DataLoader(combined_eval_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=1)
            eval_loss = evaluate_model(model, combined_eval_loader, criterion, device)
            print(f"Evaluation on EVAL data: RMSE: {eval_loss:.6f}")
            with open(os.path.join(SAVE_PATH, f"eval_loss_after_replay_{folder_tag}.csv"), "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["EVAL_Data_RMSE_After_Training"])
                writer.writerow([eval_loss])

            new_loss = evaluate_model(model, DataLoader(new_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=1), criterion, device)
            print(f"Evaluation on new folder {folder_tag}: RMSE: {new_loss:.6f}")
            with open(os.path.join(SAVE_PATH, f"new_data_loss_after_replay_{folder_tag}.csv"), "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["New_Data_RMSE_After_Training"])
                writer.writerow([new_loss])

            # ---------------------------
            # 8) Update dynamic pools:
            #    - Append the current replay folder to replay_pool_dynamic.
            #    - Remove the oldest folder from old_pool_dynamic (if available).
            # ---------------------------
            if REPLAY_ENABLED:
                parts = new_folder.strip('/').split('/')
                date_str = parts[1].replace('_full','')
                replay_folder = f"../{date_str}/npy_{date_str}/bright_npy/"  
                replay_pool_dynamic.append(replay_folder)
                print(f"Updating replay pool. Adding {replay_folder}")
                if len(pretrain_pool_dynamic) > 0:
                    removed = pretrain_pool_dynamic.pop(0)
                    print(f"Removed oldest old folder from dynamic pool: {removed}")
            else: print("Replay is disabled, not updating replay pools.")

    else: print("TRANSFER learning is disabled")

    # ---------------------------
    # 9) Visualization
    # ---------------------------

    # Visualize using the old data
    pretrain_data = BrightNPYDataset(
        folder_path=PRETRAIN_FOLDER_PATHS[0],
        center_y=CENTER_Y,
        center_x=CENTER_X,
        radius=RADIUS
    )
    visualize_prediction(
        model=model,
        dataloader=DataLoader(pretrain_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=1),
        device=device,
        save_path=os.path.join(SAVE_PATH, "prediction.png"),
        psd_save_path=os.path.join(SAVE_PATH, "psd_comparison.png"),
        line_cut_save_path=os.path.join(SAVE_PATH, "horizontal_cut.png")
    )