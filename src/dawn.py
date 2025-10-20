import os
import argparse
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

class DAWN(nn.Module):
    """
    DAWN module: Learns IPC-like filters as a trainable preprocessing layer.
    This module takes raw images and produces augmented feature maps that 
    highlight edges/structures at multiple scales.
    By default, it outputs the original image (identity) plus two extra channels:
    - fine-scale detail features
    - coarse-scale detail features
    """
    def __init__(self, in_channels: int):
        super(DAWN, self).__init__()
        self.in_channels = in_channels
        # Convolution for fine-scale edge detection (small receptive field)
        self.conv_fine = nn.Conv2d(in_channels, 1, kernel_size=3, padding=1)
        # Convolution for coarse-scale structure (larger receptive field via dilation)
        self.conv_coarse = nn.Conv2d(in_channels, 1, kernel_size=3, padding=4, dilation=4)
        # Initialize filters (optional): we can start with small random weights. 
        # One could also initialize conv_fine like a Sobel filter, etc., but learnable is fine.
        nn.init.kaiming_normal_(self.conv_fine.weight, nonlinearity='relu')
        nn.init.constant_(self.conv_fine.bias, 0.0)
        nn.init.kaiming_normal_(self.conv_coarse.weight, nonlinearity='relu')
        nn.init.constant_(self.conv_coarse.bias, 0.0)
        # Output channels = original channels + 2 (fine and coarse)
        self.out_channels = in_channels + 2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (N, C, H, W), where C = in_channels (e.g., 3 for RGB).
        # Compute fine-scale edges
        fine = self.conv_fine(x)            # (N, 1, H, W)
        # Compute coarse-scale edges (via dilated conv)
        coarse = self.conv_coarse(x)        # (N, 1, H, W)
        # Optionally, apply non-linearities (e.g., ReLU or abs) to emulate edge magnitude.
        # Here we use ReLU to focus on edge magnitude (dropping negative responses).
        fine = F.relu(fine)
        coarse = F.relu(coarse)
        # Concatenate original image with the fine and coarse feature maps
        # If original had C channels, output will have C+2 channels.
        out = torch.cat([x, fine, coarse], dim=1)
        return out

# --- U-Net Model Definition ---
class DoubleConv(nn.Module):
    """Helper module: two convolution layers (with BatchNorm+ReLU) used in each U-Net block."""
    def __init__(self, in_channels: int, out_channels: int):
        super(DoubleConv, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class UNet(nn.Module):
    """
    U-Net architecture for segmentation. 
    This implementation has an encoder (downsampling path) and decoder (upsampling path) with skip connections.
    - in_channels: number of channels in input images/feature maps.
    - out_classes: number of output classes (for binary segmentation, out_classes=1).
    """
    def __init__(self, in_channels: int, out_classes: int = 1):
        super(UNet, self).__init__()
        # Define the encoder (downsampling path)
        # We'll use four down blocks with feature sizes increasing by factor of 2.
        features = [64, 128, 256, 512]  # feature map channels at each down step
        self.enc_blocks = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        prev_channels = in_channels
        for feat in features:
            self.enc_blocks.append(DoubleConv(prev_channels, feat))
            prev_channels = feat
        # Bottleneck layer (after the last downsampling)
        # self.bottleneck = DoubleConv(prev_channels, prev_channels * 2)  # expand to 1024 if prev was 512
        # # Define the decoder (upsampling path)
        # self.dec_blocks = nn.ModuleList()
        # self.upconvs = nn.ModuleList()
        # # Decoder will mirror the encoder features in reverse
        # for feat in reversed(features):
        #     # Upsampling convolution (transpose conv) to half the channel count
        #     self.upconvs.append(nn.ConvTranspose2d(prev_channels * 2, feat, kernel_size=2, stride=2))
        #     # After upsampling, we'll concat with corresponding encoder feature (skip connection),
        #     # so the DoubleConv will have input channels = feat (from up) + feat (from skip) = 2*feat.
        #     self.dec_blocks.append(DoubleConv(feat * 2, feat))
        #     prev_channels = feat  # update for next layer (feat is the current output channels after decoding)
        self.bottleneck = DoubleConv(prev_channels, prev_channels * 2)
        bottleneck_channels = prev_channels * 2  # store for clarity
        self.dec_blocks = nn.ModuleList()
        self.upconvs = nn.ModuleList()
        for feat in reversed(features):
            # use bottleneck_channels (1024) for the first upconv, then halve each time
            self.upconvs.append(nn.ConvTranspose2d(bottleneck_channels, feat, kernel_size=2, stride=2))
            self.dec_blocks.append(DoubleConv(feat * 2, feat))
            bottleneck_channels = feat  # update for next iteration
        # Final 1x1 convolution to get the segmentation map (output has out_classes channels)
        self.final_conv = nn.Conv2d(features[0], out_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder path
        enc_features = []  # to store outputs for skip connections
        for enc_block in self.enc_blocks:
            x = enc_block(x)
            enc_features.append(x)
            x = self.pool(x)
        # Bottleneck
        x = self.bottleneck(x)
        # Decoder path
        # We have as many upconv+dec_block pairs as encoder blocks
        for i, dec_block in enumerate(self.dec_blocks):
            # Note: enc_features were appended in order; the last appended is the last encoder block before bottleneck
            # We need to use them in reverse order for decoding.
            # Retrieve the corresponding skip feature from enc_features
            skip_feat = enc_features[-(i+1)]
            # Upsample current feature map
            x = self.upconvs[i](x)
            # Concatenate skip feature and upsampled feature maps (make sure sizes match)
            # If necessary, crop or pad can be done here to handle any size mismatch due to rounding, but with even dims it's fine.
            if x.shape[2:] != skip_feat.shape[2:]:
                # In case of odd input dimensions, adjust by center-cropping the skip features
                # (This step ensures the sizes match for concatenation.)
                diffY = skip_feat.size(2) - x.size(2)
                diffX = skip_feat.size(3) - x.size(3)
                x = F.pad(x, [diffX // 2, diffX - diffX//2, diffY // 2, diffY - diffY//2])
            x = torch.cat([skip_feat, x], dim=1)  # concat along channel dimension
            # Apply double conv on the concatenated feature
            x = dec_block(x)
        # Final output layer
        logits = self.final_conv(x)
        return logits

# --- Combined Model (DAWN + UNet) ---
class DawnUNet(nn.Module):
    """
    Combined model that optionally includes the DAWN preprocessing module before the U-Net.
    - If use_dawn is True, the DAWN module processes the input and its output is fed into the U-Net.
    - If use_dawn is False, the U-Net is applied directly to the raw input.
    """
    def __init__(self, in_channels: int = 3, out_classes: int = 1, use_dawn: bool = True):
        super(DawnUNet, self).__init__()
        self.use_dawn = use_dawn
        if use_dawn:
            self.dawn = DAWN(in_channels)
            # Set up U-Net to accept the augmented channels from DAWN
            self.unet = UNet(in_channels=self.dawn.out_channels, out_classes=out_classes)
        else:
            # No DAWN: use a regular U-Net with the original input channels
            self.dawn = None
            self.unet = UNet(in_channels=in_channels, out_classes=out_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: raw input image (e.g., RGB)
        if self.use_dawn:
            x = self.dawn(x)
        logits = self.unet(x)
        return logits

# --- Dataset and Training Loop (example usage) ---
class SegmentationDataset(Dataset):
    """
    Dataset for segmentation that reads images and corresponding mask files from directories.
    Expects the directories to contain files with matching names for images and masks.
    """
    def __init__(self, images_dir: str, masks_dir: str, transform_image=None, transform_mask=None):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.image_files = sorted(os.listdir(images_dir))
        self.mask_files = sorted(os.listdir(masks_dir))
        assert len(self.image_files) == len(self.mask_files), "Mismatch in number of images and masks"
        self.transform_image = transform_image
        self.transform_mask = transform_mask

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.images_dir, self.image_files[idx])
        mask_path = os.path.join(self.masks_dir, self.mask_files[idx])
        # Open image (as RGB) and mask (as grayscale)
        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")
        # Apply transformations if provided (e.g., resize and to-tensor)
        if self.transform_image:
            image = self.transform_image(image)
        if self.transform_mask:
            mask = self.transform_mask(mask)
        # Ensure mask is binary (0 or 1) if it's not already
        # (If mask pixels are 0 and 255, after ToTensor they'll be 0.0 and 1.0)
        mask = (mask > 0.5).float()  # threshold to binary (if needed)
        return image, mask

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train DAWN+U-Net segmentation model")
    parser.add_argument("-i", "--train_images", type=str, default="im-good",
                        help="Path to the directory of training input images")
    parser.add_argument("-m", "--train_masks", type=str, default="gt-good",
                        help="Path to the directory of training ground truth masks")
    parser.add_argument("--use_dawn", action="store_true", help="Enable DAWN module preprocessing")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=8, help="Training batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate for Adam optimizer")
    args = parser.parse_args()

    # Create dataset and data loader
    transform_img = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
        # Note: We could add normalization here if needed, e.g., transforms.Normalize(...)
    ])
    transform_mask = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])
    train_dataset = SegmentationDataset(args.train_images, args.train_masks, 
                                        transform_image=transform_img, transform_mask=transform_mask)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    # Initialize model, loss, optimizer
    model = DawnUNet(in_channels=3, out_classes=1, use_dawn=args.use_dawn)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    loss_fn = nn.BCEWithLogitsLoss()  # appropriate for binary segmentation (combines sigmoid + BCELoss)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Training loop
    model.train()
    for epoch in range(20):
        total_loss, total_dice, total_iou = 0.0, 0.0, 0.0

        for batch_idx, (images, masks) in enumerate(train_loader):
            print(f"  → Training batch {batch_idx+1}/{len(train_loader)}")
            images, masks = images.to(device), masks.to(device)
            optimizer.zero_grad()
            logits = model(images)
            loss = loss_fn(logits, masks)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * images.size(0)

            # --- Dice & IoU for this batch ---
            with torch.no_grad():
                probs = torch.sigmoid(logits)
                preds = (probs > 0.5).float()
                intersection = (preds * masks).sum(dim=(1,2,3))
                union = preds.sum(dim=(1,2,3)) + masks.sum(dim=(1,2,3))
                dice = ((2 * intersection + 1e-7) / (union + 1e-7)).mean().item()
                iou = ((intersection + 1e-7) / (union - intersection + 1e-7)).mean().item()

            total_dice += dice * images.size(0)
            total_iou += iou * images.size(0)

        avg_loss = total_loss / len(train_loader.dataset)
        avg_dice = total_dice / len(train_loader.dataset)
        avg_iou = total_iou / len(train_loader.dataset)

        print(f"Epoch {epoch+1}/20 - Loss: {avg_loss:.4f}, Dice: {avg_dice:.3f}, IoU: {avg_iou:.3f}")
    
    # === Visualize predictions on a few examples ===
    model.eval()
    os.makedirs("output_visual", exist_ok=True)

    with torch.no_grad():
        for idx, (img, gt_mask) in enumerate(train_loader):
            img = img.to(device)
            logits = model(img)
            pred_mask = (torch.sigmoid(logits) > 0.5).float()

            # Show only the first image in batch
            img_np = img[0].cpu().permute(1, 2, 0).numpy()
            pred_np = pred_mask[0][0].cpu().numpy()
            gt_np = gt_mask[0][0].cpu().numpy()

            fig, axs = plt.subplots(1, 3, figsize=(12, 4))
            axs[0].imshow(img_np)
            axs[0].set_title("Input Image")
            axs[1].imshow(pred_np, cmap='gray')
            axs[1].set_title("Predicted Mask")
            axs[2].imshow(gt_np, cmap='gray')
            axs[2].set_title("Ground Truth")
            for ax in axs:
                ax.axis('off')
            plt.tight_layout()
            plt.savefig(f"output_visual/sample_{idx}.png")
            plt.close()
            
            if idx >= 4:
                break  # only save first 5 batches

