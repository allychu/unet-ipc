"""
Evaluate data on a saved U-Net checkpoint after training

Args:
- `-d`: Path to dataset should include train, validation, and test folders
    - each train/valid/test folder should contain two subfolders for input and ground truth segmentations
- `-c`: Path to the saved checkpoint file
- `-i`: Set flag to use IPC input channels
- `-m`: Set flag if using IPC as ground truth

NOTE: `-i` and `-m` should not be set at the same time!
"""

import os
import torch
import argparse
import pytorch_lightning as pl
import matplotlib.pyplot as plt
import segmentation_models_pytorch as smp
import torchvision.transforms as transforms

from PIL import Image
from torch.optim import lr_scheduler
from torch.utils.data import Dataset, DataLoader

from ipc_transform import IPCTransform


parser = argparse.ArgumentParser()
parser.add_argument('-d', "--data_dir", type=str, default="../data/DIS-", help="Path to the input imgs directory")
parser.add_argument('-i', "--ipc", action='store_true', help="Use IPC computations as input")
parser.add_argument('-m', "--mode_ipc", action='store_true', help="Train with IPC computations as ground truth")
parser.add_argument('-c', '--checkpt_path', default="/Users/KayK/projects/bv-proj/unet-ipc-project/lightning_logs/version_3/checkpoints/epoch=199-step=200.ckpt")
args = parser.parse_args()

data_dir = args.data_dir
checkpoint_path = args.checkpt_path
use_ipc = args.ipc
train_for_ipc = args.mode_ipc


class SegmentationDataset(Dataset):
    def __init__(self, images_dir, masks_dir, image_transform=None, mask_transform=None):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.image_transform = image_transform
        self.mask_transform = mask_transform
        self.image_filenames = sorted(os.listdir(images_dir))[:10]  # Ensure matching order
        self.mask_filenames = sorted(os.listdir(masks_dir))[:10]

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        img_path = os.path.join(self.images_dir, self.image_filenames[idx])
        mask_path = os.path.join(self.masks_dir, self.mask_filenames[idx])

        image = Image.open(img_path).convert("L")
        mask = Image.open(mask_path).convert("L")  

        if self.image_transform:
            image = self.image_transform(image)
        if self.mask_transform:
            mask = self.mask_transform(mask)

        return {"image":image, "mask":mask} # Return image and mask tensors
    
def get_dataset_and_loader(train_images_dir, train_masks_dir, shuffle=False):
    # Define transformations 
    img_transform = transforms.Compose([
                        transforms.Resize((256, 256)),  
                        transforms.ToTensor(),  
                    ])
    mask_transform = img_transform
    if use_ipc:
        print("===USING IPC FOR ADDITIONAL 2 CHANNEL INPUT===")
        img_transform = transforms.Compose([
            transforms.Resize((256, 256)),  
            IPCTransform(),
            transforms.ToTensor(),  
        ])
    else:
        print("===STANDARD TRAINING, NO IPC===")


    # Load dataset
    train_dataset = SegmentationDataset(train_images_dir, 
                                        train_masks_dir, 
                                        img_transform,
                                        mask_transform)

    # Example usage
    train_loader = DataLoader(train_dataset, 
                              batch_size=16, 
                              shuffle=shuffle)

    # Test loading
    for batch in train_loader:
        # Access the image and mask from the batch dictionary
        images = batch["image"]
        masks = batch["mask"]
        print(f"Image batch shape: {images.shape}, Mask batch shape: {masks.shape}")
        break

    return train_dataset, train_loader


class UnetModel(pl.LightningModule):
    def __init__(self, arch, encoder_name, in_channels, out_classes, **kwargs):
        super().__init__()
        self.model = smp.create_model(
            arch,
            encoder_name=encoder_name,
            in_channels=in_channels,
            classes=out_classes,
            **kwargs,
        )
        # preprocessing parameteres for image
        params = smp.encoders.get_preprocessing_params(encoder_name)
        self.register_buffer("std", torch.tensor(params["std"]).view(1, 3, 1, 1))
        self.register_buffer("mean", torch.tensor(params["mean"]).view(1, 3, 1, 1))

        # for image segmentation dice loss could be the best first choice
        self.loss_fn = smp.losses.DiceLoss(smp.losses.BINARY_MODE, from_logits=True)

        # initialize step metics
        self.training_step_outputs = []
        self.validation_step_outputs = []
        self.test_step_outputs = []

    def forward(self, image):
        # normalize image here
        image = (image - self.mean) / self.std
        mask = self.model(image)
        return mask

    def shared_step(self, batch, stage):
        image = batch["image"]

        # Shape of the image should be (batch_size, num_channels, height, width)
        # if you work with grayscale images, expand channels dim to have [batch_size, 1, height, width]
        assert image.ndim == 4

        # Check that image dimensions are divisible by 32,
        # encoder and decoder connected by `skip connections` and usually encoder have 5 stages of
        # downsampling by factor 2 (2 ^ 5 = 32); e.g. if we have image with shape 65x65 we will have
        # following shapes of features in encoder and decoder: 84, 42, 21, 10, 5 -> 5, 10, 20, 40, 80
        # and we will get an error trying to concat these features
        h, w = image.shape[2:]
        assert h % 32 == 0 and w % 32 == 0

        mask = batch["mask"]
        assert mask.ndim == 4

        # Check that mask values in between 0 and 1, NOT 0 and 255 for binary segmentation
        assert mask.max() <= 1.0 and mask.min() >= 0

        logits_mask = self.forward(image)

        # Predicted mask contains logits, and loss_fn param `from_logits` is set to True
        loss = self.loss_fn(logits_mask, mask)

        # Lets compute metrics for some threshold
        # first convert mask values to probabilities, then
        # apply thresholding
        prob_mask = logits_mask.sigmoid()
        pred_mask = (prob_mask > 0.5).float()

        # We will compute IoU metric by two ways
        #   1. dataset-wise
        #   2. image-wise
        # but for now we just compute true positive, false positive, false negative and
        # true negative 'pixels' for each image and class
        # these values will be aggregated in the end of an epoch
        tp, fp, fn, tn = smp.metrics.get_stats(
            pred_mask.long(), mask.long(), mode="binary"
        )
        return {
            "loss": loss,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
        }

    def shared_epoch_end(self, outputs, stage):
        # aggregate step metics
        tp = torch.cat([x["tp"] for x in outputs])
        fp = torch.cat([x["fp"] for x in outputs])
        fn = torch.cat([x["fn"] for x in outputs])
        tn = torch.cat([x["tn"] for x in outputs])

        # per image IoU means that we first calculate IoU score for each image
        # and then compute mean over these scores
        per_image_iou = smp.metrics.iou_score(
            tp, fp, fn, tn, reduction="micro-imagewise"
        )

        # dataset IoU means that we aggregate intersection and union over whole dataset
        # and then compute IoU score. The difference between dataset_iou and per_image_iou scores
        # in this particular case will not be much, however for dataset
        # with "empty" images (images without target class) a large gap could be observed.
        # Empty images influence a lot on per_image_iou and much less on dataset_iou.
        dataset_iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro")
        metrics = {
            f"{stage}_per_image_iou": per_image_iou,
            f"{stage}_dataset_iou": dataset_iou,
        }

        self.log_dict(metrics, prog_bar=True)

    def training_step(self, batch, batch_idx):
        train_loss_info = self.shared_step(batch, "train")
        # append the metics of each step to the
        self.training_step_outputs.append(train_loss_info)
        return train_loss_info

    def on_train_epoch_end(self):
        self.shared_epoch_end(self.training_step_outputs, "train")
        # empty set output list
        self.training_step_outputs.clear()
        return

    def validation_step(self, batch, batch_idx):
        valid_loss_info = self.shared_step(batch, "valid")
        self.validation_step_outputs.append(valid_loss_info)
        return valid_loss_info

    def on_validation_epoch_end(self):
        self.shared_epoch_end(self.validation_step_outputs, "valid")
        self.validation_step_outputs.clear()
        return

    def test_step(self, batch, batch_idx):
        test_loss_info = self.shared_step(batch, "test")
        self.test_step_outputs.append(test_loss_info)
        return test_loss_info

    def on_test_epoch_end(self):
        self.shared_epoch_end(self.test_step_outputs, "test")
        # empty set output list
        self.test_step_outputs.clear()
        return

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=2e-4)
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_MAX, eta_min=1e-5)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }
    


if train_for_ipc and not use_ipc:
    train_data, train_loader = get_dataset_and_loader(f"{data_dir}TR/im-good", f"{data_dir}TR/ipc-good", shuffle=True)
    valid_data, valid_loader = get_dataset_and_loader(f"{data_dir}VD/im-good", f"{data_dir}VD/ipc-good")
    test_data, test_loader = get_dataset_and_loader(f"{data_dir}TE/im-good", f"{data_dir}TE/ipc-good")
else:
    train_data, train_loader = get_dataset_and_loader(f"{data_dir}TR/im-good", f"{data_dir}TR/gt-good", shuffle=True)
    valid_data, valid_loader = get_dataset_and_loader(f"{data_dir}VD/im-good", f"{data_dir}VD/gt-good")
    test_data, test_loader = get_dataset_and_loader(f"{data_dir}TE/im-good", f"{data_dir}TE/gt-good")

unet_model = UnetModel.load_from_checkpoint(checkpoint_path, arch="Unet", encoder_name="resnet18", in_channels=3, out_classes=1)
if use_ipc:
    unet_model.double()

# check validation metrics
trainer = pl.Trainer(max_epochs=1, log_every_n_steps=1)
valid_metrics = trainer.validate(unet_model, dataloaders=valid_loader, verbose=False)
print(valid_metrics)

train_metrics = trainer.validate(unet_model, dataloaders=train_loader, verbose=False)
print(train_metrics)

batch = next(iter(train_loader))
with torch.no_grad():
    unet_model.eval()
    logits = unet_model(batch["image"])
pr_masks = logits.sigmoid()
for idx, (image, gt_mask, pr_mask) in enumerate(
    zip(batch["image"], batch["mask"], pr_masks)
):
    if idx <= 4:
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 3, 1)
        plt.imshow(image.numpy().transpose(1, 2, 0))
        plt.title("Image")
        plt.axis("off")

        plt.subplot(1, 3, 2)
        plt.imshow(gt_mask.numpy().squeeze())
        plt.title("Ground truth")
        plt.axis("off")

        plt.subplot(1, 3, 3)
        plt.imshow(pr_mask.numpy().squeeze())
        plt.title("Prediction")
        plt.axis("off")
        plt.savefig(f"imgs/img{idx}.png")
        plt.show()
    else:
        break


