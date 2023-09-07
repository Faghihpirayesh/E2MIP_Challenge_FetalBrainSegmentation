# modified from "https://github.com/Project-MONAI/tutorials/blob/main/2d_segmentation/torch/unet_training_array.py"
import os
import random
import sys
from glob import glob

import torch

import monai
from monai.data import decollate_batch, DataLoader, Dataset
from monai.inferers import SimpleInferer
from monai.metrics import DiceMetric
import monai.transforms as tr

import warnings

warnings.filterwarnings('ignore')


def train(args):
    monai.config.print_config()

    # define transforms for image and segmentation
    transformations = tr.Compose(
        [
            tr.LoadImaged(keys=["image", "label"]),
            tr.EnsureChannelFirstd(keys=["image", "label"]),
            tr.Spacingd(keys=["image", "label"], pixdim=(1.0, 1.0, -1.0), mode=("bilinear", "nearest")),
            tr.SqueezeDimd(keys=["image", "label"], dim=-1, update_meta=True),
            tr.NormalizeIntensityd(keys=["image"], nonzero=True),
            tr.Resized(keys=["image", "label"], spatial_size=(args.img_size, args.img_size)),
        ]
    )

    # load train data
    train_images = sorted(glob(os.path.join(args.training_data_path, "images/*.nii.gz")))
    train_labels = sorted(glob(os.path.join(args.training_data_path, "masks/*.nii.gz")))
    train_files = [{"image": image_name, "label": label_name} for
                   image_name, label_name in zip(train_images, train_labels)]
    train_ds = Dataset(data=train_files, transform=transformations)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=8)

    # load validation data
    validation_images = sorted(glob(os.path.join(args.validation_data_path, "images/*.nii.gz")))
    validation_labels = sorted(glob(os.path.join(args.validation_data_path, "masks/*.nii.gz")))
    validation_files = [{"image": image_name, "label": label_name} for
                        image_name, label_name in zip(validation_images, validation_labels)]
    val_ds = Dataset(data=validation_files, transform=transformations)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, num_workers=4)

    # define metrics
    dice_metric = DiceMetric(include_background=False, reduction="mean", get_not_nans=False)
    post_pred = tr.Compose([tr.AsDiscrete(argmax=True, to_onehot=args.num_classes)])
    post_label = tr.Compose([tr.AsDiscrete(to_onehot=args.num_classes)])

    # create UNet, DiceLoss and Adam optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = monai.networks.nets.UNet(
        spatial_dims=2,
        in_channels=1,
        out_channels=2,
        channels=(32, 64, 128, 256, 512),
        strides=(2, 2, 2, 2),
    ).to(device)
    loss_function = monai.losses.DiceCELoss(include_background=False,
                                            to_onehot_y=True,
                                            softmax=True,
                                            squared_pred=False,
                                            batch=True,
                                            smooth_nr=0.00001,
                                            smooth_dr=0.00001,
                                            lambda_dice=0.6,
                                            lambda_ce=0.4,
                                            )
    optimizer = torch.optim.Adam(model.parameters(), args.lr)

    # start a typical PyTorch training
    val_interval = 2
    best_metric = -1
    best_metric_epoch = -1
    epoch_loss_values = list()
    metric_values = list()
    for epoch in range(args.epochs):
        print("-" * 30)
        print(f"epoch {epoch + 1}/{args.epochs}")
        model.train()
        epoch_loss = 0
        step = 0
        for batch_data in train_loader:
            step += 1
            inputs, labels = (batch_data["image"].to(device),
                              batch_data["label"].to(device),
                              )
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            epoch_len = len(train_ds) // train_loader.batch_size
            print(f"{step}/{epoch_len}, train_loss: {loss.item():.4f}")

        epoch_loss /= step
        epoch_loss_values.append(epoch_loss)
        print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")

        if (epoch + 1) % val_interval == 0:
            model.eval()
            with torch.no_grad():
                for val_data in val_loader:
                    val_inputs, val_labels = (val_data["image"].to(device),
                                              val_data["label"].to(device),
                                              )
                    inferer = SimpleInferer()
                    val_outputs = inferer(val_inputs, model)
                    val_outputs = [post_pred(i) for i in decollate_batch(val_outputs)]
                    val_labels = [post_label(i) for i in decollate_batch(val_labels)]
                    # compute metric for current iteration
                    dice_metric(y_pred=val_outputs, y=val_labels)
                # aggregate the final mean dice result
                metric = dice_metric.aggregate().item()
                # reset the status for next validation round
                dice_metric.reset()
                metric_values.append(metric)
                if metric > best_metric:
                    best_metric = metric
                    best_metric_epoch = epoch + 1
                    torch.save(model.state_dict(), "best_metric_model_segmentation2d.pth")
                    print("saved new best metric model")
                print(
                    "current epoch: {} current mean dice: {:.4f} best mean dice: {:.4f} at epoch {}".format(
                        epoch + 1, metric, best_metric, best_metric_epoch
                    )
                )

    print(f"train completed, best_metric: {best_metric:.4f} at epoch: {best_metric_epoch}")
