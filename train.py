# modified from "https://github.com/Project-MONAI/tutorials/blob/main/2d_segmentation/torch/unet_training_array.py"
import logging
import os
import random
import sys
from glob import glob

import torch
from torch.utils.tensorboard import SummaryWriter

import monai
from monai.data import ArrayDataset, decollate_batch, DataLoader
from monai.inferers import SimpleInferer
from monai.metrics import DiceMetric
from monai.transforms import (
    Activations,
    AsDiscrete,
    Compose,
    LoadImage,
    RandSpatialCrop,
    ScaleIntensity,
)
from monai.visualize import plot_2d_or_3d_image

import warnings
warnings.filterwarnings('ignore')


def train(args):
    monai.config.print_config()
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)

    # load data
    images = sorted(glob(os.path.join(args.training_data_path, "images/*.nii.gz")))
    segs = sorted(glob(os.path.join(args.training_data_path, "masks/*.nii.gz")))

    # define transforms for image and segmentation
    train_imtrans = Compose(
        [
            LoadImage(image_only=True, ensure_channel_first=True),
            ScaleIntensity(),
            RandSpatialCrop((args.img_size, args.img_size), random_size=False)
        ]
    )
    train_segtrans = Compose(
        [
            LoadImage(image_only=True, ensure_channel_first=True),
            ScaleIntensity(),
            RandSpatialCrop((args.img_size, args.img_size), random_size=False)
        ]
    )

    val_imtrans = Compose(
        [
            LoadImage(image_only=True, ensure_channel_first=True),
            ScaleIntensity(),
            RandSpatialCrop((args.img_size, args.img_size), random_size=False)
        ]
    )
    val_segtrans = Compose(
        [
            LoadImage(image_only=True, ensure_channel_first=True),
            ScaleIntensity(),
            RandSpatialCrop((args.img_size, args.img_size), random_size=False)
        ]
    )

    # define array dataset, data loader
    check_ds = ArrayDataset(images, train_imtrans, segs, train_segtrans)
    check_loader = DataLoader(check_ds, batch_size=args.batch_size, num_workers=8)
    im, seg = monai.utils.misc.first(check_loader)
    print(im.shape, seg.shape)

    # shuffle the lists with same order
    zipped = list(zip(images, segs))
    random.shuffle(zipped)
    images, segs = zip(*zipped)

    # create a training data loader
    n_split = int(0.8 * len(images))
    train_ds = ArrayDataset(images[:n_split], train_imtrans, segs[:n_split], train_segtrans)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=8)

    # create a validation data loader
    val_ds = ArrayDataset(images[-n_split:], val_imtrans, segs[-n_split:], val_segtrans)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, num_workers=4)

    # define metrics
    dice_metric = DiceMetric(include_background=True, reduction="mean", get_not_nans=False)
    post_trans = Compose([Activations(softmax=True), AsDiscrete(argmax=True, threshold=0.5)])

    # create UNet, DiceLoss and Adam optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = monai.networks.nets.UNet(
        spatial_dims=2,
        in_channels=1,
        out_channels=2,
        channels=(32, 64, 128, 256, 512),
        strides=(2, 2, 2, 2),
    ).to(device)
    loss_function = monai.losses.DiceLoss(softmax=True, to_onehot_y=True)
    optimizer = torch.optim.Adam(model.parameters(), args.lr)

    # start a typical PyTorch training
    val_interval = 2
    best_metric = -1
    best_metric_epoch = -1
    epoch_loss_values = list()
    metric_values = list()
    writer = SummaryWriter()
    for epoch in range(args.epochs):
        print("-" * 30)
        print(f"epoch {epoch + 1}/{args.epochs}")
        model.train()
        epoch_loss = 0
        step = 0
        for batch_data in train_loader:
            step += 1
            inputs, labels = batch_data[0].to(device), batch_data[1].to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            epoch_len = len(train_ds) // train_loader.batch_size
            print(f"{step}/{epoch_len}, train_loss: {loss.item():.4f}")
            writer.add_scalar("train_loss", loss.item(), epoch_len * epoch + step)
        epoch_loss /= step
        epoch_loss_values.append(epoch_loss)
        print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")

        if (epoch + 1) % val_interval == 0:
            model.eval()
            with torch.no_grad():
                val_images = None
                val_labels = None
                val_outputs = None
                for val_data in val_loader:
                    val_images, val_labels = val_data[0].to(device), val_data[1].to(device)
                    inferer = SimpleInferer()
                    val_outputs = inferer(val_images, model)
                    val_outputs = [post_trans(i) for i in decollate_batch(val_outputs)]
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
                    torch.save(model.state_dict(), "best_metric_model_segmentation2d_array.pth")
                    print("saved new best metric model")
                print(
                    "current epoch: {} current mean dice: {:.4f} best mean dice: {:.4f} at epoch {}".format(
                        epoch + 1, metric, best_metric, best_metric_epoch
                    )
                )
                writer.add_scalar("val_mean_dice", metric, epoch + 1)
                # plot the last model output as GIF image in TensorBoard with the corresponding image and label
                plot_2d_or_3d_image(val_images, epoch + 1, writer, index=0, tag="image")
                plot_2d_or_3d_image(val_labels, epoch + 1, writer, index=0, tag="label")
                plot_2d_or_3d_image(val_outputs, epoch + 1, writer, index=0, tag="output")

    print(f"train completed, best_metric: {best_metric:.4f} at epoch: {best_metric_epoch}")
    writer.close()
