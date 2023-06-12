# modified from "https://github.com/Project-MONAI/tutorials/blob/main/2d_segmentation/torch/unet_evaluation_array.py"
import logging
import os
import sys
from glob import glob

import torch

from monai import config
from monai.data import ArrayDataset, decollate_batch, DataLoader
from monai.inferers import SliceInferer
from monai.metrics import DiceMetric
from monai.networks.nets import UNet
from monai.transforms import Activations, AsDiscrete, Compose, LoadImage, SaveImage, ScaleIntensity, ResizeWithPadOrCrop

import warnings

warnings.filterwarnings('ignore')


def test(args):
    config.print_config()
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)

    # load data
    images = sorted(glob(os.path.join(args.testing_data_path, "images/*.nii.gz")))
    segs = sorted(glob(os.path.join(args.testing_data_path, "masks/*.nii.gz")))

    # define transforms for image and segmentation
    imtrans = Compose(
        [
            LoadImage(image_only=True, ensure_channel_first=True),
            ScaleIntensity(),
            ResizeWithPadOrCrop(spatial_size=(args.img_size, args.img_size, -1))
        ]
    )
    segtrans = Compose(
        [
            LoadImage(image_only=True, ensure_channel_first=True),
            ScaleIntensity(),
            ResizeWithPadOrCrop(spatial_size=(args.img_size, args.img_size, -1))
        ]
    )

    test_ds = ArrayDataset(images, imtrans, segs, segtrans)

    # sliding window inference for one image at every iteration
    test_loader = DataLoader(test_ds, batch_size=1, num_workers=1)
    dice_metric = DiceMetric(include_background=True, reduction="mean", get_not_nans=False)
    post_trans = Compose([Activations(softmax=True), AsDiscrete(argmax=True, threshold=0.5)])

    saver = SaveImage(output_dir="./data/testing_data_prediction", output_ext=".nii.gz", separate_folder=False,
                      output_postfix="mask")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet(
        spatial_dims=2,
        in_channels=1,
        out_channels=2,
        channels=(32, 64, 128, 256, 512),
        strides=(2, 2, 2, 2),
    ).to(device)

    model.load_state_dict(torch.load(args.model_path))
    model.eval()
    with torch.no_grad():
        for test_data in test_loader:
            test_images, test_labels = test_data[0].to(device), test_data[1].to(device)

            infer = SliceInferer(roi_size=(args.img_size, args.img_size), sw_batch_size=1, cval=-1, progress=True)
            test_outputs = infer(test_images, model)
            test_outputs = [post_trans(i) for i in decollate_batch(test_outputs)]

            # compute metric for current iteration
            dice_metric(y_pred=test_outputs, y=test_labels)
            for val_output in test_outputs:
                saver(val_output)
        # aggregate the final mean dice result
        print("evaluation metric:", dice_metric.aggregate().item())
        # reset the status
        dice_metric.reset()
