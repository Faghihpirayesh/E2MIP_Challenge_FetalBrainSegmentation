# modified from "https://github.com/Project-MONAI/tutorials/blob/main/2d_segmentation/torch/unet_evaluation_array.py"
import os
from glob import glob
import numpy as np

import torch

from monai import config
import monai.transforms as tr
from monai.data import decollate_batch, Dataset, DataLoader
from monai.inferers import SliceInferer
from monai.metrics import DiceMetric
from monai.networks.nets import UNet
from monai.transforms import Activations, AsDiscrete, Compose, LoadImage, SaveImage, ScaleIntensity, \
    ResizeWithPadOrCrop, MapTransform, SaveImaged

import warnings

warnings.filterwarnings('ignore')


class SliceWiseNormalizeIntensityd(MapTransform):
    def __init__(self, keys, subtrahend=0.0, divisor=None, nonzero=True):
        super().__init__(keys)
        self.subtrahend = subtrahend
        self.divisor = divisor
        self.nonzero = nonzero

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            image = d[key]
            for i in range(image.shape[-1]):
                slice_ = image[..., i]
                if self.nonzero:
                    mask = slice_ > 0
                    if np.any(mask):
                        if self.subtrahend is None:
                            slice_[mask] = slice_[mask] - slice_[mask].mean()
                        else:
                            slice_[mask] = slice_[mask] - self.subtrahend

                        if self.divisor is None:
                            slice_[mask] /= slice_[mask].std()
                        else:
                            slice_[mask] /= self.divisor

                else:
                    if self.subtrahend is None:
                        slice_ = slice_ - slice_.mean()
                    else:
                        slice_ = slice_ - self.subtrahend

                    if self.divisor is None:
                        slice_ /= slice_.std()
                    else:
                        slice_ /= self.divisor

                image[..., i] = slice_
            d[key] = image
        return d


def test(args):
    config.print_config()

    test_transforms = tr.Compose(
        [
            tr.LoadImaged(keys=["image"]),
            tr.EnsureChannelFirstd(keys=["image"]),
            tr.Spacingd(keys="image", pixdim=(1.0, 1.0, -1.0), mode="bilinear"),
            SliceWiseNormalizeIntensityd(keys=["image"], nonzero=True),
            tr.ResizeWithPadOrCropd(keys="image", spatial_size=(args.img_size, args.img_size, -1)),
        ]
    )

    # load data
    test_images = sorted(glob(os.path.join(args.testing_data_path, "*.nii.gz")))
    test_files = [{"image": image_name} for image_name in test_images]

    test_dataset = Dataset(data=test_files, transform=test_transforms)
    test_dataloader = DataLoader(test_dataset,
                                 batch_size=1,
                                 num_workers=0)

    post_transforms = tr.Compose(
        [
            tr.Invertd(
                keys="pred",
                transform=test_transforms,
                orig_keys="image",
                meta_keys="pred_meta_dict",
                orig_meta_keys="image_meta_dict",
                meta_key_postfix="meta_dict",
                nearest_interp=False,
                to_tensor=True,
            ),
            tr.Activationsd(keys="pred", softmax=True),
            tr.AsDiscreted(keys="pred", argmax=True, to_onehot=None),
            tr.RemoveSmallObjectsd(keys="pred", min_size=50, connectivity=1),
            SaveImaged(keys="pred", meta_keys="pred_meta_dict", output_dir=args.test_save_path,
                       separate_folder=False, output_postfix="maskpred", resample=False),
        ]
    )

    device = args.device
    model = UNet(
        spatial_dims=2,
        in_channels=1,
        out_channels=2,
        channels=(32, 64, 128, 256, 512),
        strides=(2, 2, 2, 2),
    ).to(device)
    model.load_state_dict(torch.load(args.model_path))

    inferer = SliceInferer(roi_size=(256, 256),
                           spatial_dim=2,
                           progress=False)

    with torch.no_grad():
        model.eval()
        for i, test_data in enumerate(test_dataloader):
            test_inputs = test_data["image"].to(device)

            test_data["pred"] = inferer(test_inputs, model)
            test_data = [post_transforms(i) for i in decollate_batch(test_data)]
