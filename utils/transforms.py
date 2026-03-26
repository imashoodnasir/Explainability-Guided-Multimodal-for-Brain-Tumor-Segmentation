from __future__ import annotations
from typing import Sequence

from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    Orientationd,
    Spacingd,
    CropForegroundd,
    RandFlipd,
    RandRotate90d,
    RandScaleIntensityd,
    RandShiftIntensityd,
    NormalizeIntensityd,
    EnsureTyped,
    RandSpatialCropd,
    SpatialPadd,
)


def train_transforms(keys: Sequence[str], patch_size=(128, 128, 128), spacing=(1.0, 1.0, 1.0)):
    image_keys = [k for k in keys if k != "seg"]
    return Compose([
        LoadImaged(keys=keys),
        EnsureChannelFirstd(keys=keys),
        Orientationd(keys=keys, axcodes="RAS"),
        Spacingd(keys=keys, pixdim=spacing, mode=("bilinear", "bilinear", "bilinear", "bilinear", "nearest")),
        CropForegroundd(keys=keys, source_key="flair"),
        NormalizeIntensityd(keys=image_keys, nonzero=True, channel_wise=True),
        SpatialPadd(keys=keys, spatial_size=patch_size),
        RandSpatialCropd(keys=keys, roi_size=patch_size, random_size=False),
        RandFlipd(keys=keys, spatial_axis=0, prob=0.5),
        RandFlipd(keys=keys, spatial_axis=1, prob=0.5),
        RandFlipd(keys=keys, spatial_axis=2, prob=0.5),
        RandRotate90d(keys=keys, prob=0.5, max_k=3),
        RandScaleIntensityd(keys=image_keys, factors=0.1, prob=0.5),
        RandShiftIntensityd(keys=image_keys, offsets=0.1, prob=0.5),
        EnsureTyped(keys=keys),
    ])


def val_transforms(keys: Sequence[str], spacing=(1.0, 1.0, 1.0)):
    image_keys = [k for k in keys if k != "seg"]
    return Compose([
        LoadImaged(keys=keys),
        EnsureChannelFirstd(keys=keys),
        Orientationd(keys=keys, axcodes="RAS"),
        Spacingd(keys=keys, pixdim=spacing, mode=("bilinear", "bilinear", "bilinear", "bilinear", "nearest")),
        CropForegroundd(keys=keys, source_key="flair"),
        NormalizeIntensityd(keys=image_keys, nonzero=True, channel_wise=True),
        EnsureTyped(keys=keys),
    ])
