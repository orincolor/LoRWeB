# Copyright (C) 2026 NVIDIA Corporation.  All rights reserved.
#
# This work is licensed under the LICENSE file
# located at the root directory.


import copy
import json
import os
import random
import traceback
from functools import lru_cache
from typing import List, TYPE_CHECKING

import cv2
import numpy as np
import torch
from PIL import Image
from PIL.ImageOps import exif_transpose
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from tqdm import tqdm
import albumentations as A

from toolkit.buckets import get_bucket_for_image_size
from toolkit.config_modules import DatasetConfig, preprocess_dataset_raw_config
from toolkit.dataloader_mixins import CaptionMixin, BucketsMixin, Augments
from toolkit.data_transfer_object.data_loader import FileItemDTO, DataLoaderBatchDTO
from toolkit.print import print_acc

import platform

def is_native_windows():
    return platform.system() == "Windows" and platform.release() != "2"

if TYPE_CHECKING:
    from toolkit.stable_diffusion_model import StableDiffusion


image_extensions = ['.jpg', '.jpeg', '.png', '.webp']
video_extensions = ['.mp4', '.avi', '.mov', '.webm', '.mkv', '.wmv', '.m4v', '.flv']


class RescaleTransform:
    """Transform to rescale images to the range [-1, 1]."""

    def __call__(self, image):
        return image * 2 - 1


class NormalizeSDXLTransform:
    """
    Transforms the range from 0 to 1 to SDXL mean and std per channel based on avgs over thousands of images

    Mean: tensor([ 0.0002, -0.1034, -0.1879])
    Standard Deviation: tensor([0.5436, 0.5116, 0.5033])
    """

    def __call__(self, image):
        return transforms.Normalize(
            mean=[0.0002, -0.1034, -0.1879],
            std=[0.5436, 0.5116, 0.5033],
        )(image)


class NormalizeSD15Transform:
    """
    Transforms the range from 0 to 1 to SDXL mean and std per channel based on avgs over thousands of images

    Mean: tensor([-0.1600, -0.2450, -0.3227])
    Standard Deviation: tensor([0.5319, 0.4997, 0.5139])

    """

    def __call__(self, image):
        return transforms.Normalize(
            mean=[-0.1600, -0.2450, -0.3227],
            std=[0.5319, 0.4997, 0.5139],
        )(image)


class AiToolkitDataset(BucketsMixin, CaptionMixin, Dataset):

    def __init__(
            self,
            dataset_config: 'DatasetConfig',
            batch_size=1,
            sd: 'StableDiffusion' = None,
    ):
        self.dataset_config = dataset_config
        # update bucket divisibility
        self.dataset_config.bucket_tolerance = sd.get_bucket_divisibility()
        self.is_video = dataset_config.num_frames > 1
        super().__init__()
        folder_path = dataset_config.folder_path
        self.dataset_path = dataset_config.dataset_path
        if self.dataset_path is None:
            self.dataset_path = folder_path
        self.folder_size_json = folder_path + '_sizes.json'
        self.dataset_size_json = self.dataset_path+  '_sizes.json'
        if os.path.exists(self.folder_size_json):
            print('Found folder size json, loading...')
            with open(self.folder_size_json, 'r') as f:
                self.folder_size_json = json.load(f)
        else:
            print('No folder size json found, Longer...')
            self.folder_size_json = None
            
        if os.path.exists(self.dataset_size_json):
            print('Found dataset size json, loading...')
            with open(self.dataset_size_json, 'r') as f:
                self.dataset_size_json = json.load(f)
        else:
            print('No dataset size json found, Longer...')
            self.dataset_size_json = None

        self.enable_relation_captions = None
        if dataset_config.enable_relation_captions:
            with open(os.path.join(os.path.dirname(folder_path), 'analogy_metadata.json'), 'r') as f:
                enable_relation_captions = json.load(f)
                #organize by key
                self.enable_relation_captions = {i['analogy_id']: i for i in enable_relation_captions}

        self.is_generating_controls = len(dataset_config.controls) > 0
        self.epoch_num = 0

        self.sd = sd

        self.caption_type = dataset_config.caption_ext
        self.default_caption = dataset_config.default_caption
        self.random_scale = dataset_config.random_scale
        self.scale = dataset_config.scale
        self.batch_size = batch_size
        # we always random crop if random scale is enabled
        self.random_crop = self.random_scale if self.random_scale else dataset_config.random_crop
        self.resolution = dataset_config.resolution
        self.caption_dict = None
        self.file_list: List['FileItemDTO'] = []

        # check if dataset_path is a folder or json
        if os.path.isdir(self.dataset_path):
            extensions = image_extensions
            if self.is_video:
                # only look for videos
                extensions = video_extensions
            file_list = [os.path.join(root, file) for root, _, files in os.walk(self.dataset_path) for file in files if file.lower().endswith(tuple(extensions))]
        else:
            # assume json
            with open(self.dataset_path, 'r') as f:
                self.caption_dict = json.load(f)
                # keys are file paths
                file_list = list(self.caption_dict.keys())

        # remove items in the _controls_ folder
        file_list = [x for x in file_list if not os.path.basename(os.path.dirname(x)) == "_controls"]

        if self.dataset_config.num_repeats > 1:
            # repeat the list
            file_list = file_list * self.dataset_config.num_repeats

        if self.dataset_config.standardize_images:
            if self.sd.is_xl or self.sd.is_vega or self.sd.is_ssd:
                NormalizeMethod = NormalizeSDXLTransform
            else:
                NormalizeMethod = NormalizeSD15Transform

            self.transform = transforms.Compose([
                transforms.ToTensor(),
                RescaleTransform(),
                NormalizeMethod(),
            ])
        else:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                RescaleTransform(),
            ])

        # this might take a while
        print_acc(f"Dataset: {self.dataset_path}")
        if self.is_video:
            print_acc(f"  -  Preprocessing video dimensions")
        else:
            print_acc(f"  -  Preprocessing image dimensions")
        dataset_folder = self.dataset_path
        if not os.path.isdir(self.dataset_path):
            dataset_folder = os.path.dirname(dataset_folder)

        dataset_size_file = os.path.join(dataset_folder, '.aitk_size.json')
        dataloader_version = "0.1.2"
        if os.path.exists(dataset_size_file):
            try:
                with open(dataset_size_file, 'r') as f:
                    self.size_database = json.load(f)

                if "__version__" not in self.size_database or self.size_database["__version__"] != dataloader_version:
                    print_acc("Upgrading size database to new version")
                    # old version, delete and recreate
                    self.size_database = {}
            except Exception as e:
                print_acc(f"Error loading size database: {dataset_size_file}")
                print_acc(e)
                self.size_database = {}
        else:
            self.size_database = {}

        self.size_database["__version__"] = dataloader_version

        bad_count = 0
        for file_i, file in tqdm(enumerate(file_list), desc="creating FileItemDTOs", total=len(file_list)):
            try:
                filedto_kwargs = {}
                if self.dataset_size_json is not None:
                    filedto_kwargs["filesize"] = self.dataset_size_json[os.path.basename(file)]
                if self.folder_size_json is not None:
                    if 'control' in file:
                        raise NotImplementedError("aegaegaeg")

                file_item = FileItemDTO(
                    sd=self.sd,
                    path=file,
                    dataset_config=dataset_config,
                    dataloader_transforms=self.transform,
                    size_database=self.size_database,
                    dataset_root=dataset_folder,
                    **filedto_kwargs
                )
                self.file_list.append(file_item)
            except Exception as e:
                print_acc(traceback.format_exc())
                if self.is_video:
                    print_acc(f"Error processing video: {file}")
                else:
                    print_acc(f"Error processing image: {file}")
                print_acc(e)
                bad_count += 1

        # save the size database
        with open(dataset_size_file, 'w') as f:
            json.dump(self.size_database, f)

        if self.is_video:
            print_acc(f"  -  Found {len(self.file_list)} videos")
            assert len(self.file_list) > 0, f"no videos found in {self.dataset_path}"
        else:
            print_acc(f"  -  Found {len(self.file_list)} images")
            assert len(self.file_list) > 0, f"no images found in {self.dataset_path}"

        # handle x axis flips
        if self.dataset_config.flip_x:
            print_acc("  -  adding x axis flips")
            current_file_list = [x for x in self.file_list]
            for file_item in current_file_list:
                # create a copy that is flipped on the x axis
                new_file_item = copy.deepcopy(file_item)
                new_file_item.flip_x = True
                self.file_list.append(new_file_item)

        # handle y axis flips
        if self.dataset_config.flip_y:
            print_acc("  -  adding y axis flips")
            current_file_list = [x for x in self.file_list]
            for file_item in current_file_list:
                # create a copy that is flipped on the y axis
                new_file_item = copy.deepcopy(file_item)
                new_file_item.flip_y = True
                self.file_list.append(new_file_item)

        if self.dataset_config.flip_x or self.dataset_config.flip_y:
            if self.is_video:
                print_acc(f"  -  Found {len(self.file_list)} videos after adding flips")
            else:
                print_acc(f"  -  Found {len(self.file_list)} images after adding flips")

        self.setup_epoch()

    def setup_epoch(self):
        if self.epoch_num == 0:
            # initial setup
            # do not call for now
            if self.dataset_config.buckets:
                # setup buckets
                self.setup_buckets()
            if self.is_generating_controls:
                # always do this last
                self.setup_controls()
        else:
            if self.dataset_config.poi is not None:
                # handle cropping to a specific point of interest
                # setup buckets every epoch
                self.setup_buckets(quiet=True)
        self.epoch_num += 1

    def __len__(self):
        if self.dataset_config.buckets:
            return len(self.batch_indices)
        return len(self.file_list)

    def _get_single_item(self, index) -> 'FileItemDTO':
        file_item: 'FileItemDTO' = copy.deepcopy(self.file_list[index])
        file_item.load_and_process_image(self.transform)
        file_item.load_caption(self.caption_dict, enable_relation_captions=self.enable_relation_captions)
        return file_item

    def __getitem__(self, item):
        if self.dataset_config.buckets:
            # for buckets we collate ourselves for now
            # todo allow a scheduler to dynamically make buckets
            # we collate ourselves
            if len(self.batch_indices) - 1 < item:
                # tried everything to solve this. No way to reset length when redoing things. Pick another index
                item = random.randint(0, len(self.batch_indices) - 1)
            idx_list = self.batch_indices[item]
            return [self._get_single_item(idx) for idx in idx_list]
        else:
            # Dataloader is batching
            return self._get_single_item(item)


def get_dataloader_from_datasets(
        dataset_options,
        batch_size=1,
        sd: 'StableDiffusion' = None,
) -> DataLoader:
    if dataset_options is None or len(dataset_options) == 0:
        return None

    datasets = []
    has_buckets = False

    dataset_config_list = []
    # preprocess them all
    for dataset_option in dataset_options:
        if isinstance(dataset_option, DatasetConfig):
            dataset_config_list.append(dataset_option)
        else:
            # preprocess raw data
            split_configs = preprocess_dataset_raw_config([dataset_option])
            for x in split_configs:
                dataset_config_list.append(DatasetConfig(**x))

    for config in dataset_config_list:

        if config.type == 'image':
            dataset = AiToolkitDataset(config, batch_size=batch_size, sd=sd)
            datasets.append(dataset)
            if config.buckets:
                has_buckets = True
        else:
            raise ValueError(f"invalid dataset type: {config.type}")

    concatenated_dataset = ConcatDataset(datasets)

    # todo build scheduler that can get buckets from all datasets that match
    # todo and evenly distribute reg images

    def dto_collation(batch: List['FileItemDTO']):
        # create DTO batch
        batch = DataLoaderBatchDTO(
            file_items=batch
        )
        return batch

    # check if is caching latents

    dataloader_kwargs = {}

    if is_native_windows():
        dataloader_kwargs['num_workers'] = 0
    else:
        dataloader_kwargs['num_workers'] = dataset_config_list[0].num_workers
        dataloader_kwargs['prefetch_factor'] = dataset_config_list[0].prefetch_factor

    if has_buckets:
        # make sure they all have buckets
        for dataset in datasets:
            assert dataset.dataset_config.buckets, f"buckets not found on dataset {dataset.dataset_config.folder_path}, you either need all buckets or none"

        data_loader = DataLoader(
            concatenated_dataset,
            batch_size=None,  # we batch in the datasets for now
            drop_last=False,
            shuffle=True,
            collate_fn=dto_collation,  # Use the custom collate function
            **dataloader_kwargs
        )
    else:
        data_loader = DataLoader(
            concatenated_dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=dto_collation,
            **dataloader_kwargs
        )
    return data_loader


def trigger_dataloader_setup_epoch(dataloader: DataLoader):
    # hacky but needed because of different types of datasets and dataloaders
    dataloader.len = None
    if isinstance(dataloader.dataset, list):
        for dataset in dataloader.dataset:
            if hasattr(dataset, 'datasets'):
                for sub_dataset in dataset.datasets:
                    if hasattr(sub_dataset, 'setup_epoch'):
                        sub_dataset.setup_epoch()
                        sub_dataset.len = None
            elif hasattr(dataset, 'setup_epoch'):
                dataset.setup_epoch()
                dataset.len = None
    elif hasattr(dataloader.dataset, 'setup_epoch'):
        dataloader.dataset.setup_epoch()
        dataloader.dataset.len = None
    elif hasattr(dataloader.dataset, 'datasets'):
        dataloader.dataset.len = None
        for sub_dataset in dataloader.dataset.datasets:
            if hasattr(sub_dataset, 'setup_epoch'):
                sub_dataset.setup_epoch()
                sub_dataset.len = None

def get_dataloader_datasets(dataloader: DataLoader):
    # hacky but needed because of different types of datasets and dataloaders
    if isinstance(dataloader.dataset, list):
        datasets = []
        for dataset in dataloader.dataset:
            if hasattr(dataset, 'datasets'):
                for sub_dataset in dataset.datasets:
                    datasets.append(sub_dataset)
            else:
                datasets.append(dataset)
        return datasets
    elif hasattr(dataloader.dataset, 'datasets'):
        return dataloader.dataset.datasets
    else:
        return [dataloader.dataset]
