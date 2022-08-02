import argparse
import os
import requests
import numpy as np
from numpy import savez_compressed
import torch
import torch.nn as nn
from pdb import set_trace as st
from PIL import Image
from torchvision import transforms
import torch_xla
import torch_xla.core.xla_model as xm

import deeplab
from data_loader import CelebASegmentation
from utils import download_file
import cv2
from gray2color import gray2color
from matplotlib import pyplot as plt
import torchvision.models as models
from torch.utils.data import DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
import pandas as pd
import torch.utils.data as data


c_pallet = np.array([[[0, 0, 0],
                      [204, 0, 0],
                      [76, 153, 0],
                      [204, 204, 0],
                      [51, 51, 255],
                      [204, 0, 204],
                      [0, 255, 255],
                      [255, 204, 204],
                      [102, 51, 0],
                      [255, 0, 0],
                      [102, 204, 0],
                      [255, 255, 0],
                      [0, 0, 153],
                      [0, 0, 204],
                      [255, 51, 153],
                      [0, 204, 204],
                      [0, 51, 0],
                      [255, 153, 51],
                      [0, 204, 0]]], np.uint8) / 255

device = xm.xla_device()

def load_segmentation_model(model_name: str, cuda_rank: int, output_size: int = 2) -> torch.nn.Module:
    """
    Returns a ResNet model with pretrained weights using the model name.
    """

    # resolution = args.resolution
    # assert torch.cuda.is_available()
    torch.backends.cudnn.benchmark = True
    print(os.getcwd())
    model_fname = '/home/harsh/thesis_1_cuda/stylex/trained_classifiers/model.ckpt'
    # dataset_root = 'ffhq_aging{}x{}'.format(resolution,resolution)
    dataset_root = '../data/Kaggle_FFHQ_Resized_256px/flickrfaceshq-dataset-nvidia-resized-256px/resized/'
    assert os.path.isdir(dataset_root)
    # dataset = CelebASegmentation(dataset_root, crop_size=513)

    # construct loader
    # dataloader = DataLoader(dataset,
    #                         batch_size=1,
    #                         shuffle=False,
    #                         num_workers=1,)

    #     print(resnet_file_spec)

    #     if not os.path.isfile(resnet_file_spec['file_path']):
    #         print('Downloading backbone Resnet Model parameters')
    #         with requests.Session() as session:
    #             download_file(session, resnet_file_spec)

    #         print('Done!')

    # model = getattr(deeplab, 'resnet101')(
    #     pretrained=True,
    #     num_classes=len(dataset.CLASSES),
    #     num_groups=32,
    #     weight_std=True,
    #     beta=False)
    #
    # model = model.to(device)
    # model.eval()
    # # if not os.path.isfile(deeplab_file_spec['file_path']):
    # #     print('Downloading DeeplabV3 Model parameters')
    # #     with requests.Session() as session:
    # #         download_file(session, deeplab_file_spec)
    # #
    # #     print('Done!')
    #
    # checkpoint = torch.load(model_fname, map_location=torch.device("cpu"))
    # state_dict = {k[7:]: v for k, v in checkpoint['state_dict'].items() if 'tracked' not in k}
    # # print("state dict",state_dict)
    # model.load_state_dict(state_dict)

    model = models.segmentation.deeplabv3_resnet50(pretrained=False, num_classes=2)
    print("model path:", model_fname)
    checkpoint = torch.load(model_fname, map_location=torch.device("cpu"))
    state_dict = {}
    for key, val in checkpoint['state_dict'].items():
        key = key.replace("model.", "")
        state_dict[key] = val
    print("Loading checkpoint into SegFace2:")
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()

    return model

resnet_file_spec = dict(file_url='https://drive.google.com/uc?id=1oRGgrI4KNdefbWVpw0rRkEP1gbJIRokM', file_path='deeplab_model/R-101-GN-WS.pth.tar',
                        file_size=178260167, file_md5='aa48cc3d3ba3b7ac357c1489b169eb32')
deeplab_file_spec = dict(file_url='https://drive.google.com/uc?id=1w2XjDywFr2NjuUWaLQDRktH7VwIfuNlY', file_path='deeplab_model/deeplab_model.pth',
                         file_size=464446305, file_md5='8e8345b1b9d95e02780f9bed76cc0293')

class SegmentationModel():
    def __init__(self, model_name: str, cuda_rank: int, output_size: int = 2, image_size=32, normalize=True):
        self.model = load_segmentation_model(model_name, cuda_rank, output_size)

    def get_segmentation_logits(self,images) -> torch.Tensor:
        return self.model(images)['out']


class Face_dataset(data.Dataset):
    def __init__(self,
                 data_dir: str,
                 images_folder: str,
                 csv_file: str,
                 label_mask_folder: str = None,
                 augmentations=None):
        """
        Args:
            train: whether to use the training or the validation split
            data_dir: directory containing the data
            train_images_folder: train images folder name
            label_mask_folder : label mask folder name
        """
        self.data_dir = data_dir
        self.train_csv_file = csv_file
        self.train_data = pd.read_csv(csv_file)
        self.images_list = self.train_data["images"]
        self.labels_list = self.train_data["labels"]
        self.train_images_folder = images_folder
        self.label_mask_folder = label_mask_folder
        self.augmentations = augmentations

    def __getitem__(self, item: int) -> dict:
        """
        Loads and Returns a single sample

        Args:
            item: index specifying which item to load

        Returns:
            dict: the loaded sample
        """
        img = self.images_list[item].strip()
        img_path = os.path.join(self.data_dir, self.train_images_folder, img)
        img = np.asarray(Image.open(img_path))
        # label_mask = self.labels_list[item].strip()
        # label_mask_path = os.path.join(self.data_dir, self.label_mask_folder, label_mask)

        # label_mask_npz = np.load(label_mask_path)
        # label_mask_np = label_mask_npz[list(label_mask_npz.keys())[0]][:, :, 0]
        # label_mask_np[label_mask_np == 204] = 1
        if self.augmentations is not None:
            augmented = self.augmentations(image=img)
            img = augmented['image']
            # label_mask_np = augmented['mask']
            return {'data': img,
                    'img_path': img_path,
                    }
        else:
            return {'data': torch.from_numpy(img),
                    # 'label': torch.from_numpy(label_mask_np).long(),
                    'img_path': img_path,
                    # 'label_mask_path': label_mask_path
                    }

    def __len__(self) -> int:
        """
        Adds a length to the dataset

        Returns:
            int: dataset's length
        """
        return len(self.train_data)

