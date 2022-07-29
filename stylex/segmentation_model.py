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
    model_fname = './trained_classifiers/deeplab_model.pth'
    # dataset_root = 'ffhq_aging{}x{}'.format(resolution,resolution)
    dataset_root = '../data/Kaggle_FFHQ_Resized_256px/flickrfaceshq-dataset-nvidia-resized-256px/resized_sub/'
    assert os.path.isdir(dataset_root)
    dataset = CelebASegmentation(dataset_root, crop_size=513)

    #     print(resnet_file_spec)

    #     if not os.path.isfile(resnet_file_spec['file_path']):
    #         print('Downloading backbone Resnet Model parameters')
    #         with requests.Session() as session:
    #             download_file(session, resnet_file_spec)

    #         print('Done!')

    model = getattr(deeplab, 'resnet101')(
        pretrained=True,
        num_classes=len(dataset.CLASSES),
        num_groups=32,
        weight_std=True,
        beta=False)

    model = model.to(device)
    model.eval()
    # if not os.path.isfile(deeplab_file_spec['file_path']):
    #     print('Downloading DeeplabV3 Model parameters')
    #     with requests.Session() as session:
    #         download_file(session, deeplab_file_spec)
    #
    #     print('Done!')

    checkpoint = torch.load(model_fname, map_location=torch.device("cpu"))
    state_dict = {k[7:]: v for k, v in checkpoint['state_dict'].items() if 'tracked' not in k}
    # print("state dict",state_dict)
    model.load_state_dict(state_dict)

    return model

resnet_file_spec = dict(file_url='https://drive.google.com/uc?id=1oRGgrI4KNdefbWVpw0rRkEP1gbJIRokM', file_path='deeplab_model/R-101-GN-WS.pth.tar',
                        file_size=178260167, file_md5='aa48cc3d3ba3b7ac357c1489b169eb32')
deeplab_file_spec = dict(file_url='https://drive.google.com/uc?id=1w2XjDywFr2NjuUWaLQDRktH7VwIfuNlY', file_path='deeplab_model/deeplab_model.pth',
                         file_size=464446305, file_md5='8e8345b1b9d95e02780f9bed76cc0293')

class SegmentationModel():
    def __init__(self, model_name: str, cuda_rank: int, output_size: int = 2, image_size=32, normalize=True):
        self.model = load_segmentation_model(model_name, cuda_rank, output_size)

    def get_segmentation_logits(self,images) -> torch.Tensor:
        return self.model(images)

