import os

import torch
from torch import nn
import torch.nn.functional as F
from torchvision.transforms import transforms
from torchvision.transforms.functional import resize


def load_resnet_classifier(model_name: str, cuda_rank: int, output_size: int = 2) -> torch.nn.Module:
    """
    Returns a ResNet model with pretrained weights using the model name.
    """

    # Decide which device to put it on (dirty because we should use cuda rank
    device = torch.device(f"cuda:{cuda_rank}") if torch.cuda.is_available() else torch.device("cpu")

    # Load the resnet classifier
    model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=False).to(device)

    # Change the final layer for classification
    model.fc = nn.Linear(512, 2).to(device)

    # Load the weights from the checkpoint.
    model.load_state_dict(torch.load("./stylex/trained_classifiers/resnet-18-64px-gender.pt", map_location=device))

    return model


class ResNet():
    def __init__(self, model_name: str, cuda_rank: int, output_size: int = 2, image_size=32, normalize=True):
        self.model = load_resnet_classifier(model_name, cuda_rank, output_size)

        self.resnet_dim = 224
        self.image_size = image_size

        # Image transformation
        self.image_transform = transforms.Compose([
            transforms.Resize(self.resnet_dim),  # Putting back the resize because it really matters for faces accuracy.
            transforms.ToTensor()
        ])

        self.tensor_transform = transforms.Compose([
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        self.normalize = normalize

        # Freeze all layers
        for param in self.model.parameters():
            param.requires_grad = False

        # Put the model in evaluation mode.
        self.model.eval()

    def classify_images(self, images) -> torch.Tensor:
        """
        Classifies a batch of images using the given model.
        """
        if isinstance(images, torch.Tensor):
            preprocessed_images = resize(images, [self.resnet_dim, self.resnet_dim])
        else:
            preprocessed_images = self.image_transform(images)

        # I trained on MNIST without normalizing, but it still worked,
        # so I made normalization optional
        if self.normalize:
            preprocessed_images = self.tensor_transform(preprocessed_images)

        # Classify the images.
        return self.model(preprocessed_images)
