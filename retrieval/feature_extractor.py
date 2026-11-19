
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
import numpy as np

from PIL import Image


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

IMAGE_SIZE = 224

transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


class DenseNet121FeatureExtractor:

    def __init__(self, checkpoint_path):

        self.model = models.densenet121(weights=None)
        self.model.classifier = nn.Linear(1024, 5)

        checkpoint = torch.load(
            checkpoint_path,
            map_location=DEVICE
        )

        if isinstance(checkpoint, dict):
            for key in ["state_dict", "model_state_dict", "model"]:
                if key in checkpoint and isinstance(checkpoint[key], dict):
                    checkpoint = checkpoint[key]
                    break

        checkpoint = {
            key.replace("module.", "", 1): value
            for key, value in checkpoint.items()
        }

        self.model.load_state_dict(checkpoint, strict=True)
        self.model.to(DEVICE)
        self.model.eval()

    @torch.inference_mode()
    def extract_batch(self, images):

        images = images.to(DEVICE)

        features = self.model.features(images)
        features = F.relu(features, inplace=False)

        embeddings = F.adaptive_avg_pool2d(features, (1, 1))
        embeddings = torch.flatten(embeddings, 1)

        embeddings = F.normalize(embeddings, p=2, dim=1)

        return embeddings.cpu().numpy().astype("float32")

    def extract_image(self, image_path):

        image = Image.open(image_path).convert("RGB")
        image = transform(image).unsqueeze(0)

        return self.extract_batch(image)[0]
