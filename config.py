import torch
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2

LOAD_MODEL = True
#SAVE_MODEL = False
CHECKPOINT_GEN = "pretrained_models/gen.pth.tar"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LEARNING_RATE = 1e-4
HIGH_RES = 256
LOW_RES = HIGH_RES // 4
IMG_CHANNELS = 3

test_transform = A.Compose(
    [
        A.Normalize(mean=[0, 0, 0], std=[1, 1, 1]),
        ToTensorV2(),
    ]
)
