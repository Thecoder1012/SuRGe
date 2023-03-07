import torch
import os
import config
import numpy as np
from PIL import Image
from torchvision.utils import save_image
from model import Generator
from utils import load_checkpoint, plot_examples
from torch import nn
from torch import optim
import os, random
import argparse

#create the code as easy as you can

result_path = "./Test_Results"
isExist = os.path.exists(result_path)
if not isExist:
    os.makedirs(result_path)


gen = Generator(alpha = 0.4, beta = 0.1, in_channels=3).to(config.DEVICE)
opt_gen = optim.Adam(gen.parameters(), lr=config.LEARNING_RATE, betas=(0.9, 0.999))

if config.LOAD_MODEL:
        load_checkpoint(
            config.CHECKPOINT_GEN,
            gen,
            opt_gen,
            config.LEARNING_RATE,
        )

src = "./demo/"
#src = "./internet/"
plot_examples(src, gen, result_path)
