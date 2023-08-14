import torch
import config
from torch import nn
from torch import optim
from utils import load_checkpoint, save_checkpoint, plot_examples, gradient_penalty
from loss import dwloss
from torch.utils.data import DataLoader
from model import Generator, Discriminator
from tqdm import tqdm
from dataset import MyImageFolder
import cv2
import numpy as np
import torch.nn.functional as F
import ot
import scipy as sp

torch.backends.cudnn.benchmark = True

def img_to_dist(image_batch):
    norm = image_batch.astype(float) / 255.0
    flat_img = norm.reshape(image_batch.shape[0], -1)
    dist = flat_img / np.sum(flat_img, axis = 1, keepdim = True)

    return dist


def train_fn(loader, disc, gen, opt_gen, opt_disc, mse, bce, dwl, epoch):
    loop = tqdm(loader, leave=True)

    for idx, (low_res, high_res) in enumerate(loop):
        high_res = high_res.to(config.DEVICE)
        low_res = low_res.to(config.DEVICE)
        
        ### Train Discriminator: max log(D(x)) + log(1 - D(G(z)))
        fake = gen(low_res)
        disc_real = disc(high_res)
        disc_fake = disc(fake.detach())
        # print(low_res.shape, fake.shape)
        # gw_loss = gw_l_n(low_res, fake)

        '''
        disc_loss_real = bce(
            disc_real, torch.ones_like(disc_real) - 0.1 * torch.rand_like(disc_real)
        )
        disc_loss_fake = bce(disc_fake, torch.zeros_like(disc_fake))
        loss_disc = disc_loss_fake + disc_loss_real
        '''
        loss_disc = 10 * gradient_penalty(disc, high_res, fake, device = config.DEVICE)

        opt_disc.zero_grad()
        loss_disc.backward()
        opt_disc.step()

        # Train Generator: min log(1 - D(G(z))) <-> max log(D(G(z))
        disc_fake = disc(fake)
        #l2_loss = mse(fake, high_res)
        # adversarial_loss = 1e-3 * bce(disc_fake, torch.ones_like(disc_fake))
        #loss_for_vgg = vgg_loss(fake, high_res) #changed in published paper but still exists in the Arxiv paper
        # print(fake.shape)
        # print(high_res.shape)
        # js_loss = jenson_shannon_divergence(fake, high_res)
        gen_loss = dwl(low_res, fake, high_res, disc_fake)
        # gen_loss = adversarial_loss + js_loss + gw_loss

        opt_gen.zero_grad()
        gen_loss.backward()
        opt_gen.step()

    if epoch % 10 == 0:
        plot_examples("test_images/", gen, epoch)
    print("\nDisc Loss: ",loss_disc," Generator Loss: ",gen_loss)

def main():
    # dataset = MyImageFolder(root_dir="/home/iplab/Desktop/jorbangla_select/DATASET/Code/DCGAN_PAPER_EXP/augmentation/aug_dat_v1")
    dataset = MyImageFolder(root_dir="/home/iplab/Desktop/jorbangla_select/DATASET/Code/SRGAN/div_2k_orig/DIV2K_train_HR")
    loader = DataLoader(
        dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        pin_memory=True,
        num_workers=config.NUM_WORKERS,
    )
    gen = Generator(in_channels=3).to(config.DEVICE)
    disc = Discriminator(in_channels=3).to(config.DEVICE)
    opt_gen = optim.Adam(gen.parameters(), lr=config.LEARNING_RATE, betas=(0.9, 0.999))
    opt_disc = optim.Adam(disc.parameters(), lr=config.LEARNING_RATE, betas=(0.9, 0.999))
    mse = nn.MSELoss()
    bce = nn.BCEWithLogitsLoss()
    dwl = dwloss()

    if config.LOAD_MODEL:
        load_checkpoint(
            config.CHECKPOINT_GEN,
            gen,
            opt_gen,
            config.LEARNING_RATE,
        )
        load_checkpoint(
           config.CHECKPOINT_DISC, disc, opt_disc, config.LEARNING_RATE,
        )

    for epoch in range(config.NUM_EPOCHS):
        print(str(epoch + 31)+"/"+str(config.NUM_EPOCHS))
        train_fn(loader, disc, gen, opt_gen, opt_disc, mse, bce, dwl, epoch + 31)

        if config.SAVE_MODEL:
            save_checkpoint(gen, opt_gen, filename=config.CHECKPOINT_GEN)
            save_checkpoint(disc, opt_disc, filename=config.CHECKPOINT_DISC)


if __name__ == "__main__":
    main()
