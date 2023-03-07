import torch
import config
from torch import nn
from torch import optim
from utils import load_checkpoint, save_checkpoint, plot_examples, gradient_penalty
from loss import ResLoss
from torch.utils.data import DataLoader
from model import Generator, Discriminator
from tqdm import tqdm
from dataset import MyImageFolder

torch.backends.cudnn.benchmark = True


def train_fn(loader, disc, gen, opt_gen, opt_disc, mse, bce, res_loss, epoch, total):
    loop = tqdm(loader, leave=True)

    for idx, (low_res, high_res) in enumerate(loop):
        high_res = high_res.to(config.DEVICE)
        low_res = low_res.to(config.DEVICE)
        
        ### Train Discriminator: max log(D(x)) + log(1 - D(G(z)))
        fake = gen(low_res)
        disc_real = disc(high_res)
        disc_fake = disc(fake.detach())
       
        loss_disc =  -(torch.mean(disc_real) - torch.mean(disc_fake))
        + 10 * gradient_penalty(disc, high_res, fake, device = config.DEVICE)

        opt_disc.zero_grad()
        loss_disc.backward()
        opt_disc.step()

        # Train Generator: min log(1 - D(G(z))) <-> max log(D(G(z))
        disc_fake = disc(fake)
        #l2_loss = mse(fake, high_res)
        adversarial_loss = 1e-3 * bce(disc_fake, torch.ones_like(disc_fake))
        loss_res = res_loss(fake, high_res)
        gen_loss = loss_res + adversarial_loss

        opt_gen.zero_grad()
        gen_loss.backward()
        opt_gen.step()
    if epoch % 50 == 0 or epoch == (total-1):
        #plot samples during training from test folder
        plot_examples("test/", gen, epoch)
        save_checkpoint(gen, opt_gen, filename="saved"+str(epoch)+"/"+config.CHECKPOINT_GEN)
        save_checkpoint(disc, opt_disc, filename="saved"+str(epoch)+"/"+config.CHECKPOINT_DISC)
    print("\nDisc Loss: ",loss_disc," Generator Loss: ",gen_loss)

def main():
    dataset = MyImageFolder(root_dir="path_to_train_dir")
    loader = DataLoader(
        dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        pin_memory=True,
        num_workers=config.NUM_WORKERS,
    )
    alpha = 0.3
    gen = Generator(alpha, in_channels=3).to(config.DEVICE)
    param = sum(p.numel() for p in gen.parameters())
    print("gen:",param)
    disc = Discriminator(in_channels=3).to(config.DEVICE)
    param = sum(p.numel() for p in disc.parameters())
    print("disc:",param)
    opt_gen = optim.Adam(gen.parameters(), lr=config.LEARNING_RATE, betas=(0.9, 0.999))
    opt_disc = optim.Adam(disc.parameters(), lr=config.LEARNING_RATE, betas=(0.0, 0.9))
    mse = nn.MSELoss()
    bce = nn.BCEWithLogitsLoss()
    res_loss = ResLoss()

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
        print(str(epoch)+"/"+str(config.NUM_EPOCHS))
        train_fn(loader, disc, gen, opt_gen, opt_disc, mse, bce, res_loss, epoch, config.NUM_EPOCHS)

        if config.SAVE_MODEL:
            save_checkpoint(gen, opt_gen, filename=config.CHECKPOINT_GEN)
            save_checkpoint(disc, opt_disc, filename=config.CHECKPOINT_DISC)


if __name__ == "__main__":
    main()
