from __future__ import print_function
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision.utils import make_grid, save_image

import numpy as np

import argparse
import os
import shutil

from tensorboardX import SummaryWriter
from tqdm import tqdm
from models import Generator, Discriminator
from scores.fid_score import get_fid


def inits_weight(m):
    if type(m) == nn.Linear:
        nn.init.xavier_uniform_(m.weight.data, 1.0)


def noise(imgs, latent_dim):
    return ttype(np.random.normal(0, 1, (imgs.shape[0], latent_dim)))


def gener_noise(gener_batch_size, latent_dim):
    return ttype(np.random.normal(0, 1, (gener_batch_size, latent_dim)))


def save_checkpoint(states, is_best, output_dir, filename="checkpoint.pth"):
    torch.save(states, os.path.join(output_dir, filename))
    if is_best:
        torch.save(states, os.path.join(output_dir, "checkpoint_best.pth"))


dev = ""
if torch.has_mps and torch.backends.mps.is_built():
    dev = torch.device("mps")
elif torch.cuda.is_available():
    dev = torch.device("cuda")
else:
    dev = torch.device("cpu")

device = torch.device(dev)
print("Device:", device)

parser = argparse.ArgumentParser()
ttype = torch.FloatTensor
IMAGE_SIZE = 32
INITIAL_SIZE = 8
INPUT_CHANNEL = 3
PATCH_SIZE = 4
NUM_CLASSES = 1
LR_GENERATOR = 0.0001
LR_DISCRIMINATOR = 0.0001
LATENT_DIM = 1024
EMBED_DIM = 384
DIS_DEPTH = 7
N_CRITIC = 5
GEN_BATCH_SIZE = 64
DIS_BATCH_SIZE = 32
EPOCHS = 200
PHI = 1
BETA1 = 0.0
BETA2 = 0.99

LOSS = "hinge"
DIFF_AUG = "translation,cutout,color"
OUTPUT_DIR = "checkpoint"

best = 1e4
fid_stat_path = "fid_stat/fid_stats_cifar10_train.npz"
generated_dir = "generated_images"

if os.path.exists(generated_dir):
    shutil.rmtree(generated_dir)

os.makedirs(generated_dir)


generator = Generator(
    depth1=5,
    depth2=4,
    depth3=2,
    initial_size=INITIAL_SIZE,
    dim=EMBED_DIM,
    heads=4,
    mlp_ratio=4,
    drop_rate=0.5,
)
discriminator = Discriminator(
    diff_aug=DIFF_AUG,
    image_size=IMAGE_SIZE,
    patch_size=PATCH_SIZE,
    input_channel=INPUT_CHANNEL,
    num_classes=NUM_CLASSES,
    dim=EMBED_DIM,
    depth=DIS_DEPTH,
    heads=4,
    mlp_ratio=4,
    drop_rate=0.0,
)
discriminator.to(device)
generator.to(device)

generator.apply(inits_weight)
discriminator.apply(inits_weight)

optim_gen = optim.Adam(
    filter(lambda p: p.requires_grad, generator.parameters()),
    lr=LR_GENERATOR,
    betas=(BETA1, BETA2),
)

optim_dis = optim.Adam(
    filter(lambda p: p.requires_grad, discriminator.parameters()),
    lr=LR_DISCRIMINATOR,
    betas=(BETA1, BETA2),
)

writer = SummaryWriter()
writer_dict = {"writer": writer, "train_global_steps": 0.0, "valid_global_steps": 0.0}


def compute_gradient_penalty(D, real_samples, fake_samples, phi):
    """Calculates the gradient penalty loss for WGAN GP"""
    # Random weight term for interpolation between real and fake samples
    alpha = ttype(np.random.random((real_samples.size(0), 1, 1, 1))).to(
        real_samples.get_device()
    )
    # Get random interpolation between real and fake samples
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(
        True
    )
    d_interpolates = D(interpolates)
    fake = torch.ones([real_samples.shape[0], 1], requires_grad=False).to(
        real_samples.get_device()
    )
    # Get gradient w.r.t. interpolates
    gradients = torch.autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradients = gradients.contiguous().view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - phi) ** 2).mean()
    return gradient_penalty


def train(
    noise,
    generator,
    discriminator,
    optim_gen,
    optim_dis,
    epoch,
    writer,
    img_size=IMAGE_SIZE,
    latent_dim=LATENT_DIM,
    n_critic=N_CRITIC,
    gener_batch_size=GEN_BATCH_SIZE,
    gener_output_dir="generated_images",
):

    writer = writer_dict["writer"]
    gen_step = 0

    generator = generator.train()
    discriminator = discriminator.train()

    transform = transforms.Compose(
        [
            transforms.Resize(size=(img_size, img_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )
    train_set = torchvision.datasets.CIFAR10(
        root="input", train=True, download=True, transform=transform
    )
    train_loader = torch.utils.data.DataLoader(
        dataset=train_set, batch_size=30, shuffle=True
    )
    for index, (img, _) in enumerate(train_loader):
        global_steps = writer_dict["train_global_steps"]
        real_imgs = img.type(ttype).to(device)
        noise = ttype(np.random.normal(0, 1, (img.shape[0], latent_dim))).to(device)

        optim_dis.zero_grad()
        real_valid = discriminator(real_imgs)
        fake_imgs = generator(noise).detach()

        fake_valid = discriminator(fake_imgs)

        if LOSS == "hinge":
            loss_dis = torch.mean(nn.ReLU(inplace=True)(1.0 - real_valid)).to(
                device
            ) + torch.mean(nn.ReLU(inplace=True)(1 + fake_valid)).to(device)
        elif LOSS == "wgangp_eps":
            gradient_penalty = compute_gradient_penalty(
                discriminator, real_imgs, fake_imgs.detach(), PHI
            )
            loss_dis = (
                -torch.mean(real_valid)
                + torch.mean(fake_valid)
                + gradient_penalty * 10 / (PHI**2)
            )

        loss_dis.backward()
        optim_dis.step()

        writer.add_scalar("loss_dis", loss_dis.item(), global_steps)

        if global_steps % n_critic == 0:
            optim_gen.zero_grad()
            gener_noise = ttype(
                np.random.normal(0, 1, (gener_batch_size, latent_dim))
            ).to(device)
            generated_imgs = generator(gener_noise)
            fake_valid = discriminator(generated_imgs)

            gener_loss = -torch.mean(fake_valid).to(device)
            gener_loss.backward()
            optim_gen.step()
            writer.add_scalar("gener_loss", gener_loss.item(), global_steps)

            gen_step += 1

        if gen_step and index % 100 == 0:
            sample_imgs = generated_imgs[:25]
            save_image(
                sample_imgs,
                os.path.join(
                    gener_output_dir,
                    f"generated_img_{epoch}_{index % len(train_loader)}.jpg",
                ),
                nrow=5,
                normalize=True,
                scale_each=True,
            )
            tqdm.write(
                "[Epoch %d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
                % (
                    epoch + 1,
                    index % len(train_loader),
                    len(train_loader),
                    loss_dis.item(),
                    gener_loss.item(),
                )
            )


def validate(generator, writer_dict, fid_stat):  # ignored rn
    writer = writer_dict["writer"]
    global_steps = writer_dict["valid_global_steps"]

    generator = generator.eval()
    fid_score = get_fid(
        fid_stat,
        epoch,
        generator,
        num_img=5000,
        val_batch_size=60 * 2,
        latent_dim=1024,
        writer_dict=None,
        cls_idx=None,
    )

    print(f"FID score: {fid_score}")

    writer.add_scalar("FID_score", fid_score, global_steps)

    writer_dict["valid_global_steps"] = global_steps + 1
    return fid_score


for epoch in range(EPOCHS):
    train(
        noise,
        generator,
        discriminator,
        optim_gen,
        optim_dis,
        epoch,
        writer,
        img_size=IMAGE_SIZE,
        latent_dim=LATENT_DIM,
        n_critic=N_CRITIC,
        gener_batch_size=GEN_BATCH_SIZE,
        gener_output_dir=generated_dir,
    )

    checkpoint = {"epoch": epoch, "best_fid": best}
    checkpoint["generator_state_dict"] = generator.state_dict()
    checkpoint["discriminator_state_dict"] = discriminator.state_dict()

    score = validate(generator, writer_dict, fid_stat_path)

    print(f"FID score: {score} - best ID score: {best} || @ epoch {epoch+1}.")
    if epoch == 0 or epoch > 30:
        if score < best:
            save_checkpoint(checkpoint, is_best=(score < best), output_dir=OUTPUT_DIR)
            print("Saved Latest Model!")
            best = score


checkpoint = {"epoch": epoch, "best_fid": best}
checkpoint["generator_state_dict"] = generator.state_dict()
checkpoint["discriminator_state_dict"] = discriminator.state_dict()
score = validate(generator, writer_dict, fid_stat_path)  ####CHECK AGAIN
save_checkpoint(checkpoint, is_best=(score < best), output_dir=OUTPUT_DIR)
