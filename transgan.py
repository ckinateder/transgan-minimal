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
from models import Generator, Discriminator, inits_weight, compute_gradient_penalty
from scores.fid_score import get_fid


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
GEN_DEPTH_1 = 5
GEN_DEPTH_2 = 4
GEN_DEPTH_3 = 2
HEADS = 4
MLP_RATIO = 4
N_CRITIC = 5
GEN_BATCH_SIZE = 64
DIS_BATCH_SIZE = 64
EPOCHS = 200
PHI = 1
BETA1 = 0.0
BETA2 = 0.99
LOSS = "hinge"
DIFF_AUG = "translation,cutout,color"
OUTPUT_DIR = "checkpoint"

best = 1e4
FID_STAT_PATH = "fid_stat/fid_stats_cifar10_train.npz"
GENERATED_DIR = "generated_images"


def train(
    train_loader: torch.utils.data.DataLoader,
    generator: nn.Module,
    discriminator: nn.Module,
    optim_gen: torch.optim.Optimizer,
    optim_dis: torch.optim.Optimizer,
    epoch: int,
    writer_dict: dict,
    loss: str = LOSS,
    latent_dim: int = LATENT_DIM,
    n_critic: int = N_CRITIC,
    gen_batch_size: int = GEN_BATCH_SIZE,
    gen_output_dir: str = "generated_images",
):
    writer = writer_dict["writer"]
    gen_step = 0

    generator = generator.train()
    discriminator = discriminator.train()

    for index, (img, _) in enumerate(train_loader):
        global_steps = writer_dict["train_global_steps"]
        real_imgs = img.type(ttype).to(device)
        noise = ttype(np.random.normal(0, 1, (img.shape[0], latent_dim))).to(device)

        optim_dis.zero_grad()
        real_valid = discriminator(real_imgs)
        fake_imgs = generator(noise).detach()

        fake_valid = discriminator(fake_imgs)

        if loss == "hinge":
            loss_dis = torch.mean(nn.ReLU(inplace=True)(1.0 - real_valid)).to(
                device
            ) + torch.mean(nn.ReLU(inplace=True)(1 + fake_valid)).to(device)
        elif loss == "wgangp_eps":
            gradient_penalty = compute_gradient_penalty(
                discriminator,
                real_imgs,
                fake_imgs.detach(),
                PHI,
                ttype,
            )
            loss_dis = (
                -torch.mean(real_valid)
                + torch.mean(fake_valid)
                + gradient_penalty * 10 / (PHI**2)
            )
        else:
            raise NotImplementedError(f"Loss '{loss}' not implemented")

        loss_dis.backward()
        optim_dis.step()

        writer.add_scalar("loss_dis", loss_dis.item(), global_steps)

        if global_steps % n_critic == 0:
            optim_gen.zero_grad()
            gen_noise = ttype(np.random.normal(0, 1, (gen_batch_size, latent_dim))).to(
                device
            )
            gen_imgs = generator(gen_noise)
            fake_valid = discriminator(gen_imgs)

            gen_loss = -torch.mean(fake_valid).to(device)
            gen_loss.backward()
            optim_gen.step()
            writer.add_scalar("gen_loss", gen_loss.item(), global_steps)

            gen_step += 1

        if gen_step != 0 and index % 100 == 0:
            sample_imgs = gen_imgs[:25]  # gen_imgs will always be bound here
            save_image(
                sample_imgs,
                os.path.join(
                    gen_output_dir,
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
                    gen_loss.item(),  # gen_loss will always be bound here
                )
            )


def validate(generator: nn.Module, writer_dict: dict, fid_stat_path: str):
    writer = writer_dict["writer"]
    global_steps = writer_dict["valid_global_steps"]

    generator = generator.eval()
    fid_score = get_fid(
        fid_stat_path,
        epoch,
        generator,
        num_img=5000,
        val_batch_size=64 * 2,
        latent_dim=1024,
        writer_dict=None,
        cls_idx=None,
    )

    print(f"FID score: {fid_score}")

    writer.add_scalar("FID_score", fid_score, global_steps)

    writer_dict["valid_global_steps"] = global_steps + 1
    return fid_score


if __name__ == "__main__":
    transform = transforms.Compose(
        [
            transforms.Resize(size=(IMAGE_SIZE, IMAGE_SIZE)),
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

    if os.path.exists(GENERATED_DIR):
        shutil.rmtree(GENERATED_DIR)

    os.makedirs(GENERATED_DIR)

    generator = Generator(
        depth1=GEN_DEPTH_1,
        depth2=GEN_DEPTH_2,
        depth3=GEN_DEPTH_3,
        initial_size=INITIAL_SIZE,
        dim=EMBED_DIM,
        heads=HEADS,
        mlp_ratio=MLP_RATIO,
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
        heads=HEADS,
        mlp_ratio=MLP_RATIO,
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
    writer_dict = {
        "writer": writer,
        "train_global_steps": 0.0,
        "valid_global_steps": 0.0,
    }

    for epoch in range(EPOCHS):
        train(
            train_loader,
            generator,
            discriminator,
            optim_gen,
            optim_dis,
            epoch,
            writer_dict,
            loss=LOSS,
            latent_dim=LATENT_DIM,
            n_critic=N_CRITIC,
            gen_batch_size=GEN_BATCH_SIZE,
            gen_output_dir=GENERATED_DIR,
        )

        checkpoint = {"epoch": epoch, "best_fid": best}
        checkpoint["generator_state_dict"] = generator.state_dict()
        checkpoint["discriminator_state_dict"] = discriminator.state_dict()
        score = validate(generator, writer_dict, FID_STAT_PATH)

        print(f"FID score: {score} - best ID score: {best} || @ epoch {epoch+1}.")
        if epoch == 0 or epoch > 30:
            if score < best:
                save_checkpoint(
                    checkpoint,
                    is_best=(score < best),
                    output_dir=OUTPUT_DIR,
                )
                print("Saved Latest Model!")
                best = score
