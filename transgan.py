from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision.utils import make_grid, save_image

import numpy as np

import argparse
import os
import random

from tensorboardX import SummaryWriter
from tqdm import tqdm


def noise(n_samples, z_dim, device):
    return torch.randn(n_samples, z_dim).to(device)


class LinearLrDecay(object):
    def __init__(self, optimizer, start_lr, end_lr, decay_start_step, decay_end_step):

        assert start_lr > end_lr
        self.optimizer = optimizer
        self.delta = (start_lr - end_lr) / (decay_end_step - decay_start_step)
        self.decay_start_step = decay_start_step
        self.decay_end_step = decay_end_step
        self.start_lr = start_lr
        self.end_lr = end_lr

    def step(self, current_step):
        if current_step <= self.decay_start_step:
            lr = self.start_lr
        elif current_step >= self.decay_end_step:
            lr = self.end_lr
        else:
            lr = self.start_lr - self.delta * (current_step - self.decay_start_step)
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = lr
        return lr


def inits_weight(m):
    if type(m) == nn.Linear:
        nn.init.xavier_uniform_(m.weight.data, 1.0)


def noise(imgs, latent_dim):
    return torch.FloatTensor(np.random.normal(0, 1, (imgs.shape[0], latent_dim)))


def gener_noise(gener_batch_size, latent_dim):
    return torch.FloatTensor(np.random.normal(0, 1, (gener_batch_size, latent_dim)))


def save_checkpoint(states, is_best, output_dir, filename="checkpoint.pth"):
    torch.save(states, os.path.join(output_dir, filename))
    if is_best:
        torch.save(states, os.path.join(output_dir, "checkpoint_best.pth"))


def DiffAugment(x, policy="", channels_first=True):
    if policy:
        if not channels_first:
            x = x.permute(0, 3, 1, 2)
        for p in policy.split(","):
            for f in AUGMENT_FNS[p]:
                x = f(x)
        if not channels_first:
            x = x.permute(0, 2, 3, 1)
        x = x.contiguous()
    return x


def rand_brightness(x):
    x = x + (torch.rand(x.size(0), 1, 1, 1, dtype=x.dtype, device=x.device) - 0.5)
    return x


def rand_saturation(x):
    x_mean = x.mean(dim=1, keepdim=True)
    x = (x - x_mean) * (
        torch.rand(x.size(0), 1, 1, 1, dtype=x.dtype, device=x.device) * 2
    ) + x_mean
    return x


def rand_contrast(x):
    x_mean = x.mean(dim=[1, 2, 3], keepdim=True)
    x = (x - x_mean) * (
        torch.rand(x.size(0), 1, 1, 1, dtype=x.dtype, device=x.device) + 0.5
    ) + x_mean
    return x


def rand_translation(x, ratio=0.2):
    shift_x, shift_y = int(x.size(2) * ratio + 0.5), int(x.size(3) * ratio + 0.5)
    translation_x = torch.randint(
        -shift_x, shift_x + 1, size=[x.size(0), 1, 1], device=x.device
    )
    translation_y = torch.randint(
        -shift_y, shift_y + 1, size=[x.size(0), 1, 1], device=x.device
    )
    grid_batch, grid_x, grid_y = torch.meshgrid(
        torch.arange(x.size(0), dtype=torch.long, device=x.device),
        torch.arange(x.size(2), dtype=torch.long, device=x.device),
        torch.arange(x.size(3), dtype=torch.long, device=x.device),
    )
    grid_x = torch.clamp(grid_x + translation_x + 1, 0, x.size(2) + 1)
    grid_y = torch.clamp(grid_y + translation_y + 1, 0, x.size(3) + 1)
    x_pad = F.pad(x, [1, 1, 1, 1, 0, 0, 0, 0])
    x = (
        x_pad.permute(0, 2, 3, 1)
        .contiguous()[grid_batch, grid_x, grid_y]
        .permute(0, 3, 1, 2)
    )
    return x


def rand_cutout(x, ratio=0.5):
    if random.random() < 0.3:
        cutout_size = int(x.size(2) * ratio + 0.5), int(x.size(3) * ratio + 0.5)
        offset_x = torch.randint(
            0,
            x.size(2) + (1 - cutout_size[0] % 2),
            size=[x.size(0), 1, 1],
            device=x.device,
        )
        offset_y = torch.randint(
            0,
            x.size(3) + (1 - cutout_size[1] % 2),
            size=[x.size(0), 1, 1],
            device=x.device,
        )
        grid_batch, grid_x, grid_y = torch.meshgrid(
            torch.arange(x.size(0), dtype=torch.long, device=x.device),
            torch.arange(cutout_size[0], dtype=torch.long, device=x.device),
            torch.arange(cutout_size[1], dtype=torch.long, device=x.device),
        )
        grid_x = torch.clamp(
            grid_x + offset_x - cutout_size[0] // 2, min=0, max=x.size(2) - 1
        )
        grid_y = torch.clamp(
            grid_y + offset_y - cutout_size[1] // 2, min=0, max=x.size(3) - 1
        )
        mask = torch.ones(
            x.size(0), x.size(2), x.size(3), dtype=x.dtype, device=x.device
        )
        mask[grid_batch, grid_x, grid_y] = 0
        x = x * mask.unsqueeze(1)
    return x


def rand_rotate(x, ratio=0.5):
    k = random.randint(1, 3)
    if random.random() < ratio:
        x = torch.rot90(x, k, [2, 3])
    return x


AUGMENT_FNS = {
    "color": [rand_brightness, rand_saturation, rand_contrast],
    "translation": [rand_translation],
    "cutout": [rand_cutout],
    "rotate": [rand_rotate],
}


class MLP(nn.Module):
    def __init__(self, in_feat, hid_feat=None, out_feat=None, dropout=0.0):
        super().__init__()
        if not hid_feat:
            hid_feat = in_feat
        if not out_feat:
            out_feat = in_feat
        self.fc1 = nn.Linear(in_feat, hid_feat)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hid_feat, out_feat)
        self.droprateout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return self.droprateout(x)


class Attention(nn.Module):
    def __init__(self, dim, heads=4, attention_dropout=0.0, proj_dropout=0.0):
        super().__init__()
        self.heads = heads
        self.scale = 1.0 / dim**0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.attention_dropout = nn.Dropout(attention_dropout)
        self.out = nn.Sequential(nn.Linear(dim, dim), nn.Dropout(proj_dropout))

    def forward(self, x):
        b, n, c = x.shape
        qkv = self.qkv(x).reshape(b, n, 3, self.heads, c // self.heads)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)

        dot = (q @ k.transpose(-2, -1)) * self.scale
        attn = dot.softmax(dim=-1)
        attn = self.attention_dropout(attn)

        x = (attn @ v).transpose(1, 2).reshape(b, n, c)
        x = self.out(x)
        return x


class ImgPatches(nn.Module):
    def __init__(self, input_channel=3, dim=768, patch_size=4):
        super().__init__()
        self.patch_embed = nn.Conv2d(
            input_channel, dim, kernel_size=patch_size, stride=patch_size
        )

    def forward(self, img):
        patches = self.patch_embed(img).flatten(2).transpose(1, 2)
        return patches


def UpSampling(x, H, W):
    B, N, C = x.size()
    assert N == H * W
    x = x.permute(0, 2, 1)
    x = x.view(-1, C, H, W)
    x = nn.PixelShuffle(2)(x)
    B, C, H, W = x.size()
    x = x.view(-1, C, H * W)
    x = x.permute(0, 2, 1)
    return x, H, W


class Encoder_Block(nn.Module):
    def __init__(self, dim, heads, mlp_ratio=4, drop_rate=0.0):
        super().__init__()
        self.ln1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, heads, drop_rate, drop_rate)
        self.ln2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, dim * mlp_ratio, dropout=drop_rate)

    def forward(self, x):
        x1 = self.ln1(x)
        x = x + self.attn(x1)
        x2 = self.ln2(x)
        x = x + self.mlp(x2)
        return x


class TransformerEncoder(nn.Module):
    def __init__(self, depth, dim, heads, mlp_ratio=4, drop_rate=0.0):
        super().__init__()
        self.Encoder_Blocks = nn.ModuleList(
            [Encoder_Block(dim, heads, mlp_ratio, drop_rate) for i in range(depth)]
        )

    def forward(self, x):
        for Encoder_Block in self.Encoder_Blocks:
            x = Encoder_Block(x)
        return x


class Generator(nn.Module):
    """docstring for Generator"""

    def __init__(
        self,
        depth1=5,
        depth2=4,
        depth3=2,
        initial_size=8,
        dim=384,
        heads=4,
        mlp_ratio=4,
        drop_rate=0.0,
    ):
        super(Generator, self).__init__()

        self.initial_size = initial_size
        self.dim = dim
        self.depth1 = depth1
        self.depth2 = depth2
        self.depth3 = depth3
        self.heads = heads
        self.mlp_ratio = mlp_ratio
        self.droprate_rate = drop_rate

        self.mlp = nn.Linear(1024, (self.initial_size**2) * self.dim)

        self.positional_embedding_1 = nn.Parameter(torch.zeros(1, (8**2), 384))
        self.positional_embedding_2 = nn.Parameter(
            torch.zeros(1, (8 * 2) ** 2, 384 // 4)
        )
        self.positional_embedding_3 = nn.Parameter(
            torch.zeros(1, (8 * 4) ** 2, 384 // 16)
        )

        self.TransformerEncoder_encoder1 = TransformerEncoder(
            depth=self.depth1,
            dim=self.dim,
            heads=self.heads,
            mlp_ratio=self.mlp_ratio,
            drop_rate=self.droprate_rate,
        )
        self.TransformerEncoder_encoder2 = TransformerEncoder(
            depth=self.depth2,
            dim=self.dim // 4,
            heads=self.heads,
            mlp_ratio=self.mlp_ratio,
            drop_rate=self.droprate_rate,
        )
        self.TransformerEncoder_encoder3 = TransformerEncoder(
            depth=self.depth3,
            dim=self.dim // 16,
            heads=self.heads,
            mlp_ratio=self.mlp_ratio,
            drop_rate=self.droprate_rate,
        )

        self.linear = nn.Sequential(nn.Conv2d(self.dim // 16, 3, 1, 1, 0))

    def forward(self, noise):

        x = self.mlp(noise).view(-1, self.initial_size**2, self.dim)

        x = x + self.positional_embedding_1
        H, W = self.initial_size, self.initial_size
        x = self.TransformerEncoder_encoder1(x)

        x, H, W = UpSampling(x, H, W)
        x = x + self.positional_embedding_2
        x = self.TransformerEncoder_encoder2(x)

        x, H, W = UpSampling(x, H, W)
        x = x + self.positional_embedding_3

        x = self.TransformerEncoder_encoder3(x)
        x = self.linear(x.permute(0, 2, 1).view(-1, self.dim // 16, H, W))

        return x


class Discriminator(nn.Module):
    def __init__(
        self,
        diff_aug,
        image_size=32,
        patch_size=4,
        input_channel=3,
        num_classes=1,
        dim=384,
        depth=7,
        heads=4,
        mlp_ratio=4,
        drop_rate=0.0,
    ):
        super().__init__()
        if image_size % patch_size != 0:
            raise ValueError("Image size must be divisible by patch size.")
        num_patches = (image_size // patch_size) ** 2
        self.diff_aug = diff_aug
        self.patch_size = patch_size
        self.depth = depth
        # Image patches and embedding layer
        self.patches = ImgPatches(input_channel, dim, self.patch_size)

        # Embedding for patch position and class
        self.positional_embedding = nn.Parameter(torch.zeros(1, num_patches + 1, dim))
        self.class_embedding = nn.Parameter(torch.zeros(1, 1, dim))
        nn.init.trunc_normal_(self.positional_embedding, std=0.2)
        nn.init.trunc_normal_(self.class_embedding, std=0.2)

        self.droprate = nn.Dropout(p=drop_rate)
        self.TransfomerEncoder = TransformerEncoder(
            depth, dim, heads, mlp_ratio, drop_rate
        )
        self.norm = nn.LayerNorm(dim)
        self.out = nn.Linear(dim, num_classes)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        x = DiffAugment(x, self.diff_aug)
        b = x.shape[0]
        cls_token = self.class_embedding.expand(b, -1, -1)

        x = self.patches(x)
        x = torch.cat((cls_token, x), dim=1)
        x += self.positional_embedding
        x = self.droprate(x)
        x = self.TransfomerEncoder(x)
        x = self.norm(x)
        x = self.out(x[:, 0])
        return


parser = argparse.ArgumentParser()
parser.add_argument(
    "--image_size", type=int, default=32, help="Size of image for discriminator input."
)
parser.add_argument(
    "--initial_size", type=int, default=8, help="Initial size for generator."
)
parser.add_argument(
    "--patch_size", type=int, default=4, help="Patch size for generated image."
)
parser.add_argument(
    "--num_classes", type=int, default=1, help="Number of classes for discriminator."
)
parser.add_argument(
    "--lr_gen", type=float, default=0.0001, help="Learning rate for generator."
)
parser.add_argument(
    "--lr_dis", type=float, default=0.0001, help="Learning rate for discriminator."
)
parser.add_argument("--weight_decay", type=float, default=1e-3, help="Weight decay.")
parser.add_argument("--latent_dim", type=int, default=1024, help="Latent dimension.")
parser.add_argument("--n_critic", type=int, default=5, help="n_critic.")
parser.add_argument("--max_iter", type=int, default=500000, help="max_iter.")
parser.add_argument(
    "--gener_batch_size", type=int, default=64, help="Batch size for generator."
)
parser.add_argument(
    "--dis_batch_size", type=int, default=32, help="Batch size for discriminator."
)
parser.add_argument("--epoch", type=int, default=200, help="Number of epoch.")
parser.add_argument("--output_dir", type=str, default="checkpoint", help="Checkpoint.")
parser.add_argument("--dim", type=int, default=384, help="Embedding dimension.")
parser.add_argument(
    "--img_name", type=str, default="img_name", help="Name of pictures file."
)
parser.add_argument("--loss", type=str, default="wgangp_eps", help="Loss function")
parser.add_argument("--phi", type=int, default="1", help="phi")
parser.add_argument("--beta1", type=int, default="0", help="beta1")
parser.add_argument("--beta2", type=float, default="0.99", help="beta2")
parser.add_argument("--lr_decay", type=str, default=True, help="lr_decay")
parser.add_argument(
    "--diff_aug", type=str, default="translation,cutout,color", help="Data Augmentation"
)
ttype = torch.FloatTensor

dev = ""
if torch.has_mps and torch.backends.mps.is_built():
    dev = torch.device("mps")
elif torch.cuda.is_available():
    dev = torch.device("cuda")
else:
    dev = torch.device("cpu")

device = torch.device(dev)
print("Device:", device)

args = parser.parse_args()

generator = Generator(
    depth1=5,
    depth2=4,
    depth3=2,
    initial_size=8,
    dim=384,
    heads=4,
    mlp_ratio=4,
    drop_rate=0.5,
)
generator.to(device)

discriminator = Discriminator(
    diff_aug=args.diff_aug,
    image_size=32,
    patch_size=4,
    input_channel=3,
    num_classes=1,
    dim=384,
    depth=7,
    heads=4,
    mlp_ratio=4,
    drop_rate=0.0,
)
discriminator.to(device)


generator.apply(inits_weight)
discriminator.apply(inits_weight)

optim_gen = optim.Adam(
    filter(lambda p: p.requires_grad, generator.parameters()),
    lr=args.lr_gen,
    betas=(args.beta1, args.beta2),
)

optim_dis = optim.Adam(
    filter(lambda p: p.requires_grad, discriminator.parameters()),
    lr=args.lr_dis,
    betas=(args.beta1, args.beta2),
)

gen_scheduler = LinearLrDecay(
    optim_gen, args.lr_gen, 0.0, 0, args.max_iter * args.n_critic
)
dis_scheduler = LinearLrDecay(
    optim_dis, args.lr_dis, 0.0, 0, args.max_iter * args.n_critic
)


print("optim:", args.optim)

fid_stat = "fid_stat/fid_stats_cifar10_train.npz"
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
    schedulers,
    img_size=32,
    latent_dim=args.latent_dim,
    n_critic=args.n_critic,
    gener_batch_size=args.gener_batch_size,
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

        real_imgs = img.type(ttype)

        noise = ttype(np.random.normal(0, 1, (img.shape[0], latent_dim)))

        optim_dis.zero_grad()
        real_valid = discriminator(real_imgs)
        fake_imgs = generator(noise).detach()

        fake_valid = discriminator(fake_imgs)

        if args.loss == "hinge":
            loss_dis = torch.mean(nn.ReLU(inplace=True)(1.0 - real_valid)).to(
                device
            ) + torch.mean(nn.ReLU(inplace=True)(1 + fake_valid)).to(device)
        elif args.loss == "wgangp_eps":
            gradient_penalty = compute_gradient_penalty(
                discriminator, real_imgs, fake_imgs.detach(), args.phi
            )
            loss_dis = (
                -torch.mean(real_valid)
                + torch.mean(fake_valid)
                + gradient_penalty * 10 / (args.phi**2)
            )

        loss_dis.backward()
        optim_dis.step()

        writer.add_scalar("loss_dis", loss_dis.item(), global_steps)

        if global_steps % n_critic == 0:

            optim_gen.zero_grad()
            if schedulers:
                gen_scheduler, dis_scheduler = schedulers
                g_lr = gen_scheduler.step(global_steps)
                d_lr = dis_scheduler.step(global_steps)
                writer.add_scalar("LR/g_lr", g_lr, global_steps)
                writer.add_scalar("LR/d_lr", d_lr, global_steps)

            gener_noise = ttype(np.random.normal(0, 1, (gener_batch_size, latent_dim)))

            generated_imgs = generator(gener_noise)
            fake_valid = discriminator(generated_imgs)

            gener_loss = -torch.mean(fake_valid).to(device)
            gener_loss.backward()
            optim_gen.step()
            writer.add_scalar("gener_loss", gener_loss.item(), global_steps)

            gen_step += 1

        if gen_step and index % 100 == 0:
            sample_imgs = generated_imgs[:25]
            img_grid = make_grid(sample_imgs, nrow=5, normalize=True, scale_each=True)
            save_image(
                sample_imgs,
                f"generated_images/generated_img_{epoch}_{index % len(train_loader)}.jpg",
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
    fid_score = 1000

    print(f"FID score: {fid_score}")

    writer.add_scalar("FID_score", fid_score, global_steps)

    writer_dict["valid_global_steps"] = global_steps + 1
    return fid_score


best = 1e4


for epoch in range(args.epoch):
    lr_schedulers = (gen_scheduler, dis_scheduler) if args.lr_decay else None
    train(
        noise,
        generator,
        discriminator,
        optim_gen,
        optim_dis,
        epoch,
        writer,
        lr_schedulers,
        img_size=32,
        latent_dim=args.latent_dim,
        n_critic=args.n_critic,
        gener_batch_size=args.gener_batch_size,
    )

    checkpoint = {"epoch": epoch, "best_fid": best}
    checkpoint["generator_state_dict"] = generator.state_dict()
    checkpoint["discriminator_state_dict"] = discriminator.state_dict()

    score = validate(generator, writer_dict, fid_stat)

    print(f"FID score: {score} - best ID score: {best} || @ epoch {epoch+1}.")
    if epoch == 0 or epoch > 30:
        if score < best:
            save_checkpoint(
                checkpoint, is_best=(score < best), output_dir=args.output_dir
            )
            print("Saved Latest Model!")
            best = score


checkpoint = {"epoch": epoch, "best_fid": best}
checkpoint["generator_state_dict"] = generator.state_dict()
checkpoint["discriminator_state_dict"] = discriminator.state_dict()
score = validate(generator, writer_dict, fid_stat)  ####CHECK AGAIN
save_checkpoint(checkpoint, is_best=(score < best), output_dir=args.output_dir)
