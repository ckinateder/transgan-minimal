import torch
import torch.nn as nn
import os
from torchvision.utils import save_image

import torchvision.transforms as transforms
import torch.optim as optim
import torchvision.datasets as datasets
import matplotlib

from torch.utils.data import DataLoader
from matplotlib import pyplot as plt
from tqdm import tqdm, trange

matplotlib.style.use("ggplot")

device = ""
if torch.has_mps and torch.backends.mps.is_built():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


def label_real(size):
    """
    Fucntion to create real labels (ones)
    :param size: batch size
    :return real label vector
    """
    data = torch.ones(size, 1)
    return data.to(device)


def label_fake(size):
    """
    Fucntion to create fake labels (zeros)
    :param size: batch size
    :returns fake label vector
    """
    data = torch.zeros(size, 1)
    return data.to(device)


def create_noise(sample_size, nz):
    """
    Fucntion to create noise
    :param sample_size: fixed sample size or batch size
    :param nz: latent vector size
    :returns random noise vector
    """
    return torch.randn(sample_size, nz, 1, 1).to(device)


def save_generator_image(image, path):
    """
    Function to save torch image batches
    :param image: image tensor batch
    :param path: path name to save image
    """
    save_image(image, path, normalize=True)


def weights_init(m):
    """
    This function initializes the model weights randomly from a
    Normal distribution. This follows the specification from the DCGAN paper.
    https://arxiv.org/pdf/1511.06434.pdf
    Source: https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html
    """
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


# generator
class Generator(nn.Module):
    def __init__(self, nz):
        super(Generator, self).__init__()
        self.nz = nz
        self.main = nn.Sequential(
            # nz will be the input to the first convolution
            nn.ConvTranspose2d(nz, 512, kernel_size=4, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.ConvTranspose2d(
                512, 256, kernel_size=4, stride=2, padding=1, bias=False
            ),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.ConvTranspose2d(
                256, 128, kernel_size=4, stride=2, padding=1, bias=False
            ),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Tanh(),
        )

    def forward(self, input):
        return self.main(input)


# discriminator
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=0, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, input):
        return self.main(input)


# learning parameters / configurations according to paper
image_size = 64  # we need to resize image to 64x64
batch_size = 128
nz = 100  # latent vector size
beta1 = 0.5  # beta1 value for Adam optimizer
lr = 0.0002  # learning rate according to paper
sample_size = 64  # fixed sample size
epochs = 25  # number of epoch to train

# image transforms
transform = transforms.Compose(
    [
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]
)

# prepare the data
train_data = datasets.CIFAR10(
    root="input/data", train=True, download=True, transform=transform
)
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

# initialize models
generator = Generator(nz).to(device)
discriminator = Discriminator().to(device)

# initialize weights
generator.apply(weights_init)
discriminator.apply(weights_init)
print(generator)
print(discriminator)

# optimizers
optim_g = optim.Adam(generator.parameters(), lr=lr, betas=(beta1, 0.999))
optim_d = optim.Adam(discriminator.parameters(), lr=lr, betas=(beta1, 0.999))

# loss function
criterion = nn.BCELoss()

losses_g = []  # to store generator loss after each epoch
losses_d = []  # to store discriminator loss after each epoch

# function to train the discriminator network
def train_discriminator(optimizer, data_real, data_fake):
    b_size = data_real.size(0)
    # get the real label vector
    real_label = label_real(b_size).squeeze()
    # get the fake label vector
    fake_label = label_fake(b_size).unsqueeze(1).unsqueeze(1)

    optimizer.zero_grad()

    # get the outputs by doing real data forward pass
    output_real = discriminator(data_real).view(-1)
    loss_real = criterion(output_real, real_label)

    # get the outputs by doing fake data forward pass
    output_fake = discriminator(data_fake)
    loss_fake = criterion(output_fake, fake_label)

    # compute gradients of real loss
    loss_real.backward()
    # compute gradients of fake loss
    loss_fake.backward()
    # update discriminator parameters
    optimizer.step()

    return loss_real + loss_fake


# function to train the generator network
def train_generator(optimizer, data_fake):
    b_size = data_fake.size(0)
    # get the real label vector
    real_label = label_real(b_size).unsqueeze(1).unsqueeze(1)

    optimizer.zero_grad()

    # output by doing a forward pass of the fake data through discriminator
    output = discriminator(data_fake)
    loss = criterion(output, real_label)

    # compute gradients of loss
    loss.backward()
    # update generator parameters
    optimizer.step()

    return loss


# create the noise vector
noise = create_noise(sample_size, nz)
# print('SIZE', noise.size())
# print('NOISE', noise)


output_path = "outputs"
os.makedirs(output_path) if not os.path.exists(output_path) else ...
print(
    f"Training the model on {device}. Image size: {image_size}px, "
    f"Batch size: {batch_size}, Latent vector size: {nz}, "
    f"Learning rate: {lr}, Beta1: {beta1}, Epochs: {epochs}"
)
generator.train()
discriminator.train()

for epoch in trange(epochs, desc="epoch", position=1):
    loss_g = 0.0
    loss_d = 0.0

    final_bi = int(len(train_data) / int(train_loader.batch_size))
    for bi, data in tqdm(
        enumerate(train_loader),
        total=final_bi,
        desc="batch",
        position=0,
    ):
        image, _ = data
        image = image.to(device)
        b_size = len(image)

        # forward pass through generator to create fake data
        data_fake = generator(create_noise(b_size, nz)).detach()
        data_real = image
        loss_d += train_discriminator(optim_d, data_real, data_fake).cpu().detach()
        # data_fake = generator(create_noise(b_size, nz)).detach()
        loss_g += train_generator(optim_g, data_fake).cpu().detach()

    # final forward pass through generator to create fake data after training for current epoch
    generated_img = generator(noise).cpu().detach()

    # save the generated torch tensor models to disk
    save_generator_image(
        generated_img, os.path.join(output_path, f"gen_img{epoch}.png")
    )

    epoch_loss_g = loss_g / final_bi  # total generator loss for the epoch
    epoch_loss_d = loss_d / final_bi  # total discriminator loss for the epoch
    losses_g.append(epoch_loss_g)
    losses_d.append(epoch_loss_d)

    tqdm.write(
        f"Epoch {epoch+1} of {epochs} - Generator loss: {epoch_loss_g:.8f}, Discriminator loss: {epoch_loss_d:.8f}"
    )

print("DONE TRAINING")

# save the model weights to disk
torch.save(generator.state_dict(), "outputs/generator.pth")

# plot and save the generator and discriminator loss
plt.figure()
plt.plot(losses_g, label="Generator loss")
plt.plot(losses_d, label="Discriminator Loss")
plt.legend()
plt.savefig("outputs/loss.png")
plt.show()
