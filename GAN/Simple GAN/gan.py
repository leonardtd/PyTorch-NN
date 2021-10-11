import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter

#SRC code
#https://www.youtube.com/watch?v=OljTVUVzPpM
#https://github.com/aladdinpersson/Machine-Learning-Collection/blob/master/ML/Pytorch/GANs/1.%20SimpleGAN/fc_gan.py

class Discriminator(nn.Module):
    def __init__(self, img_dim):
        super().__init__()
        self.disc = nn.Sequential(
            nn.Linear(img_dim, 128),
            nn.LeakyReLU(0.01),
            nn.Linear(128,1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.disc(x)


class Generator(nn.Module):
    def __init__(self, z_dim, img_dim): #z_dim is dimensions of latent noise
        super().__init__()
        self.gen = nn.Sequential(
            nn.Linear(z_dim, 256),
            nn.LeakyReLU(0.01),
            nn.Linear(256, img_dim), #28x28x1 -> 784 flattened
            nn.Tanh(),  #normalize MNIST dataset from -1 to +1, output should also be around these values
        )

    def forward(self, x):
        return self.gen(x)

"""
GANs are very sensitive to hyperparams
"""
#Hyperparams
device = 'cuda' if torch.cuda.is_available() else 'cpu'

lr = 3e-4 #play around
z_dim = 64 #128, 256, ..
image_dim = 28*28*1

batch_size = 32
num_epochs = 50

disc = Discriminator(image_dim).to(device)
gen = Generator(z_dim, image_dim).to(device)

fixed_noise = torch.randn((batch_size, z_dim)).to(device)
transforms = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,)),] #mean and std
)

dataset = datasets.MNIST(root='dataset/', transform=transforms, download=True)
loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

opt_disc = optim.Adam(disc.parameters(),lr=lr)
opt_gen = optim.Adam(gen.parameters(),lr=lr)
criterion = nn.BCELoss()

writer_fake = SummaryWriter(f"logs/fake")
writer_real = SummaryWriter(f"logs/real")
step = 0


for epoch in range(num_epochs):
    for batch_idx, (real, _) in enumerate(loader):
        real = real.view(-1, 784).to(device) #Flatten images, keeping batch size dim
        batch_size = real.shape[0]

        ### Train for Discrminator: max log(Disc(real)) + max log(1-Disc(Gen(z)))
        noise = torch.randn(batch_size, z_dim).to(device)
        fake = gen(noise)
        disc_real = disc(real).view(-1)
        
        #Criterion -1 -> maximize log(Disc(x))
        ### max log(Disc(real)
        lossD_real = criterion(disc_real, torch.ones_like(disc_real)) #yn is torch of ones
        
        ### max log(1-Disc(Gen(z)))
        disc_fake = disc(fake).view(-1)
        lossD_fake = criterion(disc_fake, torch.zeros_like(disc_fake))

        lossD = (lossD_real + lossD_fake)/2
        disc.zero_grad()
        lossD.backward(retain_graph=True) #Dont clear gradients, re utilize fake tensor
        opt_disc.step()

        ### Train Generator: min log(1-Disc(Gen(z))) <-> Leads to saturation, slower training
        ### BETTER: max (log(Disc(Gen(z))))
        output = disc(fake).view(-1)
        lossG = criterion(output, torch.ones_like(output))

        gen.zero_grad()
        lossG.backward()
        opt_gen.step()


        ### Tensorboard
        if batch_idx == 0:
            print(
                f"Epoch: [{epoch}/{num_epochs}] \ "
                f"Loss D: {lossD:.4f}, Loss G: {lossG:.4f}"
            )
            with torch.no_grad():
                fake = gen(fixed_noise).reshape(-1,1,28,28) #bs, channels, w, h
                data = real.reshape(-1,1,28,28)
                img_grid_fake = torchvision.utils.make_grid(fake, normalize=True)
                img_grid_real = torchvision.utils.make_grid(data, normalize=True)

                writer_fake.add_image(
                    "Mnist Fake Images", img_grid_fake, global_step=step
                )
                writer_real.add_image(
                    "Mnist Fake Images", img_grid_real, global_step=step
                )
                step+=1

