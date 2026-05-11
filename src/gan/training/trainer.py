import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torchvision.utils as vutils
from src.gan.utils.weights import weights_init
from src.config.gan import (
    GAN_INPUT_SIZE, GAN_BATCH_SIZE, GAN_NUM_EPOCHS,
    GAN_LEARNING_RATE, GAN_BETA1, GAN_N_CRITIC,
    GAN_LATENT_DIM, GAN_WORKERS
)

class GANTrainer:
    def __init__(self, netG, netD, dataset_dir, image_size=GAN_INPUT_SIZE, batch_size=GAN_BATCH_SIZE,
                 num_epochs=GAN_NUM_EPOCHS, lr=GAN_LEARNING_RATE, beta1=GAN_BETA1, n_critic=GAN_N_CRITIC, nz=GAN_LATENT_DIM, workers=GAN_WORKERS,
                 device=None, models_dir="models", samples_dir="samples", resume=False):
        self.netG = netG
        self.netD = netD
        self.dataset_dir = dataset_dir
        self.image_size = image_size
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.lr = lr
        self.beta1 = beta1
        self.n_critic = n_critic
        self.nz = nz
        self.workers = workers
        self.models_dir = models_dir
        self.samples_dir = samples_dir
        
        if device is None:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
            
        self.netG.to(self.device)
        self.netD.to(self.device)
        self.netG.apply(weights_init)
        self.netD.apply(weights_init)
        
        self.criterion = nn.BCEWithLogitsLoss()
        self.fixed_noise = torch.randn(64, self.nz, 1, 1, device=self.device)
        self.real_label = 1.
        self.fake_label = 0.
        
        self.optimizerD = optim.Adam(self.netD.parameters(), lr=self.lr, betas=(self.beta1, 0.999))
        self.optimizerG = optim.Adam(self.netG.parameters(), lr=self.lr, betas=(self.beta1, 0.999))
        
        self.start_epoch = 0
        if resume:
            epoch_file = os.path.join(self.models_dir, 'epoch.txt')
            if os.path.exists(epoch_file):
                with open(epoch_file, 'r') as f:
                    self.start_epoch = int(f.read().strip()) + 1
                self.netG.load_state_dict(torch.load(os.path.join(self.models_dir, 'generator.pth'), map_location=self.device))
                self.netD.load_state_dict(torch.load(os.path.join(self.models_dir, 'discriminator.pth'), map_location=self.device))
                print(f"Resuming training from epoch {self.start_epoch}")
        
        os.makedirs(self.samples_dir, exist_ok=True)
        os.makedirs(self.models_dir, exist_ok=True)
        self.history_dir = os.path.join(self.models_dir, 'history')
        os.makedirs(self.history_dir, exist_ok=True)
        
        self.dataloader = self._prepare_dataloader()

    def _prepare_dataloader(self):
        # ImageFolder expects subdirectories representing classes.
        # Resources directory is assumed to contain a subdirectory (like 'images') with the images.
        transform = transforms.Compose([
            transforms.Resize(self.image_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        
        dataset = datasets.ImageFolder(root=self.dataset_dir, transform=transform)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, 
                                                 shuffle=True, num_workers=self.workers)
        return dataloader

    def train(self):
        print(f"Using device: {self.device}")
        print(f"Total images in dataset: {len(self.dataloader.dataset)}")
        print("Starting Training Loop...")
        
        data_iter = iter(self.dataloader)
        
        for epoch in range(self.start_epoch, self.num_epochs):
            i = 0
            while i < len(self.dataloader):
                # (1) Update D network
                for _ in range(self.n_critic):
                    try:
                        data = next(data_iter)
                    except StopIteration:
                        data_iter = iter(self.dataloader)
                        data = next(data_iter)
                    
                    self.netD.zero_grad()
                    real_cpu = data[0].to(self.device)
                    b_size = real_cpu.size(0)
                    label = torch.full((b_size,), self.real_label, dtype=torch.float, device=self.device)
                    
                    output = self.netD(real_cpu).view(-1)
                    errD_real = self.criterion(output, label)
                    errD_real.backward()
                    D_x = output.mean().item()
                    
                    noise = torch.randn(b_size, self.nz, 1, 1, device=self.device)
                    fake = self.netG(noise)
                    label.fill_(self.fake_label)
                    output = self.netD(fake.detach()).view(-1)
                    errD_fake = self.criterion(output, label)
                    errD_fake.backward()
                    D_G_z1 = output.mean().item()
                    
                    errD = errD_real + errD_fake
                    self.optimizerD.step()
                    
                    i += 1
                    if i >= len(self.dataloader):
                        break

                if i >= len(self.dataloader):
                    break

                # (2) Update G network
                self.netG.zero_grad()
                label.fill_(self.real_label)
                
                noise = torch.randn(self.batch_size, self.nz, 1, 1, device=self.device)
                fake = self.netG(noise)
                
                output = self.netD(fake).view(-1)
                if output.size(0) != label.size(0):
                    label = torch.full((output.size(0),), self.real_label, dtype=torch.float, device=self.device)
                    
                errG = self.criterion(output, label)
                errG.backward()
                D_G_z2 = output.mean().item()
                self.optimizerG.step()
                
                if i % 10 == 0 or i >= len(self.dataloader):
                    print(f'[{epoch}/{self.num_epochs}][{i}/{len(self.dataloader)}] '
                          f'Loss_D: {errD.item():.4f} Loss_G: {errG.item():.4f} '
                          f'D(x): {D_x:.4f} D(G(z)): {D_G_z1:.4f} / {D_G_z2:.4f}')
                    with torch.no_grad():
                        fake = self.netG(self.fixed_noise).detach().cpu()
                    vutils.save_image(fake, os.path.join(self.samples_dir, f'fake_samples_epoch_{epoch:03d}_batch_{i:03d}.png'), normalize=True)

            # Save models at the end of each epoch to prevent data loss if interrupted
            self.save_models(epoch)

    def save_models(self, epoch=None):
        torch.save(self.netG.state_dict(), os.path.join(self.models_dir, 'generator.pth'))
        torch.save(self.netD.state_dict(), os.path.join(self.models_dir, 'discriminator.pth'))
        
        if epoch is not None:
            with open(os.path.join(self.models_dir, 'epoch.txt'), 'w') as f:
                f.write(str(epoch))
        
        if epoch is not None and epoch % 5 == 0:
            torch.save(self.netG.state_dict(), os.path.join(self.history_dir, f'generator_epoch_{epoch}.pth'))
            torch.save(self.netD.state_dict(), os.path.join(self.history_dir, f'discriminator_epoch_{epoch}.pth'))
            
        if epoch is not None:
            print(f"Models saved to {self.models_dir} (Epoch {epoch})")
            if epoch % 5 == 0:
                print(f"History models saved to {self.history_dir} (Epoch {epoch})")
        else:
            print(f"Models saved to {self.models_dir}")
