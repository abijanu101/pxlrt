import torch
import os
from src.gan.models import Generator, Discriminator
from src.gan.training.trainer import GANTrainer
from src.config.gan import (
    GAN_LATENT_DIM,
    GAN_GENERATOR_FILTERS,
    GAN_DISCRIMINATOR_FILTERS,
    GAN_CHANNELS
)

class GAN:
    def __init__(self, nz=GAN_LATENT_DIM, ngf=GAN_GENERATOR_FILTERS, ndf=GAN_DISCRIMINATOR_FILTERS, nc=GAN_CHANNELS, device=None):
        self.nz = nz
        self.nc = nc
        self.device = device if device else torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.netG = Generator(nz=nz, ngf=ngf, nc=nc).to(self.device)
        self.netD = Discriminator(nc=nc, ndf=ndf).to(self.device)

    def fit(self, dataset_dir, **kwargs):
        """
        Trains the GAN model.
        kwargs can include hyperparameters like batch_size, num_epochs, etc.
        """
        trainer = GANTrainer(self.netG, self.netD, dataset_dir, nz=self.nz, device=self.device, **kwargs)
        trainer.train()

    def load(self, models_dir="models"):
        """
        Loads the Generator and Discriminator weights from models_dir.
        """
        gen_path = os.path.join(models_dir, 'generator.pth')
        disc_path = os.path.join(models_dir, 'discriminator.pth')
        
        if not os.path.exists(gen_path) or not os.path.exists(disc_path):
            raise FileNotFoundError(f"Model files not found in {models_dir}")
            
        self.netG.load_state_dict(torch.load(gen_path, map_location=self.device))
        self.netD.load_state_dict(torch.load(disc_path, map_location=self.device))
        print(f"Models loaded from {models_dir}")

    def generate(self, num_samples=64, output_path=None):
        """
        Generates num_samples images using the trained Generator.
        Optionally saves them to output_path.
        """
        self.netG.eval()
        with torch.no_grad():
            noise = torch.randn(num_samples, self.nz, 1, 1, device=self.device)
            fake_images = self.netG(noise).detach().cpu()
            
        if output_path:
            import torchvision.utils as vutils
            vutils.save_image(fake_images, output_path, normalize=True)
            
        return fake_images
