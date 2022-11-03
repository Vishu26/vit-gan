from vit_pytorch import ViT
import torch
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning import LightningModule
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms, datasets
from einops import repeat, rearrange
from einops.layers.torch import Rearrange
from pytorch_lightning.callbacks import ModelCheckpoint
import numpy as np
from vit_pytorch.vit import Transformer
import wandb
from pytorch_lightning.loggers import WandbLogger
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torchmetrics import Accuracy

class vitGAN(LightningModule):
    def __init__(self, dataset):
        super().__init__()
        self.dataset = dataset
        self.generator = ViT(
            image_size = 64,
            patch_size = 8,
            num_classes = 10,
            dim = 1024,
            depth = 8,
            heads = 12,
            mlp_dim = 2048,
            dropout = 0.1,
            emb_dropout = 0.1
        )
        self.discriminator = ViT(
            image_size = 64,
            patch_size = 8,
            num_classes = 1024,
            dim = 1024,
            depth = 8,
            heads = 12,
            mlp_dim = 2048,
            dropout = 0.2,
            emb_dropout = 0.2
        )
        self.acc = Accuracy()        
        self.num_patches, self.gen_dim = self.generator.pos_embedding.shape[-2:]
        self.to_patch, self.patch_to_emb = self.generator.to_patch_embedding[:2]
        self.to_image = nn.Linear(1024, 8*8*3)
        self.out = nn.Sigmoid()
        self.classify = nn.Linear(1024, 1)
        self.re = Rearrange('b (h w) (p1 p2 c) -> b c (h p1) (w p2)', p1=8, p2=8, h=8) 
    
    def sample_z(self, batch, c, h, w):
        return torch.randn((batch, c, h, w)).to(self.device)
    
    def sample_gen(self, batch, c, h, w):
        z = self.sample_z(batch, c, h, w)
        patches = self.to_patch(z)
        tokens = self.patch_to_emb(patches)
        tokens = tokens + self.generator.pos_embedding[:, 1:(self.num_patches + 1)]
        tokens = self.generator.transformer(tokens)
        fake = self.out(self.to_image(tokens))
        return fake
    
    def dis(self, x):
        xout = self.discriminator(x)
        pd = torch.mean(nn.functional.pdist(torch.flatten(x, 1)))
        xout = xout + pd
        return self.classify(xout)
        
    def training_step(self, batch, batch_idx, optimizer_idx):
        x = batch[0]

        if optimizer_idx == 0:
            g_x = self.sample_gen(*x.shape)
            g_x = self.re(g_x)
            errG = torch.mean(-self.dis(g_x) * (-torch.ones(x.shape[0], device=self.device)))
            self.log('train_g_loss', errG)
            return errG
        
        if optimizer_idx == 1:
            g_x = self.sample_gen(*x.shape)
            g_x = self.re(g_x)
            errD_real = torch.mean(-self.dis(x) * (-torch.ones(x.shape[0], device=self.device)))
            errD_fake = torch.mean(-self.dis(g_x.detach()) * torch.ones(x.shape[0], device=self.device))
            errD = errD_real + errD_fake
            self.log('train_d_loss', errD)
            return errD

    def train_dataloader(self):
        return DataLoader(self.dataset, batch_size=32, shuffle=True, num_workers=64)

    def on_epoch_end(self):
        
        g_x = self.sample_gen(3, 3, 64, 64)
        g_x = g_x = rearrange(self.re(g_x), 'b c h w -> b h w c')
        g_x = g_x.cpu().detach().numpy()
        g_x = (g_x*255).astype(np.uint8)
        self.logger.experiment.log({
            "samples": [wandb.Image(img) for img in g_x]})    
        return
    
    def configure_optimizers(self):
        g_opt = torch.optim.RMSprop(self.generator.parameters(), lr=0.00005)
        d_opt = torch.optim.RMSprop(self.discriminator.parameters(), lr=0.00001)
        sch = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(d_opt, 40)
        return [g_opt, d_opt], [sch]

if __name__ == '__main__':

    dataset = datasets.ImageFolder('./img_align_celeba/', transform=transforms.Compose([
        transforms.Resize(64),
        transforms.CenterCrop(64),
        transforms.ToTensor()
    ]))
    model = vitGAN(dataset)
    wandb_logger = WandbLogger(project="vitGAN", name="vitGAN")
    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        dirpath='checkpoints/',
        filename='vitGAN-{epoch:02d}-{val_loss:.2f}',
        save_top_k=3,
        mode='min',
    )
    trainer = pl.Trainer(accelerator='gpu', devices=4, strategy='ddp', max_epochs=500, logger=wandb_logger, callbacks=[checkpoint_callback])
    trainer.fit(model)
