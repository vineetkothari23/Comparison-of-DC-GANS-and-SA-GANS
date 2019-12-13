import torch 
import torchvision
from torch import nn, optim
from torch.autograd.variable import Variable
from torchvision import transforms, datasets
import os
from PIL import Image
import glob
import PIL
from PIL import ImageFont
from PIL import Image
from PIL import ImageDraw
import matplotlib.pyplot as plt
import random
from bokeh.io import curdoc, show, output_notebook
from bokeh.layouts import column
from bokeh.models import ColumnDataSource
from bokeh.plotting import figure
from functools import partial
from threading import Thread
from tornado import gen
import time
import pickle
from tqdm import tqdm
import torchvision.utils as vutils
from torch.utils.tensorboard import SummaryWriter

model=config.model
adv_loss=config.adv_loss
imsize=config.imsize
g_num=config.g_num
z_dim=config.z_dim
label_dim=config.label_dim
g_conv_dim=config.g_conv_dim
d_conv_dim=config.d_conv_dim
lambda_gp=config.lambda_gp

#training parameters
d_iters=config.d_iters
batch_size=config.batch_size
num_workers=config.num_workers

lr_decay=config.lr_decay
beta1=config.beta1
beta2=config.beta2

#pretrained
pretrained_model=config.pretrained_model

#misc
train=config.train
parallel=config.parallel
dataset=config.dataset
use_tensorboard=config.use_tensorboard

image_path=root_dir+"data"
log_path=root_dir+"logs"
model_save_path=root_dir+"models"
sample_path=root_dir+"samples"
attn_path=root_dir+"attn"
log_step=10
sample_step=100
model_save_step=1

import os
import time
import torch
import datetime

import torch.nn as nn
from torch.autograd import Variable
from torchvision.utils import save_image

class Trainer(object):
    def __init__(self, data_loader):

        # Data loader
        self.data_loader = data_loader

        # exact model and loss
        self.model = model
        self.adv_loss = adv_loss

        # Model hyper-parameters
        self.imsize = imsize
        self.g_num = g_num
        self.z_dim = z_dim
        self.g_conv_dim = g_conv_dim
        self.d_conv_dim = d_conv_dim
        self.parallel = parallel

        self.lambda_gp = lambda_gp
        self.total_step = total_step
        self.d_iters = d_iters
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.g_lr = g_lr
        self.d_lr = d_lr
        self.lr_decay = lr_decay
        self.beta1 = beta1
        self.beta2 = beta2
        self.pretrained_model = pretrained_model

        self.dataset = dataset
        self.use_tensorboard = use_tensorboard
        self.image_path = image_path
        self.log_path = log_path
        self.model_save_path = model_save_path
        self.sample_path = sample_path
        self.log_step = log_step
        self.sample_step = sample_step
        self.model_save_step = model_save_step
        self.version = version
        device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.fid_model=FID("./log_path",device) # as a var log path not a string changed

        # Path
        #self.log_path = os.path.join(log_path, self.version)
        self.sample_path = os.path.join(sample_path, self.version)
        self.model_save_path = os.path.join(model_save_path, self.version)

        self.build_model()

        if self.use_tensorboard:
            self.build_tensorboard()

        # Start with trained model
        if self.pretrained_model:
            self.load_pretrained_model()
          
        #tensor board writer
        self.writer = SummaryWriter(log_path)



    def train(self):

        device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # Data iterator
        data_iter = iter(self.data_loader)
        step_per_epoch = len(self.data_loader)
        model_save_step = int(self.model_save_step * step_per_epoch)

        # Fixed input for debugging
        fixed_z = tensor2var(torch.randn(self.batch_size,90 )) #self.z_dim

        # Start with trained model
        if self.pretrained_model:
            start = self.pretrained_model + 1
        else:
            start = 0
        
        # Start time
        start_time = time.time()
        for step in range(start, self.total_step):

            # ================== Train D ================== #
            self.D.train()
            self.G.train()

            try:
                real_images, labels = next(data_iter)
            except:
                data_iter = iter(self.data_loader)
                real_images, labels = next(data_iter)

            # Compute loss with real images
            real_images = tensor2var(real_images)
            labels=tensor2var(encode(labels))
            d_out_real,dr1,dr2 = self.D(real_images,labels) #labels not added in generator, generator still not sorted.
            
            d_loss_real = - torch.mean(d_out_real)
            

            z = tensor2var(torch.randn(real_images.size(0), 90))
            fake_images,gf1,gf2 = self.G(z,labels)
            d_out_fake,df1,df2 = self.D(fake_images,labels)

            
            d_loss_fake = d_out_fake.mean()
           


            # Backward + Optimize
            d_loss = d_loss_real + d_loss_fake
            self.reset_grad()
            d_loss.backward()
            self.d_optimizer.step()


            if self.adv_loss == 'wgan-gp':
                # gradient penalty
                alpha = torch.rand(real_images.size(0), 1, 1, 1).to(device).expand_as(real_images)
                interpolated = Variable(alpha * real_images.data + (1 - alpha) * fake_images.data, requires_grad=True)
                out,_,_ = self.D(interpolated,labels)

                grad = torch.autograd.grad(outputs=out,
                                           inputs=interpolated,
                                           grad_outputs=torch.ones(out.size()).to(device),
                                           retain_graph=True,
                                           create_graph=True,
                                           only_inputs=True)[0]

                grad = grad.view(grad.size(0), -1)
                grad_l2norm = torch.sqrt(torch.sum(grad ** 2, dim=1))
                d_loss_gp = torch.mean((grad_l2norm - 1) ** 2)

                # Backward + Optimize
                d_loss = self.lambda_gp * d_loss_gp

                self.reset_grad()
                d_loss.backward()
                self.d_optimizer.step()

            # ================== Train G and gumbel ================== #
            # Create random noise
            z = tensor2var(torch.randn(real_images.size(0), 90)) #self.z_dim
            fake_images,_,_ = self.G(z,labels)

            # Compute loss with fake images
            g_out_fake,_,_ = self.D(fake_images,labels)  # batch x n
            if self.adv_loss == 'wgan-gp':
                g_loss_fake = - g_out_fake.mean()
            elif self.adv_loss == 'hinge':
                g_loss_fake = - g_out_fake.mean()

            self.reset_grad()
            g_loss_fake.backward()
            self.g_optimizer.step()
            

            # Print out log info
            if (step + 1) % self.log_step == 0:
                elapsed = time.time() - start_time
                elapsed = str(datetime.timedelta(seconds=elapsed))
                
                log_info={'G_loss':g_loss_fake.item(),'D_loss':g_loss_fake.item()}
                self.writer.add_scalar('G_loss',g_loss_fake.item(),step)
                self.writer.add_scalar('D_loss',d_loss.item(),step)                  
                print("Elapsed [{}], G_step [{}/{}], D_step[{}/{}],  "
                      " G_Loss: {:.4f}, D Loss: {:.4f}".
                      format(elapsed, step + 1, self.total_step, (step + 1),
                             self.total_step ,
                             g_loss_fake.item(), d_loss.item()))

            # Sample images
            if (step + 1) % self.sample_step == 0:
                fake_images,_,_= self.G(fixed_z,labels)
                fid_score=self.fid_model.compute_fid(real_images, fake_images)
                self.writer.add_scalar('FID_score',fid_score.item(),step)
                save_image(denorm(fake_images.data),
                           os.path.join(self.sample_path, '{}_fake.png'.format(step + 1)))
                print("Elapsed [{}], G_step [{}/{}], D_step[{}/{}],  "
                      " FID score: {:.4f}".
                      format(elapsed, step + 1, self.total_step, (step + 1),
                             self.total_step ,
                             fid_score.item()))

            if (step+1) % model_save_step==0:
                torch.save(self.G.state_dict(),
                           os.path.join(self.model_save_path, '{}_G.pth'.format(step + 1)))
                torch.save(self.D.state_dict(),
                           os.path.join(self.model_save_path, '{}_D.pth'.format(step + 1)))
        self.writer.close()

    def build_model(self):
        device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.G = Generator(self.batch_size,self.imsize, self.z_dim, self.g_conv_dim).to(device)
        self.D = Discriminator(self.batch_size,self.imsize, self.d_conv_dim).to(device)
        if self.parallel:
            self.G = nn.DataParallel(self.G)
            self.D = nn.DataParallel(self.D)

        # Loss and optimizer
        # self.g_optimizer = torch.optim.Adam(self.G.parameters(), self.g_lr, [self.beta1, self.beta2])
        self.g_optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.G.parameters()), self.g_lr, [self.beta1, self.beta2])
        self.d_optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.D.parameters()), self.d_lr, [self.beta1, self.beta2])

        self.c_loss = torch.nn.CrossEntropyLoss()


    def build_tensorboard(self):
        return
        #from logger import Logger
        #self.logger = Logger(self.log_path)

    def load_pretrained_model(self):
        self.G.load_state_dict(torch.load(os.path.join(
            self.model_save_path, '{}_G.pth'.format(self.pretrained_model))))
        self.D.load_state_dict(torch.load(os.path.join(
            self.model_save_path, '{}_D.pth'.format(self.pretrained_model))))
        print('loaded trained models (step: {})..!'.format(self.pretrained_model))

    def reset_grad(self):
        self.d_optimizer.zero_grad()
        self.g_optimizer.zero_grad()

    def save_sample(self, data_iter):
        real_images, _ = next(data_iter)
        save_image(denorm(real_images), os.path.join(self.sample_path, 'real.png'))
	
from torch.backends import cudnn

cudnn.benchmark=True
# Data loader
data_loader = Data_Loader(train, dataset, image_path, imsize,
                        batch_size, shuf=train)

# Create directories if not exist
make_folder(model_save_path, version)
make_folder(sample_path,version)
make_folder(log_path,version)
make_folder(attn_path,version)

trainer = Trainer(data_loader.loader())
trainer.train()
