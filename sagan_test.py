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
from data_loader import Data_Loader
from spectral_norm import SpectralNorm
from self_attn_model import Self_Attn, Generator, Discriminator
from utils import make_folder, tensor2var, var2tensor, var2numpy, denorm, encode, log_scalar


import os
import time
import torch
import datetime

import torch.nn as nn
from torch.autograd import Variable
from torchvision.utils import save_image
from fid import  FID
from parameters import sagan_parameters


class SAGAN_test(object):
    def __init__(self, data_loader):

        # Data loader
        self.data_loader = data_loader
        self.labels_dict={0:'airplane',										
            1:'automobile',										
            2:'bird',										
            3:'cat',										
            4:'deer',										
            5:'dog',										
            6:'frog',										
            7:'horse',										
            8:'ship',										
            9:'truck'}
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

        self.d_iters = d_iters
        self.batch_size = batch_size
        self.num_workers = num_workers
 
        self.pretrained_model = pretrained_model

        self.dataset = dataset

        self.model_save_path = model_save_path
        self.test_path = test_path
 
        device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.fid_model=FID("./log_path",device) # as a var log path not a string changed

       
        self.build_model()


        # Start with trained model
        if self.pretrained_model:
            self.load_pretrained_model()
          
        
    def test(self):

        device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # Data iterator
        data_iter = iter(self.data_loader)

        # Fixed input for debugging
        fixed_z = tensor2var(torch.randn(self.batch_size,90 )) #self.z_dim

        self.G.eval()
        fid_scores=[]
        n_batches=2
        for i in range(n_batches):
           
          real_images, labels = next(iter(self.data_loader))
          
          if i==n_batches-1:
            if self.batch_size<=10:
              for l in labels:
                  print(self.labels_dict[l.item()])
            else:
              print("Avoiding to print labels since batch size greater than 10")
          # Compute loss with real images
          real_images = tensor2var(real_images)
          labels=tensor2var(encode(labels))
          
          z = tensor2var(torch.randn(real_images.size(0), 90))
          fake_images,gf1,gf2 = self.G(z,labels)
          fid_score=self.fid_model.compute_fid(real_images, fake_images)
          fid_scores.append(fid_score)
          
        
          
        fid_score=self.fid_model.compute_fid(real_images, fake_images)
        save_image(denorm(fake_images.data),'SAGAN_test.png')
        print("Image saved as SAGAN_test.png")
        avg_fid_score=sum(fid_scores)/len(fid_scores)
        print("Average FID_score for SA GAN, for ",n_batches," is:",avg_fid_score) 
        

    def build_model(self):
        device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.G = Generator(self.batch_size,self.imsize, self.z_dim, self.g_conv_dim).to(device)

    def build_tensorboard(self):
        return
        #from logger import Logger
        #self.logger = Logger(self.log_path)

    def load_pretrained_model(self):
        self.G.load_state_dict(torch.load(os.path.join(
            self.model_save_path, '{}_G.pth'.format(self.pretrained_model))))

        print('loaded trained models (step: {})..!'.format(self.pretrained_model))


    def save_sample(self, data_iter):
        real_images, _ = next(data_iter)
        save_image(denorm(real_images), os.path.join(self.sample_path, 'real.png'))
		
from torch.backends import cudnn
config=sagan_parameters()
cudnn.benchmark=True
train=False
batch_size=8
model='SAGAN'
model_save_path=""
test_path=""

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

#paths
root_dir=config.root_dir

test_loader = Data_Loader(train, dataset, imsize,
                        batch_size, shuf=True)
SAGAN_test(test_loader.loader()).test()