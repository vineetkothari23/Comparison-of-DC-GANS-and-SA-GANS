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
from torch.backends import cudnn
from fid import FID
from dc_gan_model import Generator, Discriminator
from utils import ones_target, zeros_target, images_to_vectors, vectors_to_images, show_result, show_train_hist, generate_animation, data_load, imgs_resize, random_crop, random_fliplr, imshow, noise, log_tboard
from parameters import dcgan_parameters

cudnn.benchmark=True


def make_folder(path):
        if not os.path.exists(path):
            os.makedirs(path)

config=dcgan_parameters()
#root_dir="/content/drive/My Drive/Fall 2019/Deep Learning CSE 676/Projects/1/DC Gans/"
root_dir=config.root_dir
#output images
output_dir=config.output_dir
#models
model_dir=config.model_dir
#parameters
batch_size=config.batch_size
#input
data_dir=config.data_dir
inp_width=config.inp_width
inp_height=config.inp_height
inp_channels=config.inp_channels
nc=config.nc
nz=config.nz
ngf=config.ngf
ndf=config.ndf


"""##Loading the CIFAR Data"""

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                         shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

print("Size of test set",len(testloader))

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
G = Generator(ngf)

#loss

fid_model=FID("./",device)
#fid_model.compute_fid(real_image,G_result) #todo add fid score

G_checkpoint=torch.load('generator_param.pkl',map_location=device)
G.load_state_dict(G_checkpoint['model_state_dict'])
G.to(device)
G.eval()
batch_size=8
fid_scores=[]
n_batches=2
#testing]
for _ in range(n_batches):
  real_image,_=next(iter(testloader))
  inp_noise=torch.randn(batch_size, nz, 1, 1, device=device)
  inp_noise,real_image=Variable(inp_noise.to(device)),Variable(real_image.to(device))

  G_result=G(inp_noise)
  G_result=G_result.detach()
  fid_score=fid_model.compute_fid(real_image,G_result)
  fid_scores.append(fid_score)
#save_image(denorm(G_result[0].data),'SAGAN_test.png')

print("Average FID_score for DC GAN, for ",n_batches," is:",sum(fid_scores)/len(fid_scores))
vutils.save_image(G(inp_noise).detach(),
                  "DCGAN_test.png",
                  normalize=True)
print("Image saved as DCGAN_test.png")