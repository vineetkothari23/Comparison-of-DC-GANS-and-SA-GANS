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

#data directories
#output images
output_dir=root_dir+"output/epoch/"
#input images
input_dir=root_dir+"data/augmented/"

#log dir
log_dir=root_dir+"logs/"

#models
model_dir=root_dir+"model/"
#other resources
res_dir=root_dir+"res/"
#report and logging
report_dir=root_dir+"report/"

restart_train=True

if restart_train:
  # Create directories if not exist
  make_folder(output_dir)
  make_folder(input_dir)
  make_folder(log_dir)
  make_folder(model_dir)
  make_folder(res_dir)
  make_folder(report_dir)


transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

dataset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)

train_size=int(train_split*len(dataset))
val_size=len(dataset)-train_size
trainset, valset=torch.utils.data.random_split(dataset,[train_size,val_size])
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=2)
valloader = torch.utils.data.DataLoader(valset, batch_size=batch_size,
                                          shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                         shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

print("Size of train set",len(trainloader))
print("Size of val set",len(valloader))
print("Size of test set",len(testloader))



# get some random training images
dataiter = iter(trainloader)
images, labels = dataiter.next()

# show images
#imshow(torchvision.utils.make_grid(images))
# print labels
print(' '.join('%5s' % classes[labels[j]] for j in range(batch_size)))
print("Size of images",images.size())

#from google.colab import drive
#drive.mount('/content/drive/',force_remount=True)

from torch.utils.tensorboard import SummaryWriter
print(len(dataset))
train_size=int(train_split*len(dataset))
val_size=len(dataset)-train_size
train_dataset, val_dataset=torch.utils.data.random_split(dataset,[train_size,val_size])
train_dataloader=torch.utils.data.DataLoader(dataset=train_dataset, 
                                              batch_size=batch_size,
                                             shuffle=True,
                                          num_workers=4)
num_batches=len(train_dataloader)
val_dataloader=torch.utils.data.DataLoader(dataset=val_dataset,
                                             batch_size=batch_size,
                                             shuffle=True,
                                          num_workers=4)


#from model import generator, discriminator
#import utils

#parameters
lrG=0.02
lrD=0.02
beta1=0.5
beta2=0.999
L1_lambda=2
train_epoch=5000


start_time=time.time()
epoch_start=0
epoch_end=epoch_start+train_epoch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
G = Generator(ngf)
D = Discriminator(ndf)
G_optimizer=optim.Adam(G.parameters(),lr=lrG,betas=(beta1,beta2))
D_optimizer=optim.Adam(D.parameters(),lr=lrD,betas=(beta1,beta2))
#loss

BCE_loss=nn.BCELoss().to(device)
L1_loss=nn.L1Loss().to(device)
fid_model=FID("./",device)
#fid_model.compute_fid(real_image,G_result) #todo add fid score


#summary writer
writer=SummaryWriter(log_dir)

#fixed noise for visualiing images
save_noise=torch.randn(1, nz, 1, 1, device=device)

#Loading the model if previously exists
if(os.path.isfile(model_dir+'generator_param.pkl') and os.path.isfile(model_dir+'discriminator_param.pkl')):

  G_checkpoint=torch.load(model_dir+'generator_param.pkl',map_location=device)
  D_checkpoint=torch.load(model_dir+'discriminator_param.pkl',map_location=device)
  G.load_state_dict(G_checkpoint['model_state_dict'])
  D.load_state_dict(D_checkpoint['model_state_dict'])
  G.to(device)
  D.to(device)
  G.train()
  D.train()
  #D.eval()

  G_optimizer.load_state_dict(G_checkpoint['optimizer_state_dict'])
  D_optimizer.load_state_dict(D_checkpoint['optimizer_state_dict'])

  train_hist=G_checkpoint['train_hist']
  epoch_start=G_checkpoint['epoch']
  epoch_end=epoch_start+train_epoch
#Esle creating new model
else:  
  print("Previous model not found. Restarting train process...")
  G.apply(weights_init)
  D.apply(weights_init)
  G.to(device)
  D.to(device)
  G.train()
  D.train()


  G_optimizer=optim.Adam(G.parameters(),lr=lrG,betas=(beta1,beta2))
  D_optimizer=optim.Adam(D.parameters(),lr=lrD,betas=(beta1,beta2))

  train_hist={}
  train_hist['D_losses']=[]
  train_hist['G_losses']=[]
  train_hist['per_epoch_ptimes']=[]
  train_hist['total_ptime']=[]
  epoch_start=0
  epoch_end=epoch_start+train_epoch

#training
for epoch in range(epoch_start,epoch_end):
  D_losses=[]
  G_losses=[]
  epoch_start_time=time.time()
  num_iter=0
  for (real_image, _) in train_dataloader:
    writer.add_image('Real Samples',real_image[0].cpu())
    inp_noise=torch.randn(batch_size, nz, 1, 1, device=device)
    inp_noise,real_image=Variable(inp_noise.to(device)),Variable(real_image.to(device))
    D.zero_grad()

    D_result=D(real_image).squeeze()
    
    D_real_loss=BCE_loss(D_result,Variable(torch.ones(D_result.size()).to(device)))
    
    G_result=G(inp_noise)
    G_result=G_result.detach()
    D_result=D(G_result).squeeze()
    D_fake_loss=BCE_loss(D_result,Variable(torch.zeros(D_result.size()).to(device)))
    
    D_train_loss=(D_real_loss +D_fake_loss)*0.5
    D_train_loss.backward()
    D_optimizer.step()
    train_hist['D_losses'].append(float(D_train_loss))

    D_losses.append(float(D_train_loss))
    #D_losses.append(float(0)) # if the discriminator training hast to be stopper

    #training generator
    G.zero_grad()

    G_result=G(inp_noise)
    G_result=G_result.detach()
    D_result=D(G_result).squeeze()
    G_train_loss=BCE_loss(D_result, Variable(torch.ones(D_result.size()).to(device))) #+ L1_lambda*L1_loss(G_result,real_image)
    G_train_loss.backward()
    G_optimizer.step()

    train_hist['G_losses'].append(float(G_train_loss))
    G_losses.append(float(G_train_loss))
    
    num_iter+=1

  torch.save({
            'epoch': epoch,
            'model_state_dict': G.state_dict(),
            'optimizer_state_dict': G_optimizer.state_dict(),
            'train_hist': train_hist
            }, model_dir+'generator_param.pkl')

  torch.save({
            'model_state_dict': D.state_dict(),
            'optimizer_state_dict': D_optimizer.state_dict(),
            },model_dir+'discriminator_param.pkl')

  epoch_end_time=time.time()
  per_epoch_ptime=epoch_end_time-epoch_start_time
  print('[%d/%d] - ptime: %.2f, loss_d: %.3f, loss_g: %.3f' % ((epoch + 1), train_epoch, per_epoch_ptime, torch.mean(torch.FloatTensor(D_losses)),
                                                              torch.mean(torch.FloatTensor(G_losses))))
  fixed_p =  output_dir  + str(epoch + 1) + '.png'
  
  vutils.save_image(G(save_noise).detach(),
                    fixed_p,
                    normalize=True)
  
  num_info = { 'Discriminator loss': torch.mean(torch.FloatTensor(D_losses)), 'Generator loss': torch.mean(torch.FloatTensor(G_losses)) }  
  fake_to_show=G(save_noise).detach()

  
  #tensorboard logging
  writer.add_scalars('Loss',num_info,epoch)
  writer.add_image('Fake Samples',fake_to_show[0].cpu())
  train_hist['per_epoch_ptimes'].append(per_epoch_ptime)
  if epoch%30==0:
    fid_score=fid_model.compute_fid(real_image,G_result)
    print("FID score",fid_score)
    writer.add_scalar('FID Score',fid_score,epoch)
  
  

end_time=time.time()
total_ptime=end_time-start_time
train_hist['total_ptime'].append(total_ptime)
print("Avg one epoch ptime: %.2f, total %d epochs ptime: %.2f" % (torch.mean(torch.FloatTensor(train_hist['per_epoch_ptimes'])), train_epoch, total_ptime))
writer.close()

with open(report_dir+'train_hist.pkl', 'wb') as f:
    pickle.dump(train_hist, f)

show_train_hist(train_hist, save=True, path=report_dir + 'train_hist.png')


writer.close()