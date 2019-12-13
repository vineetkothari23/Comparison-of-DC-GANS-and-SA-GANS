import torch
import torchvision.datasets as dsets
from torchvision import transforms


class Data_Loader():
    def __init__(self, train, dataset, image_size, batch_size, shuf=True):
        self.dataset = dataset
        self.imsize = image_size
        self.batch = batch_size
        self.shuf = shuf
        self.train = train

    def transform(self, resize, totensor, normalize, centercrop):
        options = []
        if centercrop:
            options.append(transforms.CenterCrop(160))
        if resize:
            options.append(transforms.Resize((self.imsize,self.imsize)))
        if totensor:
            options.append(transforms.ToTensor())
        if normalize:
            options.append(transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))
        transform = transforms.Compose(options)
        return transform

      
    def loader(self):
      transform = transforms.Compose(
          [transforms.Resize((self.imsize,self.imsize)),
           transforms.ToTensor(),
           transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

      if self.train:
        dataset_full = torchvision.datasets.CIFAR10(root='./data', train=True,
                                                download=True, transform=transform)
        train_split=0.8
        train_size=int(train_split*len(dataset_full))
        val_size=len(dataset_full)-train_size
        trainset, valset=torch.utils.data.random_split(dataset_full,[train_size,val_size])
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                                  shuffle=True, num_workers=2)

        return trainloader
      else:
        testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                                download=True, transform=transform)
        testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                                  shuffle=True, num_workers=2)

        return testloader
    def len(self):
      return len(self.loader())
    