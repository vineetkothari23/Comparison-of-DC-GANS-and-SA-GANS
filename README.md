# Comparison-of-DC-GANS-and-SA-GANS
Based on CIFAR Dataset in Pytorch.

## About:
This is a pytorch implementation of classification of chest X-rays as infected with pneumonia or not. This is examined by radiologists manually for air cavities and lumps. The model also outputs a heat map indicating the areas in the xray dominant in leading to the prediction.
The paper [CheXNet: Radiologist-Level Pneumonia Detection on Chest X-Rays with Deep Learning (https://arxiv.org/abs/1711.05225)] was published in Dec 2017 by Pranav Rajpurkar and Jeremy Irvin.

Citations:
https://arxiv.org/abs/1711.05225

## Broader look out:
The 121 layered convolutional neural network has acheived the state of the art detection of pneumonia over manual detection by human radiologist, compared by F1 scores. It uses a dataset of 100000 Chest X-rays 14 dataset annotated by four radiologists.

### Architecture

### Prerequisites
### Usage

#### Installation
#### Requirements
- Python 3.5+
- PyTorch 0.3.0
 
#### Directory structure
The directory structure is followed as 
```
.
├── ...
├── version_no                    # Version of different models and training process
│   ├── model          # saved model checkpoint files
│   ├── report         # reporting of final training, validation loss and other metrics
│   └── output          # Output directory
│       └── epoch                    # Storing the training epoch images
├── data                    # Dataset of images (Optional)
├── res                # Resources directory
│    └── Helvetica                    # Font file to generate paired images for training (optional) 
└── ...
```

#### Train/ test
1. Clone the repository
```
$ git clone https://github.com/MicroprocessorX069/Generalized-pix2pix-GAN-API.git
$ cd Generalized-pix2pix-GAN-API
```
3. Train
(i) Train
```
$ python python train.py --root_dir "./" --version "1.0" --batch_size 64 --input_width 32 --input_height 32 
```
(ii) Test
'''
$ python python test.py --root_dir "./" --version "1.0" 
'''
4. Enjoy the results
```
$ cd output/epoch
or
$ cd report
```

#### Using a pretrained model weights
Download the model weights as .ckpt file in "./model/" and hit the same commands to train and test with the correct root directory.

## Results
![Training gif](https://github.com/MicroprocessorX069/Generalized-pix2pix-GAN-API/blob/master/training_process.GIF)
## Implementation details
- [Theoritical details](docs/CONTRIBUTING.md)
- [Modules](docs/CONTRIBUTING.md)
- [Data](docs/CONTRIBUTING.md)
- [Architecture](docs/CONTRIBUTING.md)
- [Code structure](docs/CONTRIBUTING.md)
- [Class activation mappings](documentation/cam.md)
- [Distributed training](docs/CONTRIBUTING.md)
- [Docker](docs/CONTRIBUTING.md)

## Documentation

### Theoritical details
- [Self Attention GANS: Improvisation in smaller parts and quality in image generation](documentation/sagan.md)
- [Spectral normalization: ](documentation/spectral.md)
- [Conditional batch normalization](documentation/conditional_bn.md)
- [FID metric: Better way to compare similarity of two images](documentation/fid.md)

### Issues
- [Mode collapse](documentation/regularization.md)
- [Modules](docs/CONTRIBUTING.md)
- [Data](docs/CONTRIBUTING.md)
- [Architecture](docs/CONTRIBUTING.md)
- [Code structure](docs/CONTRIBUTING.md)
- [Class activation mappings](documentation/cam.md)
- [Distributed training](docs/CONTRIBUTING.md)
- [Docker](docs/CONTRIBUTING.md)

## Related projects
## Acknowledgements




