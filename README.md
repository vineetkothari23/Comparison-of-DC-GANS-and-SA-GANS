# Comparison-of-DC-GANS-and-SA-GANS
Based on CIFAR Dataset in Pytorch.

## About:
A generalized pytorch implementation for image generation task for any dataset, using Deep convolutional generative adversarial network and Self attention generative adversarial network.
The cSAW GAN stands for conditional SAGAN implemented with Wasserstein loss.

Following are the papers:
This paper on [**DC Gans**](https://arxiv.org/abs/1511.06434) was published in November 2015 by **authors Alec Radford, Luke Metz, Soumith Chintala**
This paper on [**SA Gans**](https://arxiv.org/abs/1805.08318) was published in June 2019 by **authors Han Zhang, Ian Goodfellow, Dimitris Metaxas, Augustus Odena.**

#### Citations:
- DC Gan: https://arxiv.org/abs/1511.06434
- SA Gan: https://arxiv.org/abs/1805.08318
- Wasserstein loss: https://arxiv.org/pdf/1701.07875.pdf
- Frechet Inception score: https://arxiv.org/pdf/1706.08500.pdf

## Broader look out:
- DC GAN: The DC Gan is an **unsupervised deep convolutional GAN** which generates images based on random noise.
- cSAW GAN: The conditional Self-Attention Generative Adversarial Network (cSAGAN) generates images allowing **attention-driven, long-   range dependency modeling** using **Wasserstein loss**.
- The comparison of the two algorithms is done using various metrics, but the major metric is [**Frechet Inception score**].

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


## Documentation

### Theoritical details
- [Self Attention GANS: Improvisation in smaller parts and quality in image generation](documentation/sagan.md)
- [Spectral normalization: ](documentation/spectral.md)
- [Conditional batch normalization](documentation/conditional_bn.md)
- [FID metric: Better way to compare similarity of two images](documentation/fid.md)

### Implementation
- [Modules](documentation/modules.md)
- [Data](documentation/dataset.md)
- [Architecture](documentation/CONTRIBUTING.md)
- [Code structure](docs/CONTRIBUTING.md)
- [Class activation mappings](documentation/cam.md)
- [Distributed training](docs/CONTRIBUTING.md)
- [Docker](docs/CONTRIBUTING.md)'
- [Results](documentation/results.md)

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




