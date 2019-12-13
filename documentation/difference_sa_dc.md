## Problem with deep convolutional GANs fail?
- DC GANs have difficulty in learning the image distributions of diverse multi-class datasets like Imagenet.
- There is difficulty in modeling some image classes than others when trained on multi-class datasets.
- DCGANs could easily generate images with a simpler geometry like Ocean, Sky etc. but failed on
images with some specific geometry like dogs, horses and many more.

## Why does DC GAN fail?

> This problem is arising because the **convolution is a local operation whose receptive field
depends on the spatial size of the kernel.**

![DCGAN architecture](https://gluon.mxnet.io/_images/dcgan.png)

- We can make the spatial size of kernel bigger but it would **decrease computational
efficiency** achieved by smaller filters and make the operation slow.

## The good part of Self attention GAN

> Self Attention **keeps the computational efficiency and having a large receptive field** at the same
time possible. 

![SAGAN architecture](https://miro.medium.com/max/2204/1*H29pojIh1fvscvX04gF2Xg.png)

- It helps create a balance between efficiency and **long-range dependencies** by
utilizing the famous mechanism from NLP called **attention**.

For more details on [Self attention](https://github.com/MicroprocessorX069/Comparison-of-DC-GANS-and-SA-GANS/blob/master/documentation/sagan.md).
