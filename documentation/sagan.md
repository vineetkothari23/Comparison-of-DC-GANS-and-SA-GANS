# Self attention generative adversarial networks

## About
This paper was authored by Han Zhang, Ian Goodfellow, Dimitris Metaxas and Augustus Odena  in June 2019.
The paper discusses several techniques self-attention, spectral normalization, conditional batch normalization, projection discriminator etc.
Citation: https://arxiv.org/abs/1805.08318


## What is SAGAN?
Self attention GANs are a type of generative adversarial network allowing attention-driven, long-range dependency
modeling for image generation tasks.

## Side effects of GANs
- Generic generative adversarial networks fail to capture geometrical/ structural patterns consistency accross classes.
  E.g. For generation of an image of a dog, it might be able to generate the fur texture pretty good as compared to its legs.
- Deep convolutional networks are capable of capturing long range dependencies but the optimization algorithms fails to coordniate the dependencies amongst multiple layers.
- Long and deep networks can solve the problem but are computationally inefficient.

#### SOLUTION: SAGAN!

### Examples:

![Sagan example](https://miro.medium.com/max/2574/1*g-SICHy5Yc0IT10Usl5IKg.png)

## Goal of SA GAN:
- To capture long range dependencies. E.g. The relations amongt the corners of the images also must be emphasized without being constrained by the kernel size.
- Must be computationally and statistically efficient.

## Design detail

1. Replace the last fully connected layer into 1 by 1 convolutional layer.
The output of the last layer to be passed into the convolutional layer. Hence we split into three feature maps.

Let the three maps be f(x), g(x) and h(x) for input x.
```
no_channels,height, width=x.size
kernel_size=no_channels//8
f=conv(x,kernel_size,1,1,name='f')
g=conv(x,kernel_size,1,1,name='g')
h=conv(x,no_channels,1,1,name='h')
```
2. Reshape each of the 4d Tensor output of convolutional layer into 3d tensor by merging the height and weight into one.  

![Self attention process](https://miro.medium.com/max/1600/1*oIAw_f4Zw6iJfFU6TbeoaA.jpeg)

```
f=torch.reshape(f,[-1,height*width,kernel_size])
g=torch.reshape(g,[-1,height*width,kernel_size])
h=torch.reshape(h,[-1,height*width,no_channels])

```

3.Now we refine each spatial location output with an extra term 'o' computed by self attention mechanishm
x is the original layer output and y in the new output

![attention formula](https://miro.medium.com/max/1852/1*QEmsHNTXX0jzqYEb55kckQ.jpeg)

![Attention](https://miro.medium.com/max/1637/1*idmLYTU4Ws3zo5yq73zlUg.jpeg)

```
def attention(f,g,h):
  dot_product=torch.matmul(f,g,transpose_b=True)
  attn_map=torch.nn.softmax(dot_product)
  x=torch.matmul(attn_map,h)
  return x
```

The output of the self attention is:
1. Attention Map B
2. Self Attention output.

4. Overall the activation map B1 from f(x) and g(x) is merged with h(x) to output self attention output.

The attention map acts as a mask to indicate the spatial location impact of different parts of the image.

```
o=attention(f,g,h)
o=torch.reshape(o,[-1, height, width, no_channels])
o=conv(o,no_channels,1,1)
gamma=torch.autograd.Variable([1])
x=gamma*o+x

return x
```
## Loss
SAGAN uses hinge or Wasserstein loss
![Hinge loss](https://miro.medium.com/max/1502/1*6LhILVo1m32Hf8Sinn-oEg.png)

## References
[Paper: Self-Attention Generative Adversarial Networks](https://arxiv.org/pdf/1805.08318.pdf)
[Techniques in Self-Attention Generative Adversarial Networks](https://medium.com/towards-artificial-intelligence/techniques-in-self-attention-generative-adversarial-networks-22f735b22dfb)
[GAN — Self-Attention Generative Adversarial Networks (SAGAN)](https://medium.com/@jonathan_hui/gan-self-attention-generative-adversarial-networks-sagan-923fccde790c)
