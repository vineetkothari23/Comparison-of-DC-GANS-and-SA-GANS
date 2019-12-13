# Spectral Normalization

**Spectral normalization SN constraints the Lipschitz constant of the convolutional filters.**

## Why is spectral normalization used?

Spectral norm was used as a way to stabilize the training of the discriminator network.

Prior work has shown that regularized discriminators make the GAN training slower. For this
reason, some workarounds consist of uneven the rate of update steps between the generator and
the discriminator.

To solve the problem of slow learning and imbalanced update steps, Heusel et al introduced the
two-timescale update rule (TTUR) in the GAN training. 

It consists of providing **different learning rates for optimizing the generator and discriminator**.

The discriminator trains with a learning rate 4 times greater than G - 0.004 and 0.001
respectively. A larger learning rate means that the discriminator will absorb a larger part of the
gradient signal. Hence, a higher learning rate eases the problem of slow learning of the
regularized discriminator. Also, this approach makes it possible to use the same rate of updates
for the generator and the discriminator. In fact, we use a 1:1 update interval between generator
and discriminator.
