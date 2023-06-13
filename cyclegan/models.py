# MIT License
#
# Copyright (c) 2023 Jacobus Louw
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""CycleGAN models for image-to-image translation"""

import functools
from typing import Optional, Callable, Any
from collections import defaultdict

import gin
import jax
import jax.numpy as jnp
from flax import linen as nn


Array = Any
# Instance Normalization Layer (https://arxiv.org/abs/1607.08022).
# flax.linen.GroupNorm reduces to the equivalent of instance normalization
# when used with a group size of one.
InstanceNorm = functools.partial(
    nn.GroupNorm,
    epsilon=1e-5,
    num_groups=None,
    group_size=1,
    dtype=jnp.float32
)


def zero_pad_hw(y: Array, left_top: int=1, right_bottom: int=1) -> Array:
  """Pad the height and width dimensions of an input with zeros.
  Args:
    y: Array input with shape (batch, height, width, channels).
    left_top: Integer amount of padding to add.
    right_bottom: Integer amount of padding to add.
  Returns:
    y: Array with pads
  """
  y = jnp.pad(
      y,
      pad_width=(
        (0, 0),
        (left_top, right_bottom),
        (left_top, right_bottom),
        (0, 0)
      ),
      mode='constant',
      constant_values=((0, 0), (0, 0), (0, 0), (0, 0))
  )
  return y


def reflect_pad_hw(
  y: Array, left_top: Optional[int]=1, right_bottom: int=1
) -> Array:
  """Pad the height and width dimensions of an input using reflection padding.
  Args:
    y: Array input with shape (batch, height, width, channels).
    left_top: Integer amount of padding to add.
    right_bottom: Integer amount of padding to add.
  Returns:
    y: Array with pads
    """
  y = jnp.pad(
      y,
      pad_width=((0,0),(left_top,right_bottom),(left_top,right_bottom),(0,0)),
      mode='reflect'
  )
  return y


class DiscDownBlock(nn.Module):
  """Discriminator down-sample block"""
  features: int
  n_kernel: int
  strides: int = 2
  padding: str ='SAME'
  use_norm: bool = True
  norm_fn: Callable = InstanceNorm
  bias_init: Callable = jax.nn.initializers.zeros
  kernel_init: Callable = jax.nn.initializers.normal(stddev=0.02)

  @nn.compact
  def __call__(self, x):
    x = nn.Conv(
        features=self.features,
        kernel_size=(self.n_kernel, self.n_kernel),
        strides=self.strides,
        padding=self.padding,
        use_bias=True,
        kernel_init=self.kernel_init
    )(x)

    if self.use_norm:
      x = self.norm_fn()(x)

    x = nn.leaky_relu(x, negative_slope=0.2)

    return x


@gin.register
class Discriminator(nn.Module):
  """Modified PatchGan discriminator model (https://arxiv.org/abs/1611.07004).

  This is a Flax version of the pix2pix discriminator from here:
  https://github.com/tensorflow/examples/blob/master/tensorflow_examples/models/pix2pix/pix2pix.py

  The model expects an input tensor with shape (batch_size,256,256,3) where each
  item in the batch is a normalized image of size 256 x 256. It produces an
  output tensor with shape (batch_size,30,30,1). Each cell in the 30x30 output
  array has a receptive field of 70x70 pixels in the corresponding input image.
  The loss functions that will be used with CycleGan will interpret each such
  item as a score and will use it to judge if the 70x70 patches in the input
  images were from real or fake images.

  Note, this code excludes functionality of the tensorflow version that are
  not used by CycleGan.

  Approx 2.8 million parameters"""
  norm_fn: Callable = InstanceNorm

  @nn.compact
  def __call__(self, x):
    conv = functools.partial(
      nn.Conv,
      strides=1,
      padding='VALID',
      use_bias=True,
      kernel_init=jax.nn.initializers.normal(stddev=0.02)
    )

    #comments indicate:
    # (batch, height, width, channels), receptive_field, input_stride
    x = DiscDownBlock(64, 4, use_norm=False)(x)# (b, 128, 128, 64), rf=4, st=2
    x = DiscDownBlock(
        128, 4, norm_fn=self.norm_fn
    )(x)# (b, 64, 64, 128) rf=10, s=4
    x = DiscDownBlock(
        256, 4, norm_fn=self.norm_fn
    )(x)# b, 32, 32, 256) rf=22, s=8
    x = zero_pad_hw(x)# (b,34,34,256)
    x = conv(512, (4, 4))(x)# (b, 31, 31, 512) rf=46, s=8
    x = self.norm_fn()(x)
    x = nn.leaky_relu(x, negative_slope=0.2)
    x = zero_pad_hw(x)# (b, 33, 33, 512)
    x = conv(1, (4, 4),name='Conv_Head')(x)# (b, 30, 30, 1) rf=70, s=8

    return x


@gin.register
class UNetGenerator(nn.Module):
  """Modified u-net generator model (https://arxiv.org/abs/1611.07004).

  This is a Flax version of the pix2pix unet discriminator from here:
  https://github.com/tensorflow/examples/blob/master/tensorflow_examples/models/pix2pix/pix2pix.py

  It is included in case you want to compare results with the tensorflow
  CycleGan tutorial found here:
  https://www.tensorflow.org/tutorials/generative/cyclegan
  """
  eval_flag: Optional[bool] = None
  dropout: float = 0.5

  @nn.compact
  def __call__(self, x, eval_flag=None):
    eval_flag = nn.merge_param('eval_flag', self.eval_flag, eval_flag)
    norm = InstanceNorm

    name_counts=defaultdict(int)

    def add_count(name):
      name_counts[name] += 1
      return name + f'_{name_counts[name]}'

    shortcuts = []

    def downsample(y,n_filters, n_kernel, add_shortcut=True):
      y = nn.Conv(
          n_filters,
          (n_kernel, n_kernel),
          strides=2,
          padding='SAME',
          use_bias=False,
          kernel_init=jax.nn.initializers.normal(stddev=0.02),
          name=add_count(f'downsample_{n_filters}_{n_kernel}')
      )(y)
      y = norm()(y)
      y = nn.leaky_relu(y,negative_slope=0.2)

      if add_shortcut:
        shortcuts.append(y)
      return y

    def upsample(
      y, n_filters, n_kernel,
      apply_dropout=False, use_shortcut=True, activation=nn.relu
    ):
      if use_shortcut:
        s = shortcuts.pop()
        y = jnp.concatenate((y,s),axis=-1)

      y = nn.ConvTranspose(
          n_filters,
          (n_kernel, n_kernel),
          strides=(2,2),
          padding='SAME',
          use_bias=False,
          kernel_init=jax.nn.initializers.normal(stddev=0.02),
          name=add_count(f'upsample_{n_filters}_{n_kernel}')
      )(y)
      y = norm()(y)
      y = activation(y)

      if apply_dropout:
        y = nn.Dropout(rate=self.dropout, deterministic=eval_flag)(y)
      return y

    x = downsample(x, 64, 4)  #(b, 128, 128, 64)
    x = downsample(x, 128, 4)  #(b, 64, 64, 128)
    x = downsample(x, 256, 4)  # b, 32, 32, 256)
    x = downsample(x, 512, 4)  #(b, 16, 16, 512)
    x = downsample(x, 512, 4)  #(b, 8, 8, 512)
    x = downsample(x, 512, 4)  #(b, 4, 4, 512)
    x = downsample(x, 512, 4)  #(b, 2, 2, 512)
    x = downsample(x, 512, 4, add_shortcut = False)  #(b, 1, 1, 512)
    x = upsample(
        x, 512, 4, apply_dropout=True, use_shortcut=False
    ) #(b, 2, 2, 512)
    x = upsample(x, 512, 4, apply_dropout=True)  #(b, 4, 4, 512)
    x = upsample(x, 512, 4, apply_dropout=True)  #(b, 8, 8, 512)
    x = upsample(x, 512, 4)  #(b, 16, 16, 512)
    x = upsample(x, 256, 4)  #(b, 32, 32, 256)
    x = upsample(x, 128, 4)  #(b, 64, 64, 128)
    x = upsample(x, 64, 4)  #(b, 128, 128, 64)
    x = upsample(x, 3, 4,activation=nn.tanh) #(b, 256, 256, 3)

    return x


class ResnetDownBlock(nn.Module):
  """The downsampling component of a resnet generator."""

  features: int
  n_kernel: int
  strides: int = 2
  padding: str = 'SAME'
  norm_fn: Callable = InstanceNorm
  kernel_init: Callable = jax.nn.initializers.normal(stddev=0.02)

  @nn.compact
  def __call__(self, x):
    x = nn.Conv(
        features=self.features,
        kernel_size=(self.n_kernel, self.n_kernel),
        strides=self.strides,
        padding=self.padding,
        use_bias=True,
        kernel_init=self.kernel_init
    )(x)

    x = self.norm_fn()(x)
    x = nn.relu(x)

    return x


class ResnetUpBlock(nn.Module):
  """The upsampling component of a resnet generator."""

  features:int
  n_kernel:int
  n_strides:int = 2
  padding: str = 'SAME'
  norm_fn: Callable = InstanceNorm
  use_bias: bool = True
  kernel_init:Callable = jax.nn.initializers.normal(stddev=0.02)

  @nn.compact
  def __call__(self, x):
    x = nn.ConvTranspose(
        features=self.features,
        kernel_size=(self.n_kernel, self.n_kernel),
        strides=(self.n_strides,self.n_strides),
        padding=self.padding,
        use_bias=self.use_bias,
        kernel_init=self.kernel_init
    )(x)

    x = self.norm_fn()(x)
    x = nn.relu(x)

    return x


class ResidualBlock(nn.Module):
  """Residual block of a resnet generator."""

  features:int
  pad_fn:Optional[Callable]=reflect_pad_hw
  norm_fn:Optional[Callable]=InstanceNorm
  kernel_init:Optional[Callable]=jax.nn.initializers.normal(stddev=0.02)

  @nn.compact
  def __call__(self, x):
    conv = functools.partial(
        nn.Conv,
        features=self.features,
        kernel_size=(3, 3),
        strides=1,
        padding='VALID',
        use_bias=True,
        kernel_init=self.kernel_init
    )
    #comments show layer output shapes assuming the input x is shaped:
    # (batch, height, width, channels)

    y = self.pad_fn(x) #(b,h+2,w+2,c)
    y = conv()(y) #(b,h,w,c)
    y = self.norm_fn()(y)
    y = nn.relu(y)
    y = self.pad_fn(y) #(b,h+2,w+2,c)
    y = conv()(y) #(b,h,w,c)
    y = self.norm_fn()(y)

    y = y + x #residual connection

    return y


@gin.register
class ResnetGenerator(nn.Module):
  """ResnetGenerator from the original Pytorch version of CycleGan, which
  can be found at https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix

  Some of the functionality of the original code that was not used in the
  published model has been excluded from this version for the sake of
  simplicity.

  Arguments:
    residuals: int number of residual blocks
    features: int number of filters in the last non-rgb layer of the generator
    pad_fn: callable function that will pad a given input when called.
      The function must accept an input signature of:
        pad_fn(x:DeviceArray, left_top:int, right_bottom:int)
    norm_fn: callable normalization function. The default is instance norm as
      per the paper.
    kernel_init: callable initializer for convolution kernel weights.
  """

  residuals: int = 9
  features: int = 64
  norm_fn: Callable = InstanceNorm
  kernel_init: Callable = jax.nn.initializers.normal(stddev=0.02)

  @nn.compact
  def __call__(self, x):
    down_block = functools.partial(
        ResnetDownBlock,
        norm_fn=self.norm_fn,
        kernel_init=self.kernel_init
    )
    residual_block = functools.partial(
        ResidualBlock,
        norm_fn=self.norm_fn,
        kernel_init=self.kernel_init
    )
    up_block = functools.partial(
        ResnetUpBlock,
        norm_fn=self.norm_fn,
        kernel_init=self.kernel_init
    )

    # comments show layer output shapes assuming the input x
    # is shaped (batch,256,256,3)
    x = reflect_pad_hw(x, 3, 3) #(b,262,262,3)
    x = down_block(
        self.features, 7, strides=1, padding='VALID'
    )(x) #(b,256,256,features)

    #down-sample
    x = down_block(self.features*2, 3)(x)  #(b, 128, 128, features*2)
    x = down_block(self.features*4, 3)(x)  #(b, 64, 64, features*4)

    #residual blocks
    for _ in range(self.residuals):
      x = residual_block(self.features*4)(x) #(b, 64, 64, features*4)

    #up-sample
    x = up_block(self.features*2, 3)(x)  #(b, 128, 128, features*2)
    x = up_block(self.features, 3)(x)  #(b, 256, 256, features)

    #final rgb
    x = reflect_pad_hw(x, 3, 3) #(b,262,262,3)
    x = nn.Conv(
        3, (7, 7),
        strides=1,
        padding='VALID',
        use_bias=True,
        kernel_init=self.kernel_init
    )(x) #(b,256,256,3)

    x = nn.tanh(x)

    return x


@gin.register
class CycleGenerator(nn.Module):
  """a CycleGan generator that makes use of two sub-generators to
  produce fake, identity and cycled images. This is convenient to have for
  the training loop, but not needed for exploring the final results."""

  base_model: Optional[Callable]=ResnetGenerator

  def setup(self):
    self.net_g = self.base_model()
    self.net_f = self.base_model()

  @nn.compact
  def __call__(self, inputs ):
    x,y = inputs

    #translate the images
    fake_y = self.net_g(x)
    fake_x = self.net_f(y)
    identity_x = self.net_f(x)
    identity_y = self.net_g(y)
    cycle_x = self.net_f(fake_y)
    cycle_y = self.net_g(fake_x)

    return (
        identity_x, identity_y,
        fake_y, fake_x,
        cycle_x, cycle_y
    )


@gin.register
class CycleDiscriminator(nn.Module):
  """A discriminator that makes use of two sub-networks to
  produce score generated images."""

  base_model: Optional[Callable]=Discriminator

  def setup(self):
    self.net_x = self.base_model()
    self.net_y = self.base_model()

  def __call__(self, inputs):
    x,y = inputs
    scores_x = self.net_x(x)
    scores_y = self.net_y(y)
    return (scores_x, scores_y)


@gin.configurable
def configured(assets: Any):
  """A configurable pass-through function that enables gin registered objects
  to be used in a python script after they have been configured in a gin file.

  This allows you to configure flax models that have only been decorated with
  gin.register and then use them in python as if they were decorated with
  gin.configurable. Simply assign the configured assets to this function in a
  gin file and then call this function without arguments in python to retrieve
  the configured assets in the same order.

  It is probably a bad idea to decorate the flax classes with
  @gin.external_configurable directly because doing so will replace the
  objects' __call__ methods with gin decorated versions. This function provides
  a safe alternative.
  """
  return assets
