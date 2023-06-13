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

"""Callback functions that can be executed by clu.PeriodicAction during a
training run"""

import io

import jax
from jax import numpy as jnp
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from clu import asynclib


def img_to_int(img):
  """Drop the batch dimension and convert from range [-1,1] to [0,255]"""

  img = (img+1.)*127.5
  img = np.clip(img, 0 , 255)
  return img.astype(np.int16)


def prep_scores(scores, midpoint=0.5):
  """Express discriminator scores as a distance from the midpoint between target
  scores for 'real' and 'fake' patches. Also squeeze the scores into the range
  [-1,1] so that they will have similar scale to image outputs. """
  frame = jnp.tanh(scores - midpoint)
  return frame


def tile_images(labels, imgs, file_format='png'):
  """Use matplotlib to tile an array of plots into a single image
  Args:
      labels: Array of column labels for the tiled images.
      imgs: Flattened array of images to tile in the order of top-left to
          bottom right.
      file_format: string of the desired file format to use for saved images.

  Returns:
      tiled_img: Array, a single image made up of smaller images arranged in a
          grid."""

  px = 1/plt.rcParams['figure.dpi']  # pixel in inches
  fig = plt.figure(figsize=(270*5*px,280*px))

  for i, img in enumerate(imgs):
    img = img_to_int(img[0])
    ax = plt.subplot(1, 5, i + 1,xticks=[],yticks=[])
    ax.set_title(label=labels[i], fontdict={'fontsize':12})
    ax.imshow(img, cmap=plt.cm.RdGy, vmin=0, vmax=255)

  fig.tight_layout(rect=(0,0,1,1))
  plt.subplots_adjust(top=.95)

  #convert the plot to an image in memory
  buffer = io.BytesIO()
  plt.savefig(buffer, format=file_format)
  plt.close(fig)

  #convert the image back to an array
  tiled_img = tf.image.decode_png(buffer.getvalue(), channels=4)

  return tiled_img


class GanImageCallback():
  """a Callback class that will produce a tiled image of outputs for
  each of the networks in a CycleGan model. This can be used to log images to
  Tensorboard during a training run."""

  def __init__(self, generator, discriminator):

    self.labels =  ('real','fake','score','identity','cycled')

    def generate_fn(real_imgs, state):
      real_x, real_y = real_imgs
      (
        identity_x, identity_y,
        fake_y, fake_x,
        cycled_x, cycled_y
      ) = generator.apply({'params':state.params_gen}, real_imgs )

      score_f, score_g = discriminator.apply(
          {'params':state.params_disc},(fake_x, fake_y)
      )
      score_f = prep_scores(score_f)
      score_g = prep_scores(score_g)

      images = {
      'net_G':(real_x, fake_y, score_g, identity_x, cycled_x),
      'net_F':(real_y, fake_x, score_f, identity_y, cycled_y)
      }
      return images

    self.generate_fn = jax.jit(generate_fn)

  def __call__(self, input_batches, state, writer):
    for batch_num, batch in enumerate(input_batches):
      images = self.generate_images(batch, state)
      #let the cpu handle the rest asynchronously
      self._write_images(state.step, batch_num, images, writer)

  def generate_images(self, real_imgs, state):
    images = self.generate_fn( real_imgs, state)
    return jax.device_get(images)

  @asynclib.Pool(max_workers=1)
  def _write_images(self, step, batch_num, images, writer):
    images = {
      f'{k}/{batch_num+1}': tile_images(self.labels, v)
      for k,v
      in images.items()
    }
    writer.write_images(step, images)
