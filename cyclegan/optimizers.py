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

"""Wrapped Optax optimizers for use in a CycleGAN training loop.

This module contains examples of Optax optimizers that have been wrapped in
such a way to make it easy to explore options that go beyond what was used
in the original implementation of CycleGAN. For example, the use of
optax.multitransform makes it possible to specify different optimizers (such
as adamw, sgd, etc.) for each sub-network in a generator or discriminator.

Further, in cases where optax.inject_hyperparams is used, each of the
hyper parameters (such as b1, b2) of target optimizers can be assigned a
schedule instead of a scalar.
"""

import gin
import optax
from flax.core import FrozenDict


def map_prefix_fn(*labels):
  """Return a partial prefix of a Pytree in which the nodes terminate at the
  given labels and with node values replaced by their labels.

  This is intended to be used to produce a mapping of labels for
  optax.multi_transform(). It will return a function that will recursively
  scan a given Pytree for keys that match input labels and then replace the
  associated values with the labels. Branches that do not contain matched
  labels are removed. optax.multi_transform will use the resulting mapping to
  tell which parts of the Pytree to apply specific values to
  schedules to.
  """

  def prefix_fn(d):
    if isinstance(d,(dict, FrozenDict)):
      #recursively map labels to node values for nodes that match
      #the requested labels
      mapped = {
        k:(k if k in labels else prefix_fn(v))
        for k,v in d.items()
      }
      #trim any branches that didn't terminate in a requested label
      trimmed = {k:v for k,v in mapped.items() if v is not None}
      #only return something if at least one node mapped to a requested label
      if len(trimmed)>0:
        return type(d)(trimmed)
  return prefix_fn


@gin.configurable
def generator_decay(
  lr_schedule,
  b1,
  b2,
  clipping=None,
  weight_decay=0.0001
):
  """Configure adam optimizers with weight decay and adaptive gradient clipping
  for the generators. """

  opt_f = optax.inject_hyperparams(optax.adamw)(
      learning_rate=lr_schedule, b1=b1, b2=b2, weight_decay=weight_decay
  )
  opt_g = optax.inject_hyperparams(optax.adamw)(
    learning_rate=lr_schedule, b1=b1, b2=b2, weight_decay=weight_decay
  )

  opt = optax.multi_transform(
    {'net_f': opt_f, 'net_g': opt_g},
    map_prefix_fn('net_f','net_g')
  )

  if clipping:
    opt = optax.chain(optax.adaptive_grad_clip(clipping=clipping),opt)

  return opt


@gin.configurable
def discriminator_decay(
  lr_schedule,
  b1,
  b2,
  clipping=None,
  weight_decay=0.0001
):
  """Configure adam optimizers with weight decay and adaptive gradient clipping
  for the discriminators. """

  opt_x = optax.inject_hyperparams(optax.adamw)(
      learning_rate=lr_schedule, b1=b1, b2=b2, weight_decay=weight_decay
  )
  opt_y = optax.inject_hyperparams(optax.adamw)(
    learning_rate=lr_schedule, b1=b1, b2=b2, weight_decay=weight_decay
  )

  opt = optax.multi_transform(
    {'net_x': opt_x, 'net_y': opt_y},
    map_prefix_fn('net_x','net_y')
  )
  if clipping:
    opt = optax.chain(optax.adaptive_grad_clip(clipping=clipping),opt)

  return opt


@gin.configurable
def generator_original(
  lr_schedule,
  b1=0.5,
  b2=0.999
):

  opt_g = optax.inject_hyperparams(optax.adam)(
      learning_rate=lr_schedule, b1=b1, b2=b2
  )

  opt_f = optax.inject_hyperparams(optax.adam)(
      learning_rate=lr_schedule, b1=b1, b2=b2
  )

  opt = optax.multi_transform(
    {'net_g': opt_g, 'net_f': opt_f},
    map_prefix_fn('net_g','net_f')
  )

  return opt


@gin.configurable
def discriminator_original(
  lr_schedule,
  b1=0.5,
  b2=0.999
):
  opt_x = optax.inject_hyperparams(optax.adam)(
      learning_rate=lr_schedule, b1=b1, b2=b2
  )

  opt_y = optax.inject_hyperparams(optax.adam)(
      learning_rate=lr_schedule, b1=b1, b2=b2
  )

  opt = optax.multi_transform(
    {'net_x': opt_x, 'net_y': opt_y},
    map_prefix_fn('net_x','net_y')
  )

  return opt


#make a wrapper that can be used to pass configured optimizers back to a caller
@gin.configurable
def configured_pre(assets):
  return assets
