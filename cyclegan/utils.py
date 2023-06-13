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

"""Generic functions that may be useful in various situations"""

import gin
import jax
from jax import tree_util
import numpy as np

@gin.configurable
class KeyGen():
  """Creates an object that generates PRNG keys when called without arguments.
  This class cannot be used inside of a jit compiled function because it
  keeps an internal state. That said, it is useful for model
  initialization and debug

  Synopsis:
    kg = KeyGen(seed=100)
    rng = kg()
  """
  def __init__(self,seed=None):
    if seed is None:
      self._key = jax.random.PRNGKey(np.random.randint(0,2**32-1))
    else:
      self._key = jax.random.PRNGKey(seed)

  def __call__(self):
    self._key,subkey = jax.random.split(self._key)
    return subkey

  def update_rngs(self, rngs):
    rngs = {name:self() for name in rngs}
    return rngs


def print_shapes(x, header='Printed variable'):
  """a function that can be used inspect the shapes of
  leaves in a PyTree"""
  shapes = tree_util.tree_map(lambda y:y.shape,x)
  separator = ('#'*20,)
  msg = separator + (header,) + separator + (str(shapes),) + separator*2
  print('\n'.join(msg) + '\n\n')
