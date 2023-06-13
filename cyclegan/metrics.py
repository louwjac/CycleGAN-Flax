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

"""Customized subclasses of clu.metrics."""

from flax import struct
from clu import metrics as clu_metrics


@struct.dataclass
class NestedCollection(clu_metrics.Collection):
  """Alter clu_metrics.Collection to make it possible to nest Collections
  objects.

  This is based on the same mechanics used in clu.metrics.Metric.from_output()
  Metrics that are defined with a call to "from_output" add a wrapper around
  their own "from_model_output" methods. When called with outputs, that wrapper
  will extract the subset of outputs that are intended for the wrapped metric
  and then pass it to the wrapped "from_model_output" method.
  """

  @classmethod
  def from_model_output(cls, **kwargs) -> "NestedCollection":
    return cls._from_model_output(**kwargs)

  @classmethod
  def from_output(cls, name: str):

    @struct.dataclass
    class Wrapper(cls):
      """Wrapper class whose only purpose is to intercept the model outputs
      passed to the "from_model_output" method of the parent class and to
      replace it with the subset of outputs keyed on the parent class's
      metric name."""

      @classmethod
      def from_model_output(cls, **model_output):
        output =  model_output.get(name)
        return super()._from_model_output(**output)

    Wrapper.__annotations__ = cls.__annotations__
    return Wrapper

