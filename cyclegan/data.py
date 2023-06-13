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

"""Input pipeline related code for a CycleGAN model"""

import gin
import jax
import tensorflow as tf
import tensorflow_datasets as tfds
from flax import jax_utils
from flax.training import common_utils


def load_local(db, *splits):
  """ Define and load a sub-split of the input data to be used with the current
   jax process.

  In a single process environment, this will return a split that is identical
  to the input split name. However, in a multi-process environment, this
  function will partition the splits over the number of processes so that each
  will have a unique subset of the data.

  Args:
    db: tfds.core.DatasetBuilder object
    *splits : Collection[Str] names of splits from the original dataset from
        which sub-splits will be carved out of.
  Returns:
    *ds: Tuple[tf.data.Dataset] containing only subsets of the input splits"""

  out_splits = []

  for split_name in splits:
    split_size = db.info.splits[split_name].num_examples//jax.process_count()
    start = jax.process_index() * split_size
    end = start + split_size
    sub_split_name = f'{split_name}[{start}:{end}]'
    ds = db.as_dataset(split=sub_split_name)
    out_splits.append(ds)

  if len(out_splits)==1:
    return out_splits[0]

  return out_splits


@gin.configurable
def train_prep_fn(
  jitter_height,
  jitter_width,
  jitter_margin,
  out_height,
  out_width
):
  """Create a pre-processing function that will normalize, random crop and
  random flip images and that can be applied to a Tensorflow dataset
  with the '.map' operation. The operations are intentionally using tensorflow
  instead of jax in order to avoid using GPU resources.

  Args:
    jitter_height, jitter_width: Int height and width to which an input image
        will be resized before being random cropped by the jitter margin.
    jitter_margin: Number of pixels by which the jitter_width and jitter_height
        dimensions of an input image will be reduced with the random crop
        operation.
    out_height, out_width: Int height and width of the output image in pixels.
  Return:
    fn: A function that can be mapped to a tensorflow dataset in order to
      apply pre-processing operations to dataset images.
    """

  crop_height = jitter_height-jitter_margin
  crop_width = jitter_width-jitter_margin

  def fn(image):
    image = tf.cast(image, tf.float32)/127.5 - 1.0
    image = tf.image.resize(
        image, [jitter_height, jitter_width],
        method=tf.image.ResizeMethod.NEAREST_NEIGHBOR
    )
    image = tf.image.random_crop(image, size=[crop_height, crop_width, 3])
    image = tf.image.random_flip_left_right(image)
    image = tf.image.resize(
        image, [out_height, out_width],
        method=tf.image.ResizeMethod.NEAREST_NEIGHBOR
    )
    return image
  return fn


@gin.register
def eval_prep_fn():
  return lambda x:tf.cast(x, tf.float32)/127.5 - 1.0


def _to_stream(ds_a, ds_b, prep_fn, batch_size, shuffle_size):
  """ Make an iterator that yields batches of unpaired images to be used
  with CycleGan models.

  This function will pre-process, batch and combine the two input datasets
  ds_a and ds_b, and then shard the output over the local devices that are
  available to the current process. The output is a tuple of two image
  batches with shapes:
  (
    [local_device_count, batch_size, H, W, C],  #<-data from ds_a
    [local_device_count, batch_size, H, W, C]		#<-data from ds_b
  )

Args:
  ds_a,	ds_b: tf.data.Dataset(s) to be combined in order to produce unpaired
      image batches
  prep_fn: Callable that will be mapped to ds_a, ds_b to pre-process images
  batch_size: Int, per-device batch size
  shuffle_size: Int, size of the buffer to be used for shuffling. Set to 0
      to disable shuffle.

Returns
  stream: Iterator that yields pairs of sharded image batches.
"""
  #batch_size is the size of batches on each device
  local_device_count = jax.local_device_count()
  local_batch_size = batch_size * local_device_count

  def prep_dataset(ds):
    ds = ds.map(lambda x:x['image']).cache().repeat()

    if shuffle_size>0:
      ds = ds.shuffle(shuffle_size*local_device_count)

    ds = ds.map(
        prep_fn,
        num_parallel_calls=tf.data.AUTOTUNE
    ).batch(
        local_batch_size,
        num_parallel_calls=tf.data.AUTOTUNE
    ).prefetch(4)

    return ds

  ds_a = prep_dataset(ds_a)
  ds_b = prep_dataset(ds_b)

  stream = tf.data.Dataset.zip((ds_a, ds_b))
  stream = tfds.as_numpy(stream)
  stream = map(common_utils.shard, stream)
  stream = jax_utils.prefetch_to_device(stream, 2)

  return stream


def load_splits(
  ds_name,
  download_dir,
  split_a,
  split_b
):

  db = tfds.builder('cycle_gan/' + ds_name)
  db.download_and_prepare(download_dir=download_dir)
  ds_a, ds_b = load_local(db, split_a, split_b)

  # Compute the number of samples per epoch as the max of the number of samples
  # in each of the two sub-datasets.
  count_a = db.info.splits[split_a].num_examples
  count_b = db.info.splits[split_b].num_examples
  n_samples_per_epoch = min(count_a, count_b)

  return ds_a, ds_b, n_samples_per_epoch


@gin.configurable
def create_stream(
  ds_name,
  download_dir,
  batch_size,
  shuffle_size,
  prep_fn,
  split_a='trainA',
  split_b='trainB'
):
  """Load the requested splits from the CycleGAN tensorflow dataset, apply a
  pre-processing function and return an iterator that will yield a 2-tuple of
  image batches from the two datasets in the shape (batch_a, batch_b)"""

  ds_a, ds_b, n_samples_per_epoch = load_splits(
      ds_name, download_dir, split_a, split_b
  )
  stream = _to_stream(ds_a, ds_b, prep_fn, batch_size, shuffle_size)
  return stream, n_samples_per_epoch
