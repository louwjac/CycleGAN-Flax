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

"""Training library for CycleGan models."""

import functools
import contextlib
from dataclasses import dataclass
from typing import  Any, Callable, Iterable, Tuple,  List, Dict

import gin
import optax
import jax
import jax.numpy as jnp

from jax import lax
from flax import core
from flax import struct
from flax import jax_utils
from flax.training import checkpoints
from clu import metric_writers
from clu import periodic_actions
from clu import metrics as clu_metrics
from absl import logging

from . import utils, callbacks, metrics

Array = Any

class TrainState(struct.PyTreeNode):
  """Create a dataclass to hold state variables used during training. The state
  variables include the parameters of the models; and the optimizer states that
  control gradient updates. This is useful in this application mostly because
  subclassing PyTreeNode causes flax to remember how to serialize the object.
  In other words, it will enable flax to save and restore checkpoints of the
  state variables.

  Modeled after flax.training.train_state.TrainState
  see: https://github.com/google/flax/blob/main/flax/training/train_state.py

  Note, the model.apply functions and the gradient update steps are excluded
  here in order to simplify the concept.
  """

  step: int
  params_gen: core.FrozenDict[str, Any]
  params_disc: core.FrozenDict[str, Any]
  opt_state_gen: optax.OptState
  opt_state_disc: optax.OptState
  rngs_gen: dict[str, Any]
  fake_history: Tuple[Array, Array]

  def update(
      self,
      params_gen,
      params_disc,
      opt_state_gen,
      opt_state_disc,
      fake_history
  ):
    """Increments the step counter and replaces state variables
    with updated values."""
    rngs_gen=jax.tree_map(lambda x:jax.random.split(x)[0],self.rngs_gen)

    return self.replace(
        step=self.step + 1,
        params_gen=params_gen,
        params_disc=params_disc,
        opt_state_gen=opt_state_gen,
        opt_state_disc=opt_state_disc,
        rngs_gen=rngs_gen,
        fake_history=fake_history
    )

  @classmethod
  def create(
      cls,
      generator,
      discriminator,
      optimizer_gen,
      optimizer_disc,
      rngs_gen,
      rngs_disc,
      rngs_noise,
      sample_batch,
      history_length
  ):
    """Creates a new train state instance with `step=0` and initialized
    parameters and optimizer states """
    # Initialize the models
    # Note, states_... below are not used because the models don't
    # use any layers that depend on mutable states.
    vars_gen = generator.init(rngs_gen, sample_batch)
    states_gen, params_gen = vars_gen.pop('params')
    del vars_gen, states_gen

    vars_disc = discriminator.init(rngs_disc, sample_batch)
    states_disc, params_disc = vars_disc.pop('params')
    del vars_disc, states_disc

    # Initialize the optimizers
    opt_state_gen = optimizer_gen.init(params_gen)
    opt_state_disc = optimizer_disc.init(params_disc)

    #Initialize the fake image history to noise
    def get_noise_img(sample, rng):
      shape =  (history_length,) +  sample.shape[1:]
      #sample random values from a normal distribution
      noise = jax.random.normal(rng, shape=shape, dtype=jnp.float32)
      #change the range of the values from [0,1] to [-1,1]
      #(same as what will be used for input images)
      noise = noise*2. - 1.
      return noise

    rngs_gen = jax.tree_map(lambda x:jax.random.split(x)[1],rngs_gen)
    fake_history = jax.tree_map(get_noise_img, sample_batch, rngs_noise)

    return cls(
        step=0,
        params_gen=params_gen,
        params_disc=params_disc,
        opt_state_gen=opt_state_gen,
        opt_state_disc=opt_state_disc,
        rngs_gen=rngs_gen,
        fake_history=fake_history
    )


@struct.dataclass
class GeneratorMetrics(metrics.NestedCollection):
  network_loss: clu_metrics.Average.from_output('network_loss')
  cycle_loss: clu_metrics.Average.from_output('cycle_loss')
  identity_loss: clu_metrics.Average.from_output('identity_loss')
  adversarial_loss: clu_metrics.Average.from_output('adversarial_loss')


@struct.dataclass
class DiscriminatorMetrics(metrics.NestedCollection):
  network_loss: clu_metrics.Average.from_output('network_loss')
  adversarial_loss: clu_metrics.Average.from_output('adversarial_loss')


@struct.dataclass
class ModelMetrics(metrics.NestedCollection):
  loss: clu_metrics.Average.from_output('loss')
  learn_rate: clu_metrics.LastValue.from_output('learn_rate')


@struct.dataclass
class TrainMetrics(metrics.NestedCollection):
  net_g: GeneratorMetrics.from_output('net_g')
  net_f: GeneratorMetrics.from_output('net_f')
  net_x: DiscriminatorMetrics.from_output('net_x')
  net_y: DiscriminatorMetrics.from_output('net_y')
  generator: ModelMetrics.from_output('generator')
  discriminator:ModelMetrics.from_output('discriminator')


@gin.register
def calc_mae(x: Array, y: Array|float) -> float:
  """Compute the mean absolute error of values in x wrt. target(s) y"""
  mae = jnp.mean(jnp.abs(x - y))
  return mae


@gin.register
def calc_mse(x: Array, y: Array|float) -> float:
  """Compute the mean square error of values in x wrt. target(s) y"""
  mse =  jnp.mean(jnp.square(x - y))
  return mse

@gin.register
def calc_ce(x: Array, y: Array|float) -> float:
  """Compute the average of optax.sigmoid_binary_cross_entropy for values in
   x wrt. target(s) y"""
  ce = jnp.mean(optax.sigmoid_binary_cross_entropy(x,y))
  return ce


@gin.configurable
def generator_forward_fn(
    generator: Callable,
    discriminator: Callable,
    id_scalar: float,
    cycle_scalar: float,
    lr_schedule: Callable,
    adv_loss_fn: Callable=calc_mse,
    id_loss_fn: Callable=calc_mae,
    cycle_loss_fn: Callable=calc_mae
) -> Callable:
  """Construct a function that will execute a forward pass of
  a CycleGenerator model

  Args:
    generator: Callable, instance of models.CycleGenerator
    discriminator: Callable, instance of models.CycleDiscriminator
    id_scalar: Float value that controls the weight of identity loss in the
      total loss calculation.
    cycle_scalar: Float value that controls the weight of cycle loss in the
      total loss calculation.
    lr_schedule: Callable that returns a float learning rate when
      called with an integer step value.
    adv_loss_fn: Callable that computes adversarial loss when called with patch
      scores produced by the discriminator. The function must have an input
      signature of fn(scores, target) where "scores" and "target" need to have
      either the same shapes, or be broadcastable. In practice, this will either
      "calc_mse" or "calc_ce". Original CycleGAN used mean squared error.
    id_loss_fn: Callable that computes identity loss when called with a real and
      an identity image.
    cycle_loss_fn: Callable that computes cycle loss when called with a real and
      a cycled image.

  Returns:
    fn: The generator forward function
  """

  def _calc_network_loss(
      real_img: Array,
      cycled_img: Array,
      identity_img: Array,
      patch_scores_fake: Array
  ) -> Tuple[float, Dict]:
    cycle_loss = cycle_loss_fn(real_img, cycled_img)
    id_loss = id_loss_fn(real_img, identity_img)
    adv_loss = adv_loss_fn(patch_scores_fake, 1.)
    network_loss = adv_loss + id_loss * id_scalar + cycle_loss * cycle_scalar

    network_metrics = {
        'network_loss': network_loss,
        'cycle_loss': cycle_loss,
        'identity_loss': id_loss,
        'adversarial_loss': adv_loss,
    }

    return network_loss, network_metrics

  def fn(
      params_gen: core.FrozenDict[str, Any],
      params_disc: core.FrozenDict[str, Any],
      real_imgs: Tuple[Array, Array],
      step: int
  ) -> Tuple[float, Any]:
    """Execute a forward pass of the generator model provided to outer
    function. This consists of computing the generator outputs and
    the loss function. The output signature is:
        loss, (auxiliary variables)
    so that it can be used with jax.grad

    Args:
      params_gen: Pytree of CycleGenerator model parameters
      params_disc: Pytree of CycleDiscriminator model parameters
      real_imgs: tuple of real image batches (real_x, real_y)
      step: int value that is used to compute scheduled values of scalars
        used in the loss function

    Returns:
      loss: float loss value
      metrics: Dict of metrics to log
      fake_imgs: Tuple of output image batches (fake_x, fake_y)
    """

    #encode the input images
    real_x, real_y = real_imgs

    (
        identity_x,
        identity_y,
        fake_y,
        fake_x,
        cycle_x,
        cycle_y,
    ) = generator.apply({'params':params_gen}, real_imgs)

    fake_imgs = (fake_x, fake_y)
    # get the discriminator scores for the fake images
    # to be used with the generator loss
    score_fake_x, score_fake_y = discriminator.apply(
        {'params':params_disc}, fake_imgs
    )
    loss_g, metrics_g = _calc_network_loss(
        real_x, cycle_x, identity_x, score_fake_y
    )
    loss_f, metrics_f = _calc_network_loss(
        real_y, cycle_y, identity_y, score_fake_x
    )

    total_loss = loss_g + loss_f
    learn_rate = lr_schedule(step)

    gen_metrics = {
        'net_g': metrics_g,
        'net_f': metrics_f,
        'generator': {
            'learn_rate': learn_rate,
            'loss': total_loss,
        },
    }

    return total_loss, (gen_metrics, real_imgs, fake_imgs)
  return fn


@gin.configurable
def discriminator_forward_fn(
    discriminator: Callable,
    lr_schedule: Callable,
    loss_fn: Callable = calc_mse
) -> Tuple[float, Any]:
  """Construct a function that will execute a forward pass of a
  CycleDiscriminator model

  Args:
    discriminator: Callable instance of models.CycleDiscriminator
    lr_schedule: Callable that returns a float learning rate when
      called with an integer step value.
    loss_fn: Callable that computes adversarial loss when called with patch
      scores produced by the discriminator. The function must have an input
      signature of fn(scores, target) where "scores" and "target" need to have
      either the same shapes, or be broadcastable. In practice, this will either
      "calc_mse" or "optax.sigmoid_binary_cross_entropy". Original CycleGAN used
      mean squared error.

    Returns:
      fn: The forward pass function
    """

  def _calc_loss_component(
      params: core.FrozenDict[str, Any],
      imgs: Tuple[Array, Array],
      target: float
  ) -> Tuple[float, float]:
    scores_x, scores_y  = discriminator.apply(params, imgs)
    loss_x = loss_fn(scores_x, target)
    loss_y = loss_fn(scores_y, target)
    return loss_x, loss_y

  def fn(
      params_disc: core.FrozenDict[str, Any],
      real_imgs: Tuple[Array, Array],
      fake_imgs: Tuple[Array, Array],
      step: int
  ) -> Tuple[float, Dict]:
    """Execute a forward pass of the discriminator model.
    This consists of computing the discriminator outputs and the loss function.

    Args:
      params_disc: Pytree of CycleDiscriminator model parameters
      real_imgs: Tuple of real image batches (real_x, real_y)
      fake_imgs: Tuple of fake image batches (fake_x, fake_y)
    Returns:
      loss: float loss value
      metrics: Dict of metrics to log
    """

    params = {'params':params_disc}
    loss_real_x, loss_real_y = _calc_loss_component(params, real_imgs, 1.)
    loss_fake_x, loss_fake_y = _calc_loss_component(params, fake_imgs, 0.)
    network_loss_x = loss_real_x + loss_fake_x
    network_loss_y = loss_real_y + loss_fake_y
    total_loss = network_loss_x + network_loss_y
    learn_rate = lr_schedule(step)

    # save metrics for logging purposes
    disc_metrics = {
        'net_x': {
            'network_loss': network_loss_x,
            'adversarial_loss': loss_fake_x,
        },
        'net_y': {
            'network_loss': network_loss_y,
            'adversarial_loss': loss_fake_y,
        },
        'discriminator': {
            'learn_rate': learn_rate,
            'loss': total_loss,
        },
    }

    return total_loss, disc_metrics
  return fn


@gin.configurable
def train_step_fn(
    gen_forward: Callable,
    disc_forward: Callable,
    optimizer_gen: Callable,
    optimizer_disc: Callable,
    pmap_axis='batch'
) -> Callable:
  """Construct a function that will execute a train step of a CycleGan model

  Args:
    gen_forward:Callable, the forward pass of the generator model
    disc_forward:Callable, the forward pass of the discriminator model
    optimizer_gen: Callable, instance of an optax optimizer that will be used
      to update the generator parameters
    optimizer_disc: Callable, instance of an optax optimizer that will be
      used to update the discriminator parameters

  Returns:
    fn: The train step function
  """
  gen_grad_fn = jax.grad(gen_forward, has_aux=True)
  disc_grad_fn = jax.grad(disc_forward, has_aux=True)

  def apply_updates(grads, opt_states, params, optimizer):
    updates, new_opt_state = optimizer.update(grads, opt_states, params)
    new_params = optax.apply_updates(params, updates)
    new_params = lax.pmean(new_params, axis_name=pmap_axis)
    return new_params, new_opt_state

  def fn(
      state: TrainState,
      real_imgs: Tuple[Array, Array],
      train_metrics: TrainMetrics
  ) -> Tuple[TrainState, TrainMetrics]:
    """Execute a train step of a CycleGan model.
    This consists of executing the generator and discriminator forward
    functions, calculating the gradients, and applying the updates to
    model parameters.

    Args:
      state: TrainState object that holds current model parameters
      real_imgs: Tuple of real image batches (real_x, real_y)
      train_metrics: TrainMetrics object that holds values of train metrics
        that have been accumulated up to the current train step. The train
        loop will periodically compute, log and reset this object.
    Returns:
      new_state: a new TrainState that contains updated model parameters
      metrics: accumulated TrainMetrics from the current train step
    """

    #train the generator
    grads_gen, (metrics_gen, real_imgs, fake_imgs)  = gen_grad_fn(
        state.params_gen, state.params_disc, real_imgs, state.step
    )

    params_gen, opt_state_gen = apply_updates(
        grads_gen, state.opt_state_gen, state.params_gen, optimizer_gen
    )

    #update the fake buffer with the new fake images
    fake_history = jax.tree_map(
        lambda x,y: jnp.concatenate((x[y.shape[0]:],y),axis=0),
        state.fake_history,
        fake_imgs
    )

    # train the discriminator
    grads_disc, metrics_disc  = disc_grad_fn(
        state.params_disc, real_imgs, fake_history, state.step
    )

    params_disc, opt_state_disc = apply_updates(
        grads_disc, state.opt_state_disc, state.params_disc, optimizer_disc
    )

    #update the model states
    new_state = state.update(
        params_gen, params_disc, opt_state_gen, opt_state_disc, fake_history
    )

    # sync the metric values on all devices
    # according to the 'reduce()' method specific to each metric type.
    # For example, metrics that use the 'Average' reduce function will get
    # updated to the average value over all devices on every device.
    curr_metrics=train_metrics.gather_from_model_output(
        **metrics_gen, **metrics_disc,
        axis_name=pmap_axis
    )

    return new_state, curr_metrics

  if not pmap_axis:
    return fn

  return jax.pmap(fn, axis_name=pmap_axis)


@dataclass
class Loop():
  """a base class for training loops. Subclasses should, at minimum provide
  a 'create' method to instantiate instances."""
  step: int
  state: TrainState
  sample_batch: Any
  train_stream: Iterable
  train_metrics: TrainMetrics
  train_step_fn: Callable
  ckpt_save_fn: Callable
  image_fn: Callable
  hooks: List[periodic_actions.PeriodicAction]
  train_steps: int
  progress_steps: int
  metrics_steps: int
  images_steps: int
  checkpoint_steps: int
  work_dir: str
  log_dir: str
  image_dir: str
  config_dir: str
  checkpoint_dir:str
  counter: int=0
  writers: Tuple=tuple()

  def __post_init__(self):
    #save checkpoints on regular interval
    ckpt_hook= periodic_actions.PeriodicCallback(
        every_steps=self.checkpoint_steps,
        callback_fn=self.save_checkpoint,
        pass_step_and_time=False
    )
    self.hooks.append(ckpt_hook)

    # keep a list of all the metrics writers that are created so that they can
    # be flushed by the context manager on exit
    writers = []

    # Log training progress related values such as steps-per-second and
    # % completion to STDOUT and to Tensorboard
    progress_writer = metric_writers.create_default_writer(
        logdir=self.log_dir,
        collection='progress'
    )
    writers.append(progress_writer)

    progress_hook = periodic_actions.ReportProgress(
      num_train_steps=self.train_steps,
      every_steps=self.progress_steps,
      writer=progress_writer
    )
    self.hooks.append(progress_hook)

    train_writers = {
        collection_name:metric_writers.create_default_writer(
            logdir=self.log_dir,
            collection=collection_name
        )
        for collection_name
        in self.train_metrics.__annotations__.keys()
    }
    writers += [*train_writers.values()]

    def metrics_callback():
      #compute summaries of the aggregated metrics and log them to Tensorboard
      summary = self.train_metrics.unreplicate().compute()
      for collection_name, writer in train_writers.items():
        writer.write_scalars(self.step,summary[collection_name])
        writer.flush()

      #reset metrics in the instance
      self.train_metrics = jax_utils.replicate(self.train_metrics.empty())

    metrics_hook= periodic_actions.PeriodicCallback(
        every_steps=self.metrics_steps,
        callback_fn=metrics_callback,
        pass_step_and_time=False
    )
    self.hooks.append(metrics_hook)

    # log images to Tensorboard and potentially save them as separate files in a
    # designated folder
    image_writer = metric_writers.SummaryWriter(logdir=self.image_dir)
    writers.append(image_writer)

    def write_images():
      state = jax_utils.unreplicate(self.state)
      train_batch = jax_utils.unreplicate(next(self.train_stream))

      self.image_fn(
          input_batches=(self.sample_batch, train_batch),
          state=state,
          writer=image_writer
      )
      image_writer.flush()

    images_hook= periodic_actions.PeriodicCallback(
        every_steps=self.images_steps,
        callback_fn=write_images,
        pass_step_and_time=False
    )
    self.hooks.append(images_hook)
    self.writers = tuple(writers)

  def run_one_step(self):
    """Run one step of the training loop """
    train_batch = next(self.train_stream)
    self.state, curr_metrics = self.train_step_fn(
        self.state, train_batch, self.train_metrics
    )
    self.train_metrics = self.train_metrics.merge(curr_metrics)
    self.step += 1
    self.counter += 1

    for h in self.hooks:
      h(self.step)

  def run(self):
    self.log_config()
    logging.info(
        'Executing the loop starting with step %d of %d',
        self.step, self.train_steps
    )
    count=0
    for _ in range(self.step,self.train_steps):
      self.run_one_step()
      count+=1

  def log_config(self):
    cfg = gin.operative_config_str()
    logging.info('Using operative gin config:\n%s', cfg)
    cfg = gin.markdown(cfg)

    with contextlib.closing(
        metric_writers.SummaryWriter(logdir=self.config_dir)
    ) as writer:
      writer.write_texts(self.step,{f'step {self.step}':cfg})
      writer.flush()

  def save_checkpoint(self):
    chkpt_state = jax.device_get(jax_utils.unreplicate(self.state))
    self.ckpt_save_fn(target=chkpt_state,step=self.step)

  def load_checkpoint(self, step=None, ckpt_dir=None):
    if not ckpt_dir:
      ckpt_dir=self.checkpoint_dir

    state = checkpoints.restore_checkpoint(
        ckpt_dir=ckpt_dir,
        target=jax_utils.unreplicate(self.state),
        step=step
    )
    self.step=state.step.tolist()
    self.state=jax_utils.replicate(state)

  @classmethod
  def create(cls):
    #needs to be implemented in a subclass
    raise NotImplementedError

  def context(self):
    try:
      yield self
    finally:
      for writer in self.writers:
        writer.flush()

      if self.counter > 1000:
        #make sure that the current train state is saved as a checkpoint
        self.save_checkpoint()

        #save the generator and discriminator parameters separately for sharing
        params = jax.device_get(jax_utils.unreplicate(self.state.params_gen))
        checkpoints.save_checkpoint(
            ckpt_dir=self.checkpoint_dir,
            prefix='params_gen',
            overwrite=True,
            keep=1,
            target=params,
            step=self.step
        )

        params = jax.device_get(jax_utils.unreplicate(self.state.params_disc))
        checkpoints.save_checkpoint(
            ckpt_dir=self.checkpoint_dir,
            prefix='params_disc',
            overwrite=True,
            keep=1,
            target=params,
            step=self.step
        )


@gin.register
class TrainLoop(Loop):
  """CycleGan train loop"""

  @classmethod
  @contextlib.contextmanager
  def create(
    cls,
    generator,
    discriminator,
    optimizer_gen,
    optimizer_disc,
    train_stream,
    work_dir,
    ckpt_keep=5,
    ckpt_keep_every_n_steps= None,
    epochs=1,
    samples_per_epoch=1,
    metrics_steps=100,
    progress_steps=100,
    images_steps=500,
    checkpoint_steps=1000,
    fake_buffer_size=1
  ):
    train_steps = epochs*samples_per_epoch

    folders = {
        'work_dir':work_dir,
        'log_dir':f'{work_dir}/log',
        'image_dir':f'{work_dir}/log/images',
        'config_dir':f'{work_dir}/log/config',
        'checkpoint_dir':f'{work_dir}/checkpoints',
    }

    sample_batch = jax_utils.unreplicate(next(train_stream))
    kg = utils.KeyGen()

    #initialize CycleGan models
    rngs_gen = {'params': kg()}
    rngs_disc = {'params': kg()}
    rngs_noise = (kg(),kg())

    state = TrainState.create(
        generator,
        discriminator,
        optimizer_gen,
        optimizer_disc,
        rngs_gen,
        rngs_disc,
        rngs_noise,
        sample_batch,
        fake_buffer_size
    )

    step = state.step
    state = jax_utils.replicate(state)

    # Define the forward functions.
    # Arguments designated with 'gin.REQUIRED' must be provided in a gin
    # configuration file.
    # Optional inputs which are not provided here may be configured with gin
    gen_forward = generator_forward_fn(
        generator,
        discriminator,
        id_scalar=gin.REQUIRED,
        cycle_scalar=gin.REQUIRED,
        lr_schedule=gin.REQUIRED
    )

    disc_forward = discriminator_forward_fn(
        discriminator,
        lr_schedule=gin.REQUIRED
    )

    #set up the training loop
    train_fn = train_step_fn(
        gen_forward,
        disc_forward,
        optimizer_gen,
        optimizer_disc,
        pmap_axis='batch'
    )

    # note the use of async_manager to allow the training process to continue
    # running while the cpu saves a checkpoint
    ckpt_save_fn = functools.partial(
        checkpoints.save_checkpoint,
        ckpt_dir=folders['checkpoint_dir'],
        overwrite=True,
        keep=ckpt_keep,
        keep_every_n_steps=ckpt_keep_every_n_steps,
        async_manager=checkpoints.AsyncManager()
    )

    image_fn = callbacks.GanImageCallback(generator, discriminator)
    train_metrics = TrainMetrics.empty()
    instance = cls(
        train_steps=train_steps,
        progress_steps=progress_steps,
        metrics_steps=metrics_steps,
        images_steps=images_steps,
        checkpoint_steps=checkpoint_steps,
        step=step,
        state=state,
        sample_batch=sample_batch,
        train_metrics=jax_utils.replicate(train_metrics),
        train_step_fn=train_fn,
        train_stream=train_stream,
        ckpt_save_fn=ckpt_save_fn,
        image_fn=image_fn,
        hooks=[],
        **folders
    )

    return instance.context()


@gin.configurable
def create_loop(loop_class,*args,**kwargs):
  """A wrapper that allows a selected loop class to be configured in a
  gin file"""
  return loop_class.create(*args,**kwargs)
