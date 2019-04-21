import numpy as np
import tensorflow as tf
from mpi4py import MPI
from utils.mpi_tools import broadcast


def flat_concat(xs):
  return tf.concat([tf.reshape(x, (-1,)) for x in xs], axis=0)


def _get_params_from_flat(flat_params, params):

  splits = tf.split(flat_params, [tf.size(p) for p in params])
  new_params = [tf.reshape(p_new, p.shape) for p, p_new in zip(params, splits)]
  return new_params


def _get_sync_params(params):
  flat_params = flat_concat(params).numpy()
  broadcast(flat_params, root=0)
  return _get_params_from_flat(flat_params, params)


def sync_model(model):
  params = model.get_weights()
  new_params = _get_sync_params(params)
  model.set_weights(new_params)


class MpiAdamOptimizer(tf.optimizers.Adam):
  """
  Adam optimizer that averages gradients across MPI processes.
  The compute_gradients method is taken from Baselines `MpiAdamOptimizer`_.
  For documentation on method arguments, see the Tensorflow docs page for
  the base `AdamOptimizer`_.
  .. _`MpiAdamOptimizer`: https://github.com/openai/baselines/blob/master/baselines/common/mpi_adam_optimizer.py
  .. _`AdamOptimizer`: https://www.tensorflow.org/api_docs/python/tf/train/AdamOptimizer
  """

  def __init__(self, **kwargs):
    self.comm = MPI.COMM_WORLD
    super().__init__(**kwargs)

  def apply_gradient(self, grads_and_vars, name=None):
    """
    Same as normal compute_gradients, except average grads over processes.
    """
    grads_and_vars = [(g, v) for g, v in grads_and_vars if g is not None]
    flat_grad = flat_concat([g for g, v in grads_and_vars])
    sizes = [tf.size(v) for g, v in grads_and_vars]

    num_tasks = self.comm.Get_size()
    buf = np.zeros_like(flat_grad)

    self.comm.Allreduce(flat_grad, buf, op=MPI.SUM)
    avg_flat_grad = buf / num_tasks

    avg_grads = tf.split(avg_flat_grad, sizes, axis=0)
    avg_grads_and_vars = [(tf.reshape(g, v.shape), v)
                          for g, (_, v) in zip(avg_grads, grads_and_vars)]

    return super().apply_gradient(avg_grads_and_vars, name)
