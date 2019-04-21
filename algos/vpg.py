import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_probability as tfp
import gym
from gym.spaces import Box, Discrete
import time
from utils.mpi_tools import mpi_fork, proc_id, mpi_statistics_scalar, num_procs
from utils.mpi_tf import MpiAdamOptimizer, sync_model
import scipy.signal

tfd = tfp.distributions


"""
functions in core.py
"""
EPS = 1e-8


def combined_shape(length, shape=None):
  if shape is None:
    return (length,)
  return (length, shape) if np.isscalar(shape) else (length, *shape)


def discount_cumsum(x, discount):
  """
  magic from rllab for computing discounted cumulative sums of vectors.
  input:
      vector x,
      [x0,
       x1,
       x2]
  output:
      [x0 + discount * x1 + discount^2 * x2,
       x1 + discount * x2,
       x2]
  """
  return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]


def gaussian_likelihood(x, mu, log_std):
  pre_sum = -0.5 * (((x - mu) / (tf.exp(log_std) + EPS))**2 + 2 * log_std + np.log(2 * np.pi))
  return tf.reduce_sum(pre_sum, axis=1)


class MLP(tf.keras.Model):
  def __init__(self,
               sizes,
               activation='tanh',
               output_activation=None,
               is_continue_action=False,
               act_dim=None):
    super().__init__()

    self.denses = [tf.keras.layers.Dense(size, activation=activation) for size in sizes[:-1]]
    self.out = tf.keras.layers.Dense(sizes[-1], activation=output_activation)
    if is_continue_action:
      if act_dim is None:
        raise TypeError("__init__() missing 1 argument: 'act_dim' when log_std=True")
      self.log_std = tf.Variable(name='log_std', initial_value=-0.5 * np.ones(act_dim, dtype=np.float32))

  @tf.function
  def call(self, x):
    for dense in self.denses:
      x = dense(x)
    return self.out(x)


"""
Actor-Critics
"""


class ActorCritic:
  def __init__(self, hidden_sizes=(64, 64), activation='tanh', output_activation=None, action_space=None, pi_lr=3e-4, vf_lr=1e-3, train_v_iters=80):
    if isinstance(action_space, Box):
      act_dim = len(action_space.sample())
      self.pi_mlp = MLP(list(hidden_sizes) + [act_dim], activation, output_activation, is_continue_action=True)
      self.policy = self._mlp_gaussian_policy

    elif isinstance(action_space, Discrete):
      act_dim = action_space.n
      self.pi_mlp = MLP(list(hidden_sizes) + [act_dim], activation, None)
      self.policy = self._mlp_categorical_policy
      self.action_space = action_space
    self.v_mlp = MLP(list(hidden_sizes) + [1], activation, None)
    # optimizers
    self.pi_optimizer = MpiAdamOptimizer(learning_rate=pi_lr)
    self.vf_optimizer = MpiAdamOptimizer(learning_rate=vf_lr)
    self.train_v_iters = train_v_iters

  @tf.function
  def __call__(self, observation, action=None):
    pi, logp_pi = self.policy(observation, action)
    v = tf.squeeze(self.v_mlp(observation), axis=1)
    return pi, logp_pi, v

  def _mlp_categorical_policy(self, observation, action):
    act_dim = self.action_space.n
    logits = self.pi_mlp(observation)
    logp_all = tf.nn.log_softmax(logits)
    pi = tfd.Categorical(logits).sample()  # pi is the next action
    if action is not None:
      action = tf.cast(action, tf.int32)
      logp = tf.reduce_sum(tf.one_hot(action, act_dim, dtype=tf.float64) * logp_all, axis=1)
    else:
      logp = tf.reduce_sum(tf.one_hot(pi, act_dim, dtype=tf.float64) * logp_all, axis=1)
    return pi, logp

  def _mlp_gaussian_policy(self, observation, action):
    mu = self.pi_mlp(observation)
    std = tf.exp(self.pi_mlp.log_std)
    pi = mu + tf.random.normal(tf.shape(mu)) * std  # pi is the next action
    if action is not None:
      logp = gaussian_likelihood(action, mu, self.pi_mlp.log_std)
    else:
      logp = gaussian_likelihood(pi, mu, self.pi_mlp.log_std)
    return pi, logp

  def update(self, buf):
    obs_buf, act_buf, adv_buf, ret_buf, logp_buf = buf.get()

    with tf.GradientTape() as pi_tape, tf.GradientTape() as vf_tape:
      pi, logp, v = self.__call__(obs_buf, act_buf)

      pi_loss = -tf.reduce_mean(logp * adv_buf)
      v_loss = tf.reduce_mean((ret_buf - v) ** 2)

    # if hasattr(self, 'log_std'):
    #   all_trainable_variables = [self.log_std, *self.pi_mlp.trainable_variables]
    #   pi_grads = pi_tape.gradient(pi_loss, all_trainable_variables)
    #   self.pi_optimizer.apply_gradients(zip(pi_grads, all_trainable_variables))
    # else:
      # pi_grads = pi_tape.gradient(pi_loss, self.pi_mlp.trainable_variables)
      # self.pi_optimizer.apply_gradients(zip(pi_grads, self.pi_mlp.trainable_variables))

    pi_grads = pi_tape.gradient(pi_loss, self.pi_mlp.trainable_variables)
    self.pi_optimizer.apply_gradients(zip(pi_grads, self.pi_mlp.trainable_variables))

    vf_grads = vf_tape.gradient(v_loss, self.v_mlp.trainable_variables)
    for _ in range(self.train_v_iters):
      self.vf_optimizer.apply_gradients(zip(vf_grads, self.v_mlp.trainable_variables))

    # make sure that params across processes are synchronized
    self.synchronize()

  def synchronize(self):
    sync_model(self.pi_mlp)
    sync_model(self.v_mlp)


"""
Experience Buffer
"""


class VPGBuffer:
  """
  A buffer for storing trajectories experienced by a VPG agent interacting
  with the environment, and using Generalized Advantage Estimation (GAE-Lambda)
  for calculating the advantages of state-action pairs.
  """

  def __init__(self, obs_dim, act_dim, size, gamma=0.99, lam=0.95):
    self.obs_buf = np.zeros(combined_shape(size, obs_dim), dtype=np.float32)
    self.act_buf = np.zeros(combined_shape(size, act_dim), dtype=np.float32)
    self.adv_buf = np.zeros(size, dtype=np.float32)
    self.rew_buf = np.zeros(size, dtype=np.float32)
    self.ret_buf = np.zeros(size, dtype=np.float32)
    self.val_buf = np.zeros(size, dtype=np.float32)
    self.logp_buf = np.zeros(size, dtype=np.float32)
    self.gamma, self.lam = gamma, lam
    self.ptr, self.path_start_idx, self.max_size = 0, 0, size

  def store(self, obs, act, rew, val, logp):
    """
    Append one timestep of agent-environment interaction to the buffer.
    """
    assert self.ptr < self.max_size     # buffer has to have room so you can store
    self.obs_buf[self.ptr] = obs
    self.act_buf[self.ptr] = act
    self.rew_buf[self.ptr] = rew
    self.val_buf[self.ptr] = val
    self.logp_buf[self.ptr] = logp
    self.ptr += 1

  def finish_path(self, last_val=0):
    """
    Call this at the end of a trajectory, or when one gets cut off
    by an epoch ending. This looks back in the buffer to where the
    trajectory started, and uses rewards and value estimates from
    the whole trajectory to compute advantage estimates with GAE-Lambda,
    as well as compute the rewards-to-go for each state, to use as
    the targets for the value function.
    The "last_val" argument should be 0 if the trajectory ended
    because the agent reached a terminal state (died), and otherwise
    should be V(s_T), the value function estimated for the last state.
    This allows us to bootstrap the reward-to-go calculation to account
    for timesteps beyond the arbitrary episode horizon (or epoch cutoff).
    """

    path_slice = slice(self.path_start_idx, self.ptr)
    rews = np.append(self.rew_buf[path_slice], last_val)
    vals = np.append(self.val_buf[path_slice], last_val)

    # the next two lines implement GAE-Lambda advantage calculation
    deltas = rews[:-1] + self.gamma * vals[1:] - vals[:-1]
    self.adv_buf[path_slice] = discount_cumsum(deltas, self.gamma * self.lam)

    # the next line computes rewards-to-go, to be targets for the value function
    self.ret_buf[path_slice] = discount_cumsum(rews, self.gamma)[:-1]

    self.path_start_idx = self.ptr

  def get(self):
    """
    Call this at the end of an epoch to get all of the data from
    the buffer, with advantages appropriately normalized (shifted to have
    mean zero and std one). Also, resets some pointers in the buffer.
    """
    assert self.ptr == self.max_size    # buffer has to be full before you can get
    self.ptr, self.path_start_idx = 0, 0
    # the next two lines implement the advantage normalization trick
    adv_mean, adv_std = mpi_statistics_scalar(self.adv_buf)
    self.adv_buf = (self.adv_buf - adv_mean) / adv_std
    return self.obs_buf, self.act_buf, self.adv_buf, self.ret_buf, self.logp_buf


"""
Vanilla Policy Gradient
(with GAE-Lambda for advantage estimation)
"""


def vpg(env, ac_kwargs=None, seed=0, steps_per_epoch=4000, epochs=50, gamma=0.99, lam=0.97, max_ep_len=1000, save_freq=10):

  seed += 10000 * proc_id()
  tf.random.set_seed(seed)
  np.random.seed(seed)
  # Create actor-critic agent and synchronize it
  ac_kwargs['action_space'] = env.action_space

  actor_critic = ActorCritic(**ac_kwargs)
  # actor_critic.synchronize()

  # Experience buffer
  obs_dim = env.observation_space.shape
  act_dim = env.action_space.shape
  local_steps_per_epoch = int(steps_per_epoch / num_procs())
  buf = VPGBuffer(obs_dim, act_dim, local_steps_per_epoch, gamma, lam)

  """
  Main loop: collect experience in env and update/log each epoch
  """

  # o for observation, r for reward, d for done
  o, r, d, ep_ret, ep_len = env.reset(), 0, False, 0, 0

  all_ep_ret = []
  summary_ep_ret = []
  totalEnvInteracts = []
  for epoch in range(epochs):
    for t in range(local_steps_per_epoch):
      a, logp_t, v_t = actor_critic(o.reshape(1, -1))

      # save and log
      a = a.numpy()[0]
      buf.store(o, a, r, v_t, logp_t)

      o, r, d, _ = env.step(a)
      ep_ret += r
      ep_len += 1

      terminal = d or (ep_len == max_ep_len)
      if terminal or (t == local_steps_per_epoch - 1):
        if not(terminal) and proc_id() == 0:
          print('Warning: trajectory cut off by epoch at %d steps.' % ep_len)
        # if trajectory didn't reach terminal state, bootstrap value target
        last_val = r if d else v_t
        buf.finish_path(last_val)

        if terminal:
          all_ep_ret.append(ep_ret)
        # reset environment
        o, r, d, ep_ret, ep_len = env.reset(), 0, False, 0, 0

    # Perform VPG update!
    actor_critic.update(buf)
    mean, std = mpi_statistics_scalar(all_ep_ret)
    all_ep_ret = []
    if proc_id() == 0:
      print(f'epoch {epoch}: mean {mean}, std {std}')
    summary_ep_ret.append(mean)
    totalEnvInteracts.append((epoch + 1) * steps_per_epoch)

  if proc_id() == 0:
    plt.plot(totalEnvInteracts, summary_ep_ret)
    plt.grid(True)
    plt.show()


def main():
  import argparse
  parser = argparse.ArgumentParser()
  parser.add_argument('--env', type=str, default='CartPole-v0')
  parser.add_argument('--hid', type=int, default=64)
  parser.add_argument('--l', type=int, default=2)
  parser.add_argument('--gamma', type=float, default=0.99)
  parser.add_argument('--seed', '-s', type=int, default=0)
  parser.add_argument('--cpu', type=int, default=4)
  parser.add_argument('--steps', type=int, default=4000)
  parser.add_argument('--epochs', type=int, default=70)
  parser.add_argument('--exp_name', type=str, default='vpg')
  args, _ = parser.parse_known_args()

  mpi_fork(args.cpu)  # run parallel code with mpi

  vpg(gym.make(args.env),
      ac_kwargs=dict(hidden_sizes=[args.hid] * args.l), gamma=args.gamma,
      seed=args.seed, steps_per_epoch=args.steps, epochs=args.epochs)
