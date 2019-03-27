from algos.vpg import vpg, make_mlp_model
from utils.mpi_tools import mpi_fork, proc_id, mpi_statistics_scalar, num_procs
import gym
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--env', type=str, default='CartPole-v0')
parser.add_argument('--hid', type=int, default=64)
parser.add_argument('--l', type=int, default=2)
parser.add_argument('--gamma', type=float, default=0.99)
parser.add_argument('--seed', '-s', type=int, default=0)
parser.add_argument('--cpu', type=int, default=4)
parser.add_argument('--steps', type=int, default=4000)
parser.add_argument('--epochs', type=int, default=50)
parser.add_argument('--exp_name', type=str, default='vpg')
args = parser.parse_args()

mpi_fork(args.cpu)  # run parallel code with mpi

vpg(gym.make(args.env),
    ac_kwargs=dict(hidden_sizes=[args.hid] * args.l), gamma=args.gamma,
    seed=args.seed, steps_per_epoch=args.steps, epochs=args.epochs)

# model = make_mlp_model(10, (64, 64))

# print(model.get_weights())
