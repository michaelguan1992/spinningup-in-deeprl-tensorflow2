# DeepRL-in-Tensorflow2
Recently, Google releases their next generation end-to-end open source machine learning platform Tensorflow 2.0 (March 2019), and back in September 2018, OpenAI releases a beginner-level tutorial series of reinforcement learing called Spinning Up in Deep RL. I love these two project, and one thought came to my mind: why not re-implement the spinning up deep RL algorithms with Tensorflow 2.0? It may also benefits beginners. 

The implementation includes two parts: python script file and jupyter notebook. The jupyter notebook implementation is a simplified version that can be run by itself, but with no MPI support.

This repository is actively under development. 

Setup:
Install OpenMPI:
For Ubuntu:
>sudo apt-get update && sudo apt-get install libopenmpi-dev


For Mac OS X, you need to install Homebrew, and then
>brew install openmpi

Required python packages:
* Tensorflow 2.0 alpha
* Tensorflow probability nightly
* gym
* mpi4py (not necessary for jupyter notebook version)
* matplotlib
* jupyter notebook/lab

Links:
OpenAI SpinningUp:
>https://spinningup.openai.com/en/latest/index.html
