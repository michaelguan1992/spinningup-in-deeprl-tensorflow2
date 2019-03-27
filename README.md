Recently, Google releases their next generation end-to-end open source machine learning platform Tensorflow 2.0 (March 2019), and back in September 2018, OpenAI releases a beginner-level tutorial series of reinforcement learing called Spinning Up in Deep RL. I love these two project, and one thought came to my mind: why don't I re-implement the spinning up deep RL algorithms with Tensorflow 2.0? It may also benefits beginners. What a thought!

I tried to keep the codes the same as the original codes if possible, so that one can easier to compare Tensorflow 1.x codes and Tensorflow 2.0 codes. The biggest change from the original implementation is that 

Setup:
Tensorflow 2.0 alpha
Tensorflow probability nightly
gym
mpi4py
