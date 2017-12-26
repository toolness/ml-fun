This is an attempt to create a model capable of playing
tic-tac-toe through reinforcement learning.

At first I was going to use Q-Learning, as explained in
the paper [Playing Atari with Deep Reinforcement Learning][atari],
but after reading Andrej Karpathy's
[Deep Reinforcement Learning: Pong from Pixels][karpathy] I opted
for Policy Gradients. I'm not sure if I actually implemented them
correctly, though.

## Quick start

This project doesn't use the virtualenv used by the rest of
the projects in this repository. Instead, you'll probably want
to install Anaconda with keras.

To run tests:

```
python test.py
```

To train:

```
python train.py
```

To have the model play one game of tic-tac-toe against itself:

```
python play.py
```

[atari]: https://deepmind.com/research/publications/playing-atari-deep-reinforcement-learning/
[karpathy]: http://karpathy.github.io/2016/05/31/rl/
