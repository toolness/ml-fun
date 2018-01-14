This repository contains a bunch of machine learning experiments I'm
doing.

## Experiments

Here's what I've got so far:

* [Brooklyn housing price predictions](bk-housing/BkHousing.ipynb) - Experiments with linear regression and gradient descent.
* [Super simple neural net](super-simple-nn.ipynb) - A ridiculously simple neural net.
* [Architecture convnet](architecture-convnet/) - A convolutional neural net to recognize different architectural styles in buildings.
* [Tic-tac-toe RL](rl/tic-tac-toe-rl/) - An attempt to implement a model that learns to play tic tac toe via reinforcement learning.
* [Easy21](rl/easy21/) - A reinforcement learning assignment to optimally play a variant of Blackjack.

## Running locally

To run the Jupyter notebook experiments on your own system, run:

```
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
jupyter notebook
```

In some cases, the experiments aren't in notebooks and don't use
virutalenvs; see their respective `README.md` files for more details.
