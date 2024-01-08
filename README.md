# OAC-BVR

A PyTorch implementation for our paper "Offline Reinforcement Learning With Behavior Value Regularization". Our code is built off of [TD3-BC](https://github.com/sfujim/TD3_BC).

## Prerequisites

- PyTorch 2.0.1 with Python 3.7 
- MuJoCo 2.00 with mujoco-py 2.1.2.14
- [d4rl](https://github.com/rail-berkeley/d4rl) 1.1 or higher (with v2 datasets)
-
## Usage

For training OAC-BVR on `Envname` (e.g. `walker2d-medium-v2`), run:

```
python main.py --env walker2d-medium-v2
```


