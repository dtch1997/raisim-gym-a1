## raisim gym env a1

### How to use this repo
This repo is a reformed version of raisimGymTorch, branched from  [raisimLib/raisimGymTorch](https://github.com/raisimTech/raisimlib)

Mainly features:
- Extra interfaces of logging and rendering;
- Interfaces for setting desired velocity;
- Self-defined type of reward kernel transformation;
- Flags and index variables to manipulate robot;
- a minimal A1 locomotion training environment.

### Dependencies
- python3
- [RaiSim](www.raisim.com)>1.002
- pytorch=1.15
- tensorboard
- conda(optional)

### Run
0. Install RaiSim as the instruction in [raisim webpage](https://raisim.com/sections/Installation.html)
1. ```cd``` into repo folder (given that you have cloned it)
2. Compile raisimGym: ```python setup develop```
3. run runner.py of the task (for A1 example): ```python ./raisimGymTorch/env/envs/rsg_a1/runner.py```

* Hyperparameter tuning can be done by adjusting the values in `./raisimGymTorch/env/envs/rsg_a1/cfg.yaml`

* Training results would be stored in `./raisimGymTorch/data/` by default, while video would be saved in `Screenshot` folder beside raisimUnity executable if you are using raisimUnity for rendering.

### Debugging
1. Compile raisimGym with debug symbols: ```python setup develop --Debug```. This compiles <YOUR_APP_NAME>_debug_app
2. Run it with Valgrind. I strongly recommend using Clion

### Continue Training

You can also load pre-train models to accelerate training. Use following command to continue training:
```
python raisimGymTorch/env/envs/rsm_a1/runner.py -m retrain -w <ABSOLUTE/PATH/OF/FULL.PT/FILE>
```

### Policy Evaluation

There's also a script provided to test trained policy without modifying it. Command:
```
python raisimGymTorch/env/envs/rsm_a1/tester.py -w <ABSOLUTE/PATH/OF/FULL.PT/FILE>
```

### Acknowledgement

* [RaiSim](www.raisim.com)