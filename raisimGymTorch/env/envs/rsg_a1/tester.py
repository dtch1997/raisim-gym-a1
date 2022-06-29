from ruamel.yaml import YAML, dump, RoundTripDumper
from raisimGymTorch.env.bin import rsg_a1
from raisimGymTorch.env.RaisimGymVecEnv import RaisimGymVecEnv as VecEnv
import raisimGymTorch.algo.ppo.module as ppo_module
import os
import math
import time
import torch
import numpy as np
from datetime import datetime
import argparse

# configuration
parser = argparse.ArgumentParser()
parser.add_argument('-w', '--weight', help='trained weight path', type=str, default='')
args = parser.parse_args()

# directories
task_path = os.path.dirname(os.path.realpath(__file__))
home_path = task_path + "/../../../../.."

# config
cfg = YAML().load(open(task_path + "/cfg.yaml", 'r'))

# create environment from the configuration file
cfg['environment']['num_envs'] = 1

env = VecEnv(rsg_a1.RaisimGymEnv(home_path + "/rsc", dump(cfg['environment'], Dumper=RoundTripDumper)),
             cfg['environment'])

# shortcuts
ob_dim = env.num_obs
act_dim = env.num_acts

weight_path = args.weight
weight_dir = weight_path.rsplit('/', 1)[0] + '/'
experiment_dir = weight_path.rsplit('/', 1)[0].rsplit('/', 1)[0] + '/'
iteration_number = weight_path.rsplit('/', 1)[0].rsplit('/', 1)[1]

print(weight_path, weight_dir, iteration_number)

if weight_path == "":
    print("Can't find trained weight, please provide a trained weight with --weight switch\n")
else:
    print("Loaded weight from {}\n".format(weight_path))
    start = time.time()
    env.set_vel_target(0, np.array([0.5, 0, 0]))
    env.reset()
    reward_ll_sum = 0
    done_sum = 0
    average_dones = 0.
    n_steps = math.floor(cfg['environment']['max_time'] / cfg['environment']['control_dt'])
    total_steps = n_steps * 1
    start_step_id = 0

    print("Visualizing and evaluating the policy: ", weight_path)
    loaded_graph = ppo_module.MLP(cfg['architecture']['policy_net'], torch.nn.LeakyReLU, ob_dim, act_dim)
    loaded_graph.load_state_dict(torch.load(weight_path)['actor_architecture_state_dict'])

    env.load_scaling(experiment_dir, int(iteration_number))
    env.turn_on_visualization()
    env.start_video_recording(f"{weight_dir}/test_" + datetime.now().strftime("%d%m%y_%H%M%S") + ".mp4")
    env.start_logging(f"{weight_dir}/test_log_" + datetime.now().strftime("%d%m%y_%H%M%S") + ".csv")

    # max_steps = 1000000
    max_steps = 1000  # 10 secs

    observeList = []
    actionList = []
    for step in range(max_steps):
        time.sleep(0.01)
        obs = env.observe(False)
        action_ll = loaded_graph.architecture(torch.from_numpy(obs).cpu())
        reward_ll, dones = env.step(action_ll.cpu().detach().numpy())
        reward_ll_sum = reward_ll_sum + reward_ll[0]
        if dones or step == max_steps - 1:
            print('\n----------------------------------------------------')
            print('{:<40} {:>6}'.format("average ll reward: ", '{:0.10f}'.format(reward_ll_sum)))
            print('{:<40} {:>6}'.format("time elapsed [sec]: ", '{:6.4f}'.format((step + 1 - start_step_id) * 0.01)))
            start_step_id = step + 1
            reward_ll_sum = 0.0
            env.set_vel_target(0, np.array([0.5, 0, 0]))

    env.turn_off_visualization()
    env.stop_logging()
    env.reset()
    print("Finished at the maximum visualization steps")
