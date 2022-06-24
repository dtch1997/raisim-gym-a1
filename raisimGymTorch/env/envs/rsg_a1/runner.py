from ruamel.yaml import YAML, dump, RoundTripDumper
from raisimGymTorch.env.bin.rsg_a1 import RaisimGymEnv
from raisimGymTorch.env.bin.rsg_a1 import NormalSampler
from raisimGymTorch.env.RaisimGymVecEnv import RaisimGymVecEnv as VecEnv
from raisimGymTorch.helper.raisim_gym_helper import ConfigurationSaver, load_param, tensorboard_launcher
import os
import math
import time
import raisimGymTorch.algo.ppo.module as ppo_module
import raisimGymTorch.algo.ppo.ppo as PPO
import torch.nn as nn
import numpy as np
import torch
import datetime
import argparse


# task specification
task_name = "a1_locomotion"

# configuration
parser = argparse.ArgumentParser()
parser.add_argument('-m', '--mode', help='set mode either train or test', type=str, default='train')
parser.add_argument('-w', '--weight', help='pre-trained weight path', type=str, default='')
args = parser.parse_args()
mode = args.mode
weight_path = args.weight

# check if gpu is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# directories
task_path = os.path.dirname(os.path.realpath(__file__))
home_path = task_path + "/../../../../.."

# config
cfg = YAML().load(open(task_path + "/cfg.yaml", 'r'))

# create environment from the configuration file
cfgDump = dump(cfg['environment'], Dumper=RoundTripDumper)
impl = RaisimGymEnv(home_path + "/rsc", cfgDump)
env = VecEnv(impl, cfg['environment'])

# shortcuts
ob_dim = env.num_obs
act_dim = env.num_acts
num_threads = cfg['environment']['num_threads']

# Training
n_steps = math.floor(cfg['environment']['max_time'] / cfg['environment']['control_dt'])
total_steps = n_steps * env.num_envs

avg_rewards = []

actor = ppo_module.Actor(ppo_module.MLP(cfg['architecture']['policy_net'], nn.LeakyReLU, ob_dim, act_dim),
                         ppo_module.MultivariateGaussianDiagonalCovariance(act_dim,
                                                                           env.num_envs,
                                                                           1.0,
                                                                           NormalSampler(act_dim),
                                                                           cfg['seed']),
                         device)
critic = ppo_module.Critic(ppo_module.MLP(cfg['architecture']['value_net'], nn.LeakyReLU, ob_dim, 1),
                           device)

saver = ConfigurationSaver(log_dir=home_path + "/rsmGymA1/data/"+task_name,
                           save_items=[task_path + "/cfg.yaml", task_path + "/Environment.hpp"])
tensorboard_launcher(saver.data_dir+"/..")  # press refresh (F5) after the first ppo update

ppo = PPO.PPO(actor=actor,
              critic=critic,
              num_envs=cfg['environment']['num_envs'],
              num_transitions_per_env=n_steps,
              num_learning_epochs=4,
              gamma=0.996,
              lam=0.95,
              num_mini_batches=4,
              device=device,
              log_dir=saver.data_dir,
              shuffle_batch=False,
              )

if mode == 'retrain':
    load_param(weight_path, env, actor, critic, ppo.optimizer, saver.data_dir)

startExec = time.time()
for update in range(1000000):
    if update % cfg['environment']['eval_every_n'] == 0:
        os.mkdir(f'{saver.data_dir}/{str(update)}/')
        print("Visualizing and evaluating the current policy")
        torch.save({
            'actor_architecture_state_dict': actor.architecture.state_dict(),
            'actor_distribution_state_dict': actor.distribution.state_dict(),
            'critic_architecture_state_dict': critic.architecture.state_dict(),
            'optimizer_state_dict': ppo.optimizer.state_dict(),
        },f'{saver.data_dir}/{str(update)}/full.pt')
        env.save_scaling(saver.data_dir, str(update))
        # we create another graph just to demonstrate the save/load method
        loaded_graph = ppo_module.MLP(cfg['architecture']['policy_net'], nn.LeakyReLU, ob_dim, act_dim)
        loaded_graph.load_state_dict(torch.load(f'{saver.data_dir}/{str(update)}/full.pt')['actor_architecture_state_dict'])

        if cfg['environment']['render']:
            env.turn_on_visualization()
            env.start_video_recording(f'{saver.data_dir}/{str(update)}/train_policy.mp4')

        test_steps = n_steps*2 if update>20 else n_steps//2
        observeList = []
        actionList = []

        for step in range(test_steps):
            with torch.no_grad():
                frame_start = time.time()
                obs = env.observe(False)
                action_ll = loaded_graph.architecture(torch.from_numpy(obs).cpu())
                reward_ll, dones = env.step(action_ll.cpu().detach().numpy())
                observeList.append(obs)
                actionList.append(action_ll.cpu().detach().numpy())
                frame_end = time.time()
                wait_time = cfg['environment']['control_dt'] - (frame_end-frame_start)
                if wait_time > 0.:
                    time.sleep(wait_time)
        observeMat = np.stack(observeList)
        actionMat = np.stack(actionList)
        np.save(f'{saver.data_dir}/{str(update)}/observation.npy', observeMat)
        np.save(f'{saver.data_dir}/{str(update)}/action.npy', actionMat)
        if cfg['environment']['render']:
            env.stop_video_recording()
            env.turn_off_visualization()

        env.reset()

    # actual training
    start = time.time()
    env.reset()
    reward_ll_sum = 0
    done_sum = env.num_envs # at least we have (num_envs) trajectories

    for step in range(n_steps):
        obs = env.observe()
        action = ppo.act(obs)
        reward, dones = env.step(action)
        ppo.step(value_obs=obs, rews=reward, dones=dones)
        done_sum = done_sum + np.sum(dones)
        reward_ll_sum = reward_ll_sum + np.sum(reward)

    # take st step to get value obs
    obs = env.observe()

    endGather = time.time()
    ppo.update(actor_obs=obs, value_obs=obs, log_this_iteration=update % 10 == 0, update=update)
    average_ll_performance = reward_ll_sum / total_steps
    average_dones = done_sum / total_steps
    avg_rewards.append(average_ll_performance)

    actor.update()
    actor.distribution.enforce_minimum_std((torch.ones(12)*0.2).to(device))

    # curriculum update. Implement it in Environment.hpp
    env.curriculum_callback()

    endTrain = time.time()

    print(f"Iter: {update},  Avg Rwd: {reward_ll_sum/done_sum:.4f}, Traj Cnt: {done_sum}, " +
          f"Time(Gather-Train): {endGather - start:.2f}-{endTrain - endGather:.2f}s, Tot Time: {endTrain-startExec:6.1f}s")

    ppo.writer.add_scalar('performance/reward_per_traj', reward_ll_sum/done_sum, update)
    ppo.writer.add_scalar('performance/overall term', done_sum, update)
    ppo.writer.add_scalar('performance/fps', total_steps / (endTrain - start), update)
    ppo.writer.add_scalar('performance/time_factor', total_steps / (endTrain - start) * cfg['environment']['control_dt'], update)
