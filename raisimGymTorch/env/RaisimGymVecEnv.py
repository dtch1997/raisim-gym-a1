# //----------------------------//
# // This file is part of RaiSim//
# // Copyright 2020, RaiSim Tech//
# //----------------------------//

import numpy as np


class RaisimGymVecEnv:

    def __init__(self, impl, cfg, normalize_ob=True, seed=0, normalize_rew=True, clip_obs=10.):
        self.normalize_ob = normalize_ob
        self.normalize_rew = normalize_rew
        self.clip_obs = clip_obs
        self.wrapper = impl
        self.wrapper.init()
        self.num_obs = self.wrapper.getObDim()
        self.num_acts = self.wrapper.getActionDim()
        self.seq_len = int(cfg['control_dt'] / cfg['simulation_dt'])
        self._observation = np.zeros([self.num_envs, self.num_obs], dtype=np.float32)
        self.obs_rms = RunningMeanStd(shape=[self.num_obs])
        self._reward = np.zeros(self.num_envs, dtype=np.float32)
        self._done = np.zeros(self.num_envs, dtype=np.bool)
        self.rewards = [[] for _ in range(self.num_envs)]
        self.steps_per_iteration = int(cfg['max_time'] / cfg['control_dt'] * self.num_envs)

    def seed(self, seed=None):
        self.wrapper.setSeed(seed)

    def set_cfg(self, cfg):
        self.wrapper.setCfg(cfg)

    def set_command(self, command):
        self.wrapper.setCommand(command)

    def set_vel_target(self, idx, vel):
        self.wrapper.setBaseVelTarget(idx, vel)

    def get_vel_target(self, idx):
        return self.wrapper.getBaseVelTarget(idx)

    def get_stat_info(self):
        return self.obs_rms.mean, self.obs_rms.var, self.obs_rms.count

    def set_stat_info(self, mean, var, count):
        self.obs_rms.mean = mean
        self.obs_rms.var = var
        self.obs_rms.count = count

    def turn_on_visualization(self):
        self.wrapper.turnOnVisualization()

    def turn_off_visualization(self):
        self.wrapper.turnOffVisualization()

    def start_video_recording(self, file_name):
        self.wrapper.startRecordingVideo(file_name)

    def stop_video_recording(self):
        self.wrapper.stopRecordingVideo()

    def log_matedata(self, file_name):
        self.wrapper.logMetadata(file_name)

    def start_logging(self, file_name):
        self.wrapper.startLogging(file_name)

    def stop_logging(self):
        self.wrapper.stopLogging()

    def set_max_time(self, maxtime):
        self.wrapper.setMaxTime(maxtime)

    def reset_random(self, rand):
        self.wrapper.resetRandom(rand)

    def step(self, action):
        self.wrapper.step(action, self._reward, self._done)
        return self._reward.copy(), self._done.copy()

    def load_scaling(self, dir_name, iteration):
        mean_file_name = f"{dir_name}/mean.csv"
        var_file_name = f"{dir_name}/var.csv"
        self.obs_rms.mean = np.loadtxt(mean_file_name, dtype=np.float32, delimiter=",")
        self.obs_rms.var = np.loadtxt(var_file_name, dtype=np.float32, delimiter=",")
        self.obs_rms.count = self.steps_per_iteration * int(iteration)

    def save_scaling(self, dir_name, iteration):
        mean_file_name = f"{dir_name}/{iteration}/mean.csv"
        var_file_name = f"{dir_name}/{iteration}/var.csv"
        np.savetxt(mean_file_name, self.obs_rms.mean, delimiter=",")
        np.savetxt(var_file_name, self.obs_rms.var, delimiter=",")

    def observe(self, update_mean=True):
        self.wrapper.observe(self._observation)

        if self.normalize_ob:
            if update_mean:
                self.obs_rms.update(self._observation)
            return self._normalize_observation(self._observation)
        else:
            return self._observation.copy()

    def observe_raw(self):
        self.wrapper.observe(self._observation)

        return self._observation.copy()

    def reset(self):
        self._reward = np.zeros(self.num_envs, dtype=np.float32)
        self.wrapper.reset()

    def _normalize_observation(self, obs):
        if self.normalize_ob:
            return np.clip((obs - self.obs_rms.mean) / np.sqrt(self.obs_rms.var + 1e-8), -self.clip_obs, self.clip_obs)
        else:
            return obs

    def reset_and_update_info(self):
        return self.reset(), self._update_epi_info()

    def _update_epi_info(self):
        info = [{} for _ in range(self.num_envs)]

        for i in range(self.num_envs):
            eprew = sum(self.rewards[i])
            eplen = len(self.rewards[i])
            epinfo = {"r": eprew, "l": eplen}
            info[i]['episode'] = epinfo
            self.rewards[i].clear()

        return info

    def close(self):
        self.wrapper.close()

    def curriculum_callback(self):
        self.wrapper.curriculumUpdate()

    @property
    def num_envs(self):
        return self.wrapper.getNumOfEnvs()

    @property
    def extra_info_names(self):
        return self._extraInfoNames


class RunningMeanStd(object):
    def __init__(self, epsilon=1e-4, shape=()):
        """
        calulates the running mean and std of a data stream
        https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm

        :param epsilon: (float) helps with arithmetic issues
        :param shape: (tuple) the shape of the data stream's output
        """
        self.mean = np.zeros(shape, 'float32')
        self.var = np.ones(shape, 'float32')
        self.count = epsilon

    def update(self, arr):
        batch_mean = np.mean(arr, axis=0)
        batch_var = np.var(arr, axis=0)
        batch_count = arr.shape[0]
        # for q in range(batch_count):
        #     print("arr[",q,"]: ", arr[q,:,:])
        # print("mean ", self.mean[0,:])
        # print("bach ", batch_mean[0,:])
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean, batch_var, batch_count):
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m_2 = m_a + m_b + np.square(delta) * (self.count * batch_count / tot_count)
        new_var = m_2 / tot_count

        self.mean = new_mean
        self.var = new_var
        self.count = tot_count
