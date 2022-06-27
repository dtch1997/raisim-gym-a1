//
// Created by jemin on 20. 9. 22..
//

#ifndef _RAISIM_GYM_TORCH_RAISIMGYMTORCH_ENV_REWARD_HPP_
#define _RAISIM_GYM_TORCH_RAISIMGYMTORCH_ENV_REWARD_HPP_

#include <initializer_list>
#include <string>
#include <fstream>
#include <map>
#include <math.h>
#include "Yaml.hpp"
#include "Common.hpp"


namespace raisim {

  struct RewardElement {
    float coefficient;
    float decay;
    float reward;
    float integral;
    std::string type;
  };

  class Reward {
  public:
    Reward(std::initializer_list<std::string> names) {
      for (auto &nm: names)
        rewards_[nm] = raisim::RewardElement();
    }

    Reward() = default;

    void initFromCfg(const Yaml::Node &cfg) {
      // std::cout<<"cfg File has the size of: "<<cfg.Size()<<std::endl;
      for (auto rw = cfg.Begin(); rw != cfg.End(); rw++) {
        rewards_[(*rw).first] = raisim::RewardElement();
        rewards_[(*rw).first].coefficient = (*rw).second["coeff"].template As<float>();
        rewards_[(*rw).first].decay = (*rw).second["decay"].template As<float>();
        rewards_[(*rw).first].type = (*rw).second["type"].template As<std::string>();
//        std::cout<<"Name； "<<(*rw).first<<"c: "<<rewards_[(*rw).first].coefficient<<"d: "<<rewards_[(*rw).first].decay<<std::endl;
      }
    }

    const float &operator[](const std::string &name) {
      return rewards_[name].reward;
    }

    void record(const std::string &name, double reward, double targ = 0., bool accumulate = false) {
      RSFATAL_IF(rewards_.find(name) == rewards_.end(), name << " was not found in the configuration file");
      RSFATAL_IF(isnan(reward), name << " is nan");

      if (!accumulate) rewards_[name].reward = 0.f;
      if (rewards_[name].type == "linear") rewards_[name].reward += -reward * rewards_[name].coefficient;
      else if (rewards_[name].type == "expMSE")
        rewards_[name].reward += expMSE(name, reward - targ) * rewards_[name].coefficient;
      else if (rewards_[name].type == "kernel")
        rewards_[name].reward += kernel(name, reward, targ) * rewards_[name].coefficient;
      else RSFATAL(name << " type unknown..");
      rewards_[name].integral += rewards_[name].reward;
    }

    void record(const std::string &name, Vec3 val, Vec3 targ=Eigen::Vector3d::Zero() , bool accumulate = false) {
      RSFATAL_IF(rewards_.find(name) == rewards_.end(), name << " was not found in the configuration file");
      RSFATAL_IF(val.hasNaN(), name << " has nan");

      if (!accumulate) rewards_[name].reward = 0.f;
      if (rewards_[name].type == "linear") rewards_[name].reward += -sqrt(val.squaredNorm()) * rewards_[name].coefficient;
      else if (rewards_[name].type == "expMSE")
        rewards_[name].reward += exp(-rewards_[name].decay * (val - targ).squaredNorm()) * rewards_[name].coefficient;
      else RSFATAL(name << " type unknown..");
      rewards_[name].integral += rewards_[name].reward;
    }

    float sum() {
      float sum = 0.f;
      for (auto &rw: rewards_)
        sum += rw.second.reward;
      return sum;
    }

    float kernel(const std::string &name, float val, float targ = 0.) {
      RSFATAL_IF(rewards_.find(name) == rewards_.end(), name << " was not found in the configuration file");
      return exp(-rewards_[name].decay * (val - targ) * (val - targ));
    }

    float expMSE(const std::string &name, float val) {
      RSFATAL_IF(rewards_.find(name) == rewards_.end(), name << " was not found in the configuration file");
      return exp(-rewards_[name].decay * val * val);
    }

    void displayRewards(float time) {
      std::cout << "----------------------------------------" << std::endl;
      std::cout << "Time: " << time << std::endl;
      for (auto &rw: rewards_)
        std::cout << rw.first << ":" << rw.second.reward << std::endl;
      std::cout << "----------------------------------------" << std::endl;
    }

    void logRewards(std::ofstream &out) {
      for (auto &rw: rewards_)
        out << "," << rw.second.reward;
    }

    void logMetadata(std::ofstream &out) {
      for (auto &rw: rewards_)
        out << "," << rw.first;
    }

    void setZero() {
      for (auto &rw: rewards_)
        rw.second.reward = 0.f;
    }

    void reset() {
      for (auto &rw: rewards_) {
        rw.second.integral = 0.f;
        rw.second.reward = 0.f;
      }
    }

    const std::map<std::string, float> &getStdMapOfRewardIntegral() {
      for (auto &rw: rewards_)
        costSum_[rw.first] = rw.second.integral;

      return costSum_;
    }

  private:
    std::map<std::string, raisim::RewardElement> rewards_;
    std::map<std::string, float> costSum_;
  };

}

#endif //_RAISIM_GYM_TORCH_RAISIMGYMTORCH_ENV_REWARD_HPP_
