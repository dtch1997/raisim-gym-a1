//----------------------------//
// This file is part of RaiSim//
// Copyright 2020, RaiSim Tech//
//----------------------------//

#ifndef SRC_RAISIMGYMENV_HPP
#define SRC_RAISIMGYMENV_HPP

#include <vector>
#include <memory>
#include <unordered_map>
#include <Eigen/Core>
#include "raisim/World.hpp"
#include "raisim/RaisimServer.hpp"
#include "Yaml.hpp"
#include "Reward.hpp"

#define __RSG_MAKE_STR(x) #x
#define _RSG_MAKE_STR(x) __RSG_MAKE_STR(x)
#define RSG_MAKE_STR(x) _RSG_MAKE_STR(x)

#define READ_YAML(a, b, c) RSFATAL_IF(!&c, "Node "<<RSG_MAKE_STR(c)<<" doesn't exist") b = c.template As<a>();

namespace raisim {

class RaisimGymEnv {

 public:
  explicit RaisimGymEnv (std::string resourceDir, const Yaml::Node& cfg) : resourceDir_(std::move(resourceDir)), cfg_(cfg) {
    world_ = std::make_unique<raisim::World>();
  }

  virtual ~RaisimGymEnv() { close(); };

  /// methods to implement
  virtual void init() = 0;
  virtual void reset() = 0;
  virtual void observe(Eigen::Ref<EigenVec> ob) = 0;
  virtual float step(const Eigen::Ref<EigenVec>& action) = 0;
  virtual bool isTerminalState(float& terminalReward) = 0;

  /// optional methods
  virtual void curriculumUpdate() {};
  virtual void close() { if(server_) server_->killServer(); };
  virtual void setSeed(int seed) {};
  virtual void logMetadata(std::string metaPath) {};
  virtual void setCFG(Yaml::Node cfg) {};

  /// Assitive Methods
  void setInitTime(double t) { initTime_ = t; }
  void setMaxTime(double t) { max_time = t; }
  void setSimulationTimeStep(double dt) { simulation_dt_ = dt; world_->setTimeStep(dt); }
  void setControlTimeStep(double dt) { control_dt_ = dt; }
  void setBaseVelTarget(Vec3 velTarg) { bVel_fin = velTarg; }
  int getObDim() const { return obDim_; }
  int getActionDim() const { return actionDim_; }
  int getRefDim() const { return refDim_; }
  int getSeqLen() const { return seqLen; }
  double getControlTimeStep() const { return control_dt_; }
  double getSimulationTimeStep() const { return simulation_dt_; }
  raisim::World* getWorld() { return world_.get(); }
  void turnOffVisualization() { server_->hibernate(); }
  void turnOnVisualization() { server_->wakeup(); }
  void startRecordingVideo(const std::string& videoName ) { server_->startRecordingVideo(videoName); }
  void stopRecordingVideo() { server_->stopRecordingVideo(); }

  template <typename T>
  int sign (const T &val) { return (val > 0) - (val < 0); }
  Vec3 Quat2EulerXYZ(Eigen::Vector4d quat)
  {
    Eigen::Vector3d eul;
    eul.setZero();
    if(quat.tail(3).squaredNorm()>1E-12)
    {
      eul[1] = std::asin(2*(quat[0]*quat[2]+quat[1]*quat[3]));
      eul[0] = std::atan2(2*(quat[0]*quat[1]-quat[2]*quat[3]),(1-2*(quat[1]*quat[1]+quat[2]*quat[2])));
      eul[2] = std::atan2(2*(quat[0]*quat[3]-quat[2]*quat[1]),(1-2*(quat[2]*quat[2]+quat[3]*quat[3])));
    }
    return eul;
  }

  Vec3 RotMat2EulerXYZ(Mat3 R)
  {
    Eigen::Vector3d eul;
    eul.setZero();
    eul[0] = std::atan2(R(2,1),R(2,2));
    eul[1] = std::atan2(-R(2,0),std::sqrt(R(2,1)*R(2,1)+R(2,2)*R(2,2)));
    eul[2] = std::atan2(R(1,0),R(0,0));
    return eul;
  }

 public:
  int envIdx = 0;
  int seqLen = 1;
  bool logFlag = false, refFlag = false;
  double max_time=1.0, curTime = 0;
  Vec3 bVel_fin;
  Eigen::VectorXd refVec;
  std::string logPath;
  std::default_random_engine randEng;
 protected:
  double simulation_dt_ = 0.001;
  double control_dt_ = 0.01;
  double initTime_ = 0.0;
  raisim::Reward rewards_;
  std::string resourceDir_;
  std::ofstream outFile;
  Yaml::Node cfg_;
  int obDim_=0, actionDim_=0,refDim_=0;
  std::unique_ptr<raisim::World> world_;
  std::unique_ptr<raisim::RaisimServer> server_;
  std::random_device rd;
};

}

#endif //SRC_RAISIMGYMENV_HPP
