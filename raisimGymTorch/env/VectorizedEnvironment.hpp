//----------------------------//
// This file is part of RaiSim//
// Copyright 2020, RaiSim Tech//
//----------------------------//

#ifndef SRC_RAISIMGYMVECENV_HPP
#define SRC_RAISIMGYMVECENV_HPP

#include "RaisimGymEnv.hpp"
#include "omp.h"
#include "Yaml.hpp"
#include <random>

namespace raisim {

int THREAD_COUNT;

template<class ChildEnvironment>
class VectorizedEnvironment {

 public:

  explicit VectorizedEnvironment(std::string resourceDir, std::string cfg)
      : resourceDir_(resourceDir) {
    Yaml::Parse(cfg_, cfg);
    raisim::World::setActivationKey(raisim::Path(resourceDir + "/activation.raisim").getString());
    srand(time(nullptr));
    if(&cfg_["render"])
      render_ = cfg_["render"].template As<bool>();
  }

  ~VectorizedEnvironment() {
    for (auto *ptr: environments_){
        delete ptr;
    }
  }

  void init() {

    THREAD_COUNT = cfg_["num_threads"].template As<int>();
    omp_set_num_threads(THREAD_COUNT);
    num_envs_ = cfg_["num_envs"].template As<int>();
    refFlag = cfg_["reference"]["enable"].template As<bool>();
    randInitFlag_ = cfg_["random"]["start_time"]["enable"].template As<bool>();
    control_dt_ = cfg_["control_dt"].template As<double>();

    for (int i = 0; i < num_envs_; i++) {
      environments_.push_back(new ChildEnvironment(resourceDir_, cfg_, render_ && i == 0));
      environments_.back()->setSimulationTimeStep(cfg_["simulation_dt"].template As<double>());
      environments_.back()->setControlTimeStep(control_dt_);
      environments_.back()->envIdx = i;
    }

    // setSeed(time(nullptr));
    if(refFlag) {
      refDim_ = cfg_["reference"]["dimension"].template As<int>();
      readRefTraj(resourceDir_ + cfg_["ref_path"].template As<std::string>());
      ref_dt_ = getRefDt();
      nFrames = int(nFrames*ref_dt_/control_dt_);
    }
    for (int i = 0; i < num_envs_; i++) {
      double rand_init = 0.0;
      if(randInitFlag_){
        minInitTime = cfg_["random"]["start_time"]["min"].template As<double>();
        maxInitTime = cfg_["random"]["start_time"]["max"].template As<double>();
        std::uniform_real_distribution<double> randNorm(-1, 1);
        double initTime = randNorm(environments_[i]->randEng) * (maxInitTime - minInitTime) / 2 + (maxInitTime + minInitTime) / 2;
        rand_init = int(initTime/control_dt_)*control_dt_;
      }
      environments_[i]->init();
      environments_[i]->setInitTime(rand_init);
      if(refFlag) environments_[i]->refVec = matchRefTraj(rand_init);
      environments_[i]->reset();
    }

    obDim_ = environments_[0]->getObDim();
    actionDim_ = environments_[0]->getActionDim();
    obSeqLen_ = environments_[0]->getSeqLen();
    RSFATAL_IF(obDim_ == 0 || actionDim_ == 0, "Observation/Action dimension must be defined in the constructor of each environment!")
    
  }

  // resets all environments and returns observation
  void reset() {
    for (auto env: environments_){
      double rand_init = 0.0;
      if(randInitFlag_){
        std::uniform_real_distribution<double> randNorm(-1, 1);
        double initTime = randNorm(env->randEng) * (maxInitTime - minInitTime) / 2 + (maxInitTime + minInitTime) / 2;
        rand_init = int(initTime/control_dt_)*control_dt_;
      }
      env->setInitTime(rand_init);
      if(refFlag) env->refVec = matchRefTraj(rand_init);
      env->reset();
    }
  }

  void observe(Eigen::Ref<EigenRowMajorMat> &ob) {
    ob.resize(num_envs_, obDim_);
    for (int i = 0; i < num_envs_; i++)
      environments_[i]->observe(ob.row(i));
    // std::cout<<"obFlat: "<<ob<<std::endl;
  }

  void step(Eigen::Ref<EigenRowMajorMat> &action,
            Eigen::Ref<EigenVec> &reward,
            Eigen::Ref<EigenBoolVec> &done) {
#pragma omp parallel for
    for (int i = 0; i < num_envs_; i++)
      perAgentStep(i, action, reward, done);
  }

  void logMetadata(std::string metaPath){
    environments_[0]->logMetadata(metaPath);
  }
  void startLogging(std::string logPath) {
    environments_[0]->logFlag = 1;
    environments_[0]->logPath = logPath;
  }
  void stopLogging() {environments_[0]->logFlag = 0;}
  void turnOnVisualization() { if(render_) environments_[0]->turnOnVisualization(); }
  void turnOffVisualization() { if(render_) environments_[0]->turnOffVisualization(); }
  void startRecordingVideo(const std::string& videoName) { if(render_) environments_[0]->startRecordingVideo(videoName); }
  void stopRecordingVideo() { if(render_) environments_[0]->stopRecordingVideo(); }

  void setSeed(int seed) {
    int seed_inc = seed;
    for (auto *env: environments_)
      env->setSeed(seed_inc++);
  }

  void close() {
    for (auto *env: environments_)
      env->close();
  }

  void isTerminalState(Eigen::Ref<EigenBoolVec>& terminalState) {
    for (int i = 0; i < num_envs_; i++) {
      float terminalReward;
      terminalState[i] = environments_[i]->isTerminalState(terminalReward);
    }
  }

  void setCfg(std::string cfg){
    Yaml::Parse(cfg_, cfg);
    for (auto *env: environments_) env->setCFG(cfg_);
  }

  void setBaseVelTarget(int idx, Vec3 velTarg){
    RSFATAL_IF(idx>num_envs_, "envIdx Out of Bound when setting velocity")
    environments_[idx]->setBaseVelTarget(velTarg);
  }

  Vec3 getBaseVelTarget(int idx){
    RSFATAL_IF(idx>num_envs_, "envIdx Out of Bound when setting velocity")
    return environments_[idx]->getBaseVelTarget();
  }

  void setSimulationTimeStep(double dt) {
    for (auto *env: environments_)
      env->setSimulationTimeStep(dt);
  }

  void setControlTimeStep(double dt) {
    for (auto *env: environments_)
      env->setControlTimeStep(dt);
  }

  int getObDim() { return obDim_; }
  int getActionDim() { return actionDim_; }
  int getNumOfEnvs() { return num_envs_; }

  ////// optional methods //////

  int getSeqLen() { return obSeqLen_; }

  void setMaxTime(double maxtime){ 
    for (auto *env: environments_)
      env->setMaxTime(maxtime);
  }
  void curriculumUpdate() {
    for (auto *env: environments_)
      env->curriculumUpdate();
  };

 private:

  inline void perAgentStep(int agentId,
                           Eigen::Ref<EigenRowMajorMat> &action,
                           Eigen::Ref<EigenVec> &reward,
                           Eigen::Ref<EigenBoolVec> &done) {

    if(refFlag) environments_[agentId]->refVec = matchRefTraj(environments_[agentId]->curTime);
    reward[agentId] = environments_[agentId]->step(action.row(agentId));

    float terminalReward = 0;
    done[agentId] = environments_[agentId]->isTerminalState(terminalReward);

    if (done[agentId]) {
      double rand_init = 0.0;
      if(randInitFlag_){
        std::uniform_real_distribution<double> randNorm(-1, 1);
        double initTime = randNorm(environments_[agentId]->randEng) * (maxInitTime - minInitTime) / 2
                          + (maxInitTime + minInitTime) / 2;
        rand_init = int(initTime/control_dt_)*control_dt_;
      }
      environments_[agentId]->setInitTime(rand_init);
      if(refFlag) environments_[agentId]->refVec = matchRefTraj(rand_init);
      environments_[agentId]->reset();
      reward[agentId] += terminalReward;
    }
  }


  Eigen::VectorXd matchRefTraj(float phaseTime) {
    phaseTime = fmod(phaseTime, nFrames * ref_dt_);
    float idx = phaseTime / ref_dt_;
    int idx2 = int(idx) + 1;
    if(idx2==nFrames) idx2 = 0;
    return (int(idx)+1-idx) * refTrajMat.row(int(idx)) + (idx-int(idx)) * refTrajMat.row(idx2);
  }

  void readRefTraj(std::string path) {
    refTrajFile.open(path);
    refTrajMat.conservativeResize(nFrames,refDim_);
    if(refTrajFile.good()) std::cout << "Start Loading File...";
    else std::cout << "Traj File not Found!";
      
    refTrajFile.seekg(0);
    while(refTrajFile.good()){
      refTrajMat.conservativeResize(nFrames,refDim_);
      for(int i = 0; i < refDim_; i++)
      {
        if(i==refDim_-1) getline(refTrajFile,tempValue,'\n');
        else             getline(refTrajFile,tempValue,',' );
        if(tempValue.length()>0) refTrajMat(nFrames-1, i) = std::stod(tempValue);
      }
      nFrames++;
    }
    nFrames--;
    if(nFrames>0) std::cout << nFrames << "Frames has been Loaded.\n";
    //nFrames is the exact numbers of frames in  csv file.
  }

  double getRefDt(){
    return refTrajMat(1,0)-refTrajMat(0,0);
  }

  std::vector<ChildEnvironment *> environments_;

  int num_envs_ = 1, nFrames = 1;
  int obDim_ = 0, actionDim_ = 0, obSeqLen_ = 1, refDim_ = 0;
  bool recordVideo_=false, render_=false, randInitFlag_=false, refFlag = false;
  double control_dt_ = 0.0, ref_dt_ = 0.001;
  double minInitTime = 0.0, maxInitTime = 0.0;
  std::string resourceDir_,tempValue;
  Yaml::Node cfg_;
  Eigen::MatrixXd refTrajMat;
  std::ifstream refTrajFile;
};

class NormalDistribution {
 public:
  NormalDistribution() : normDist_(0.f, 1.f) {}

  float sample() { return normDist_(gen_); }
  void seed(int i) { gen_.seed(i); }

 private:
  std::normal_distribution<float> normDist_;
  static thread_local std::mt19937 gen_;
};
thread_local std::mt19937 raisim::NormalDistribution::gen_;


class NormalSampler {
 public:
  NormalSampler(int dim) {
    dim_ = dim;
    normal_.resize(THREAD_COUNT);
    seed(0);
  }

  void seed(int seed) {
    // this ensures that every thread gets a different seed
#pragma omp parallel for schedule(static, 1)
    for (int i = 0; i < THREAD_COUNT; i++)
      normal_[0].seed(i + seed);
  }

  inline void sample(Eigen::Ref<EigenRowMajorMat> &mean,
                     Eigen::Ref<EigenVec> &std,
                     Eigen::Ref<EigenRowMajorMat> &samples,
                     Eigen::Ref<EigenVec> &log_prob) {
    int agentNumber = log_prob.rows();

#pragma omp parallel for schedule(auto)
    for (int agentId = 0; agentId < agentNumber; agentId++) {
      log_prob(agentId) = 0;
      for (int i = 0; i < dim_; i++) {
        const float noise = normal_[omp_get_thread_num()].sample();
        samples(agentId, i) = mean(agentId, i) + noise * std(i);
        log_prob(agentId) -= noise * noise * 0.5 + std::log(std(i));
      }
      log_prob(agentId) -= float(dim_) * 0.9189385332f;
    }
  }
  int dim_;
  std::vector<NormalDistribution> normal_;
};


}

#endif //SRC_RAISIMGYMVECENV_HPP
