//----------------------------//
// This file is part of RaiSim//
// Copyright 2020, RaiSim Tech//
//----------------------------//

#pragma once

#include <stdlib.h>
#include <set>
#include "../../RaisimGymEnv.hpp"

namespace raisim {

    class ENVIRONMENT : public RaisimGymEnv {

    public:

        explicit ENVIRONMENT(const std::string &resourceDir, const Yaml::Node &cfg, bool visualizable) :
                RaisimGymEnv(resourceDir, cfg), visualizable_(visualizable), normDist_(0, 1) {

          /// create world
          world_ = std::make_unique<raisim::World>();

          /// add objects
          a1_ = world_->addArticulatedSystem(resourceDir_ + "/a1/urdf/a1.urdf");
          a1_->setName("a1");
          a1_->setControlMode(raisim::ControlMode::PD_PLUS_FEEDFORWARD_TORQUE);
          world_->addGround();

          /// get robot data
          gcDim_ = a1_->getGeneralizedCoordinateDim();
          gvDim_ = a1_->getDOF();
          nJoints_ = gvDim_ - 6;

          /// initialize containers
          gc_.setZero(gcDim_);
          gc_init_.setZero(gcDim_);
          gv_.setZero(gvDim_);
          gv_init_.setZero(gvDim_);
          pTarget_.setZero(gcDim_);
          vTarget_.setZero(gvDim_);
          pTarget12_.setZero(nJoints_);

          /// this is nominal configuration of a1
          gc_init_
                  << 0, 0, 0.29, 1.0, 0.0, 0.0, 0.0, -0.1, 0.75, -1.6, 0.1, 0.75, -1.6, -0.1, 0.75, -1.6, 0.1, 0.75, -1.6;

          /// set pd gains
          Eigen::VectorXd jointPgain(gvDim_), jointDgain(gvDim_);
          jointPgain.setZero();
          jointPgain.tail(nJoints_).setConstant(50.0);
          jointDgain.setZero();
          jointDgain.tail(nJoints_).setConstant(0.2);
          a1_->setPdGains(jointPgain, jointDgain);
          a1_->setGeneralizedForce(Eigen::VectorXd::Zero(gvDim_));

          bVel_des.setZero();
          bVel_fin.setZero();
          heightTarget_ = gc_init_[2];


          gaitFreq = cfg["gait"]["step_freq"].As<double>();
          dutyCycle = cfg["gait"]["duty_cycle"].As<double>();
          transRadius = cfg["gait"]["transition_threshold"].As<double>();
          gaitType = cfg["gait"]["gait_type"].As<std::string>();
          if (gaitType == gaitSet[0]) gaitOffset << 0.0, 0.5, 0.5, 0.;
          else if (gaitType == gaitSet[1]) gaitOffset << 0.5, 0.5, 0., 0.;
          else RSFATAL("Unimplemented Gait: " + gaitType)

          /// MUST BE DONE FOR ALL ENVIRONMENTS
          obDim_ = 38;
          actionDim_ = nJoints_;
          actionMean_.setZero(actionDim_);
          actionStd_.setZero(actionDim_);
          obDouble_.setZero(obDim_);

          /// action scaling
          actionMean_ = gc_init_.tail(nJoints_);
          actionStd_.setConstant(0.3);

          /// Reward coefficients
          rewards_.initFromCfg(cfg["reward"]);

          /// indices of links that should not make contact with ground
          footIndices_.insert(a1_->getBodyIdx("FR_calf"));
          footIndices_.insert(a1_->getBodyIdx("FL_calf"));
          footIndices_.insert(a1_->getBodyIdx("RR_calf"));
          footIndices_.insert(a1_->getBodyIdx("RL_calf"));

          /// visualize if it is the first environment
          if (visualizable_) {
            server_ = std::make_unique<raisim::RaisimServer>(world_.get());
            server_->launchServer();
            server_->focusOn(a1_);
          }
        }

        void init() final {}

        void reset() final {
          a1_->setState(gc_init_, gv_init_);

          std::uniform_real_distribution<double> randNorm(0, 1);
          double spdDec = randNorm(randEng);
          if (spdDec < 0.1) bVel_fin[0] = 0.0;
          else if (spdDec < 0.3) bVel_fin[0] = 0.7;
          else if (spdDec < 0.6) bVel_fin[0] = 1.4;
          else bVel_fin[0] = 2.0;
          bVel_des[0] = 0.;
          accMax = 0.6 * control_dt_;
          world_->setWorldTime(0.0);
          phase = 0;

          updateObservation();
        }

        float step(const Eigen::Ref<EigenVec> &action) final {
          /// action scaling
          pTarget12_ = action.cast<double>();
          pTarget12_ = pTarget12_.cwiseProduct(actionStd_);
          pTarget12_ += actionMean_;
          pTarget_.tail(nJoints_) = pTarget12_;

          /// \todo: add external pushing;

          a1_->setPdTarget(pTarget_, vTarget_);
          for (int i = 0; i < int(control_dt_ / simulation_dt_ + 1e-10); i++) {
            if (server_) server_->lockVisualizationServerMutex();
            world_->integrate();
            if (server_) server_->unlockVisualizationServerMutex();
          }

          updatePhaseIndicator();
          updateObservation();

          rewards_.record("torq", a1_->getGeneralizedForce().norm());
          rewards_.record("bHgt", gc_[2], heightTarget_);
          rewards_.record("bRot", gc_.segment(3, 4).transpose() * gc_init_.segment(3, 4), 1.0);
          rewards_.record("bVel", (gv_.head(3) - bVel_des).norm());
          rewards_.record("eFrc", getContactForceWeightedSum());
          rewards_.record("eVel", getEndVelWeightedSum());


          if (logFlag) logData();

          // if(envIdx==0) rewards_.displayRewards(curTime);

          return rewards_.sum();
        }

        void updateObservation() {
          a1_->getState(gc_, gv_);
          raisim::Vec<4> quat;
          raisim::Mat<3, 3> rot;
          quat[0] = gc_[3];
          quat[1] = gc_[4];
          quat[2] = gc_[5];
          quat[3] = gc_[6];
          raisim::quatToRotMat(quat, rot);
          bodyLinearVel_ = rot.e().transpose() * gv_.segment(0, 3);
          bodyAngularVel_ = rot.e().transpose() * gv_.segment(3, 3);
          if (abs(bVel_des[0] - bVel_fin[0]) > 1e-3) bVel_des[0] += sign(bVel_fin[0] - bVel_des[0]) * accMax;

          /// \todo: add randomizers;

          obDouble_ << gc_[2],                    /// body height
                  rot.e().row(2).transpose(),       /// body orientation
                  gc_.tail(12),                    /// joint angles
                  bodyLinearVel_, bodyAngularVel_,    /// body linear&angular velocity (in base frame, IMU measures in baseFrame)
                  rot.e().transpose() * bVel_des,     /// desired linear velocity in base frame
                  gv_.tail(12),                    /// joint velocity
                  phase;
        }

        void observe(Eigen::Ref<EigenVec> ob) final {
          /// convert it to float
          ob = obDouble_.cast<float>();
        }

        bool isTerminalState(float &terminalReward) final {
          terminalReward = float(terminalRewardCoeff_);

          /// if the contact body is not feet
          for (auto &contact: a1_->getContacts())
            if (footIndices_.find(contact.getlocalBodyIndex()) == footIndices_.end())
              return true;

          terminalReward = 0.f;
          return false;
        }

        void logData() {
          outFile.open(logPath, std::ios::out | std::ios::app);
          if (outFile.is_open()) {
            outFile << world_->getWorldTime();
            for (int i = 0; i < gcDim_; i++) outFile << "," << gc_[i];
            for (int i = 0; i < gvDim_; i++) outFile << "," << gv_[i];
            for (int i = 0; i < obDim_; i++) outFile << "," << obDouble_[i];
            for (int i = 0; i < 4; i++) outFile << "," << spdRwdWeight[i];
            for (int i = 0; i < 4; i++) outFile << "," << frcRwdWeight[i];
            outFile << "," << phase;
            rewards_.logRewards(outFile);
            outFile << "\n";
            outFile.close();
          } else RSWARN("log File Open Failed...")
        }

        void logMetadata(std::string metaPath) {
          outFile.open(metaPath, std::ios::out | std::ios::app);
          if (outFile.is_open()) {
            outFile << "time";
            for (int i = 0; i < gcDim_; i++) outFile << ",gc_[" << i << "]";
            for (int i = 0; i < gvDim_; i++) outFile << ",gv_[" << i << "]";
            for (int i = 0; i < obDim_; i++) outFile << ",obDouble_[" << i << "]";
            for (int i = 0; i < 4; i++) outFile << ",spdRwdWeight[" << i << "]";
            for (int i = 0; i < 4; i++) outFile << ",frcRwdWeight[" << i << "]";
            outFile << ",phase";
            rewards_.logMetadata(outFile);
            outFile << std::endl;
            outFile.close();
            RSINFO("Metadata Merged.")
          } else RSWARN("Metadata File Open Failed...")
        }


        void updatePhaseIndicator() {
          phase += control_dt_ * gaitFreq;
          phase = fmod(phase, 1.0);
          /// phase<dutyCycle: leg is supposed to be at swing state
          for (int i = 0; i < 4; i++) {
            double footPhase = fmod(phase + gaitOffset[i], 1.0);
            if (fabs(footPhase - dutyCycle) < transRadius)
              frcRwdWeight[i] = (dutyCycle - footPhase) / (2 * transRadius) + 0.5;
            else if (footPhase < transRadius) frcRwdWeight[i] = footPhase / (2 * transRadius) + 0.5;
            else if (footPhase > 1 - transRadius) frcRwdWeight[i] = (footPhase - 1.0) / (2 * transRadius) + 0.5;
            else if (footPhase < dutyCycle) frcRwdWeight[i] = 1.0;
            else frcRwdWeight[i] = 0.0;
            spdRwdWeight[i] = 1.0 - frcRwdWeight[i];
          }
        }

        double getContactForceWeightedSum() {
          double sum = 0.;
          Vec3 endFrc;
          size_t shankBodyIdxs[4] = {a1_->getBodyIdx("FR_calf"),
                                     a1_->getBodyIdx("FL_calf"),
                                     a1_->getBodyIdx("RR_calf"),
                                     a1_->getBodyIdx("RL_calf")};
          for (auto &contact: a1_->getContacts()) {
            if (contact.skip()) continue; /// if the contact is internal, one contact point is set to 'skip'
            for (int i = 0; i < 4; i++)
              if (shankBodyIdxs[i] == contact.getlocalBodyIndex())
                sum += frcRwdWeight[i] * contact.getImpulse().e().norm() / simulation_dt_;
          }
          return sum;

        }

        double getEndVelWeightedSum() {
          double sum = 0.;
          raisim::Vec<3> endVel;
          size_t footFrameIdxs[4] = {a1_->getFrameIdxByName("FR_foot_fixed"),
                                     a1_->getFrameIdxByName("FL_foot_fixed"),
                                     a1_->getFrameIdxByName("RR_foot_fixed"),
                                     a1_->getFrameIdxByName("RL_foot_fixed")};
          for (int i = 0; i < 4; i++) {
            a1_->getFrameVelocity(footFrameIdxs[i], endVel);
            sum += spdRwdWeight[i] * endVel.e().norm();
          }
          return sum;
        }

        Vec4 getContactFlagVec() {
          Vec4 cnctFlagVec;
          cnctFlagVec.setZero();
          size_t shankBodyIdxs[4] = {a1_->getBodyIdx("FR_calf"),
                                     a1_->getBodyIdx("FL_calf"),
                                     a1_->getBodyIdx("RR_calf"),
                                     a1_->getBodyIdx("RL_calf")};
          for (auto &contact: a1_->getContacts()) {
            if (contact.skip()) continue;
            for (int i = 0; i < 4; i++)
              if (shankBodyIdxs[i] == contact.getlocalBodyIndex())
                cnctFlagVec[i] = 1;
          }
          return cnctFlagVec;
        }

        /// \todo: add metadata logging;
        void curriculumUpdate() {};


    private:
        int gcDim_, gvDim_, nJoints_;
        bool visualizable_ = false;
        raisim::ArticulatedSystem *a1_;
        Eigen::VectorXd gc_init_, gv_init_, gc_, gv_, pTarget_, pTarget12_, vTarget_;
        double terminalRewardCoeff_ = -200.;
        double heightTarget_ = .3;
        double accMax = 1e-3;

        double phase, gaitFreq, dutyCycle, transRadius = 0.1;
        Vec4 gaitOffset, frcRwdWeight, spdRwdWeight;
        std::string gaitType;
        const std::string gaitSet[2] = {"trot", "bound"};


        Vec3 bVel_des, bVel_fin;
        Eigen::VectorXd actionMean_, actionStd_, obDouble_;
        Eigen::Vector3d bodyLinearVel_, bodyAngularVel_;
        std::set<size_t> footIndices_;

        /// these variables are not in use. They are placed to show you how to create a random number sampler.
        std::normal_distribution<double> normDist_;
        thread_local static std::mt19937 gen_;
    };

    thread_local std::mt19937 raisim::ENVIRONMENT::gen_;

}

