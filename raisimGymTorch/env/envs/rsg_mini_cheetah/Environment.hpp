//----------------------------//
// This file is part of RaiSim//
// Copyright 2020, RaiSim Tech//
//----------------------------//

#pragma once

#include <cstdlib>
#include <set>
#include "../../RaisimGymEnv.hpp"

#define PRINT_INFO false

#if PRINT_INFO
#define RSG_INFO(x) if(envIdx==0)std::cout<<x<<std::endl
#else
#define RSG_INFO(x) {}
#endif

namespace raisim {

    class ENVIRONMENT : public RaisimGymEnv {

    public:

        explicit ENVIRONMENT(const std::string &resourceDir, const Yaml::Node &cfg, bool visualizable) :
                RaisimGymEnv(resourceDir, cfg), visualizable_(visualizable), normDist_(0, 1) {

          randEng.seed(rd());
          /// create world
          world_ = std::make_unique<raisim::World>();

          /// add objects
          robot_ = world_->addArticulatedSystem(resourceDir_ + "/mini_cheetah/urdf/mini_cheetah_rsm.urdf");
          robot_->setName("a1");
          robot_->setControlMode(raisim::ControlMode::PD_PLUS_FEEDFORWARD_TORQUE);
          world_->addGround();

          /// get robot data
          gcDim_ = (int)robot_->getGeneralizedCoordinateDim();
          gvDim_ = (int)robot_->getDOF();
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
          gc_init_ << 0, 0, 0.29, 1.0, 0.0, 0.0, 0.0, -0.1, 0.75, -1.6, 0.1, 0.75, -1.6, -0.1, 0.75, -1.6, 0.1, 0.75, -1.6;

          /// set pd gains
          Eigen::VectorXd jointPgain(gvDim_), jointDgain(gvDim_);
          jointPgain.setZero();
          jointPgain.tail(nJoints_).setConstant(50.0);
          jointDgain.setZero();
          jointDgain.tail(nJoints_).setConstant(0.2);
          robot_->setPdGains(jointPgain, jointDgain);
          robot_->setGeneralizedForce(Eigen::VectorXd::Zero(gvDim_));

          bVel_des.setZero();
          bVel_fin.setZero();
          heightTarget_ = gc_init_[2];


          /// MUST BE DONE FOR ALL ENVIRONMENTS
          obDim_ = 38;
          actionDim_ = nJoints_;
          actionMean_.setZero(actionDim_);
          actionStd_.setZero(actionDim_);
          obDouble_.setZero(obDim_);
          obRaw.setZero(obDim_);

          /// action scaling
          actionMean_ = gc_init_.tail(nJoints_);
          actionStd_.setConstant(0.3);

          /// indices of links that should not make contact with ground
          footIndices_.insert(robot_->getBodyIdx("FR_calf"));
          footIndices_.insert(robot_->getBodyIdx("FL_calf"));
          footIndices_.insert(robot_->getBodyIdx("RR_calf"));
          footIndices_.insert(robot_->getBodyIdx("RL_calf"));

          loadConfiguration(cfg);

          /// visualize if it is the first environment
          if (visualizable_) {
            server_ = std::make_unique<raisim::RaisimServer>(world_.get());
            server_->launchServer();
            server_->focusOn(robot_);
          }
        }

        void setCFG(Yaml::Node cfg) final {
          cfg_ = cfg;
          loadConfiguration(cfg);
        }

        void loadConfiguration(const Yaml::Node &cfg) {
          rewards_.initFromCfg(cfg["reward"]);
          setMaxTime(cfg["max_time"].As<double>());

          gaitFreq = cfg["gait"]["step_freq"].As<double>();
          dutyCycle = cfg["gait"]["duty_cycle"].As<double>();
          transRadius = cfg["gait"]["transition_threshold"].As<double>();
          maxVxCmd = cfg["gait"]["vel_x_max"].As<double>();

          gaitType = cfg["gait"]["gait_type"].As<std::string>();
          if (gaitType == gaitSet[0]) gaitOffset << 0.0, 0.5, 0.5, 0.;
          else if (gaitType == gaitSet[1]) gaitOffset << 0.5, 0.5, 0., 0.;
          else if (gaitType == gaitSet[2]) gaitOffset << 0.75, 0.5, 0.25, 0.;
          else RSFATAL("Unimplemented Gait: " + gaitType)

          randDynFlag = cfg["random"]["dynamics"]["enable"].As<bool>();
          randStatFlag = cfg["random"]["state"]["enable"].As<bool>();
          randExtFrcFlag = cfg["random"]["force_ext"]["enable"].As<bool>();
          randVyFlag = cfg["random"]["hori_vel_target"]["enable"].As<bool>();

          if (randDynFlag) {
            inertiaRandRate = cfg["random"]["dynamics"]["link_inertia"].As<double>();
            massRandRate = cfg["random"]["dynamics"]["link_mass"].As<double>();
            comRandRate = cfg["random"]["dynamics"]["link_CoM"].As<double>();
          }
          if (randStatFlag) {
            RSG_INFO("Activating Observation Noise");
            obAmp.setZero(obDim_);
            rpyNoise.setZero();
            obAmp << 0.05,                   /// body height
                    0.02, 0.02, 0.1,         /// body orientation
                    0,0,0,0,0,0,0,0,0,0,0,0, /// joint angles
                    0.1,0.1,0.2,             /// body linear velocity
                    0.3,0.3,0.3,             /// body angular velocity
                    0,0,0,                   /// desired linear velocity
                    0,0,0,0,0,0,0,0,0,0,0,0, /// joint velocity
                    0;                       /// phase indicator
            rpyNoise << 0.15,0.15,0.15;
          }
          if (randExtFrcFlag) {
            RSG_INFO("Activating External Force");
            extFrcPeriod = cfg["random"]["force_ext"]["period"].As<double>();
            extFrcValidTime = cfg["random"]["force_ext"]["effect"].As<double>();
            extFrcRange.setZero();
            extFrcRange[0] = cfg["random"]["force_ext"]["force_x"].As<double>();
            extFrcRange[1] = cfg["random"]["force_ext"]["force_y"].As<double>();
          }

          if(randVyFlag){
            maxVyCmd = cfg["random"]["hori_vel_target"]["vel_y_max"].As<double>();
          }

          std::uniform_real_distribution<double> randNorm(-1, 1);
          if (randDynFlag) {
            double mass;
            raisim::Mat<3, 3> inertia{};
            raisim::Vec<3> COM;
            for (auto &bodyName: robot_->getBodyNames()) {
              raisim::ArticulatedSystem::LinkRef linkTmp = robot_->getLink(bodyName);
              mass = linkTmp.getWeight();
              inertia = linkTmp.getInertia();
              COM = linkTmp.getComPositionInParentFrame();
              mass *= 1 + randNorm(randEng) * massRandRate;
              for (int i = 0; i < 3; i++)
                for (int j = i; j < 3; j++) {
                  inertia(i, j) *= 1 + randNorm(randEng) * inertiaRandRate;
                  inertia(j, i) = inertia(i, j);
                }
              for (int i = 0; i < 3; ++i) COM[i] *= 1 + randNorm(randEng) * comRandRate;
              linkTmp.setWeight(mass);
              linkTmp.setInertia(inertia);
              linkTmp.setComPositionInParentFrame(COM);
              RSG_INFO("Changed Mass to: "<<mass);
            }
          }
        }

        void init() final {}

        void reset() final {
          robot_->setState(gc_init_, gv_init_);
          bVel_fin.setZero();
          if (randomVelFlag) {
            std::uniform_real_distribution<double> randNorm(0, 1);
            double spdDec = randNorm(randEng);
            if (spdDec < 0.1) bVel_fin[0] = 0.0;
            else if (spdDec < 0.3) bVel_fin[0] = maxVxCmd / 3.;
            else if (spdDec < 0.6) bVel_fin[0] = maxVxCmd * 2 / 3;
            else bVel_fin[0] = maxVxCmd;
            RSG_INFO("Randomized Vx: "<<bVel_fin[0]<<"using "<<spdDec);
          }
          if (randVyFlag){
            std::uniform_real_distribution<double> randNorm(0, 1);
            double spdDec = randNorm(randEng);
            if (spdDec < 0.2) bVel_fin[1] = maxVyCmd;
            else if (spdDec < 0.4) bVel_fin[1] = maxVyCmd / 2.;
            else if (spdDec < 0.6) bVel_fin[1] = 0.;
            else if (spdDec < 0.8) bVel_fin[1] = -maxVyCmd / 2.;
            else bVel_fin[1] = -maxVyCmd;
            RSG_INFO("Randomized Vy: "<<bVel_fin[1]<<"using "<<spdDec);
          }
          if(randExtFrcFlag){
            robot_->clearExternalForcesAndTorques();
          }
          bVel_des.setZero();
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


          if(randExtFrcFlag) {
            if (extFrcCnt <= 0) {
              extFrcCnt = extFrcPeriod / control_dt_;    // cnt for disturbing force period
              extFrcLastFlag = false;
            }
            extFrcCnt -= 1;
            // generate disturbing force when enter disturbing time (distPeriod * distRatio)
            // disturbing time is right-aligned with the period
            if (extFrcCnt <= extFrcValidTime / control_dt_ && !extFrcLastFlag) {
              extFrcLastFlag = true;
              pushPos.setZero();
              extFrc.setZero();

              std::uniform_real_distribution<double> randNorm(-1, 1);
              pushPos << randNorm(randEng) * 0.133, randNorm(randEng) * 0.097, randNorm(randEng) * 0.057;
              extFrc << randNorm(randEng) * extFrcRange[0], randNorm(randEng) * extFrcRange[1], 0.;
              RSG_INFO("Applying force: "<<extFrc.transpose());
            }
          }

          robot_->setPdTarget(pTarget_, vTarget_);
          for (int i = 0; i < int(control_dt_ / simulation_dt_ + 1e-10); i++) {
            if (randExtFrcFlag && extFrcLastFlag)
              robot_->setExternalForce(0, pushPos, extFrc);
            else
              robot_->clearExternalForcesAndTorques();
            if (server_) server_->lockVisualizationServerMutex();
            world_->integrate();
            if (server_) server_->unlockVisualizationServerMutex();
          }

          updatePhaseIndicator();
          updateObservation();

          rewards_.record("torq", robot_->getGeneralizedForce().norm());
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
          robot_->getState(gc_, gv_);
          raisim::Vec<4> quat;
          raisim::Mat<3, 3> rot{};
          quat[0] = gc_[3];
          quat[1] = gc_[4];
          quat[2] = gc_[5];
          quat[3] = gc_[6];
          raisim::quatToRotMat(quat, rot);
          bodyLinearVel_ = rot.e().transpose() * gv_.segment(0, 3);
          bodyAngularVel_ = rot.e().transpose() * gv_.segment(3, 3);
          if (abs(bVel_des[0] - bVel_fin[0]) > 1e-3) bVel_des[0] += (bVel_fin - bVel_des).cwiseSign()[0] * accMax;
          if (abs(bVel_des[1] - bVel_fin[1]) > 1e-3) bVel_des[1] += (bVel_fin - bVel_des).cwiseSign()[1] * accMax/3;

          obRaw << gc_[2],                    /// body height
                  rot.e().row(2).transpose(),       /// body orientation
                  gc_.tail(12),                    /// joint angles
                  bodyLinearVel_, bodyAngularVel_,    /// body linear&angular velocity (in base frame, IMU measures in baseFrame)
                  rot.e().transpose() * bVel_des,     /// desired linear velocity in base frame
                  gv_.tail(12),                    /// joint velocity
                  phase;
          if (randStatFlag){
            std::uniform_real_distribution<double> randNorm(-1, 1);
            double eul[3] = {0};
            double quatArr[4] = {0};
            raisim::Vec<4> quat;
            raisim::Vec<3> rpyVec;
            raisim::Mat<3, 3> rotNoisy{};
            for(int i=0;i<4;i++) quatArr[i] = gc_[3+i];
            raisim::quatToEulerVec(quatArr,eul);
            for(int i=0;i<3;i++) rpyVec[i] = roundmod(eul[i] + rpyNoise[i]*randNorm(randEng));
            raisim::rpyToRotMat_extrinsic(rpyVec,rotNoisy);
            for(int i=0;i<obDim_;i++) obDouble_[i] = obRaw[i] + obAmp[i] * randNorm(randEng);
            obDouble_.segment(1,3) = rotNoisy.e().row(2).transpose();
            obDouble_.segment(22,3) = rotNoisy.e().transpose()*bVel_des;
            RSG_INFO("Noised Observation: "<<obDouble_.transpose());
            RSG_INFO("Raw Observation: "<<obRaw.transpose());
          }
          else
            obDouble_ = obRaw;
        }

        void observe(Eigen::Ref<EigenVec> ob) final {
          /// convert it to float
          ob = obDouble_.cast<float>();
        }

        bool isTerminalState(float &terminalReward) final {
          terminalReward = float(terminalRewardCoeff_);

          /// if the contact body is not feet
          for (auto &contact: robot_->getContacts())
            if (footIndices_.find(contact.getlocalBodyIndex()) == footIndices_.end())
              return true;

          terminalReward = 0.f;
          return false;
        }

        void logData() {
          outFile.open(logPath, std::ios::out | std::ios::app);
          if (outFile.is_open()) {
            double bEul[3] = {0};
            double quat[4] = {0};
            for (int i = 0; i < 4; i++) quat[i] = gc_[i+3];
            quatToEulerVec(quat, bEul);
            outFile << world_->getWorldTime();
            for (int i = 0; i < gcDim_; i++) outFile << "," << gc_[i];
            for (int i = 0; i < gvDim_; i++) outFile << "," << gv_[i];
            for (int i = 0; i < obDim_; i++) outFile << "," << obRaw[i];
            for (int i = 0; i < obDim_; i++) outFile << "," << obDouble_[i];
            for (int i = 0; i < 3; i++) outFile << "," << bVel_fin[i];
            for (int i = 0; i < 3; i++) outFile << "," << bVel_des[i];
            for (int i = 0; i < 4; i++) outFile << "," << spdRwdWeight[i];
            for (int i = 0; i < 4; i++) outFile << "," << frcRwdWeight[i];
            for (int i = 0; i < 3; i++) outFile << "," << bEul[i];
            outFile << "," << phase;
            rewards_.logRewards(outFile);
            outFile << "\n";
            outFile.close();
          } else RSWARN("log File Open Failed...")
        }

        void logMetadata(std::string metaPath) override {
          outFile.open(metaPath, std::ios::out | std::ios::app);
          if (outFile.is_open()) {
            outFile << "time";
            for (int i = 0; i < gcDim_; i++) outFile << ",gc_[" << i << "]";
            for (int i = 0; i < gvDim_; i++) outFile << ",gv_[" << i << "]";
            for (int i = 0; i < obDim_; i++) outFile << ",obRaw[" << i << "]";
            for (int i = 0; i < obDim_; i++) outFile << ",obDouble_[" << i << "]";
            for (int i = 0; i < 3; i++) outFile << ",bVel_fin[" << i << "]";
            for (int i = 0; i < 3; i++) outFile << ",bVel_des[" << i << "]";
            for (int i = 0; i < 4; i++) outFile << ",spdRwdWeight[" << i << "]";
            for (int i = 0; i < 4; i++) outFile << ",frcRwdWeight[" << i << "]";
            for (int i = 0; i < 3; i++) outFile << ",baseRPY[" << i << "]";
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
          /// phase<dutyCycle: leg is supposed to be swinging
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
          size_t shankBodyIdxs[4] = {robot_->getBodyIdx("FR_calf"),
                                     robot_->getBodyIdx("FL_calf"),
                                     robot_->getBodyIdx("RR_calf"),
                                     robot_->getBodyIdx("RL_calf")};
          for (auto &contact: robot_->getContacts()) {
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
          size_t footFrameIdxs[4] = {robot_->getFrameIdxByName("FR_foot_fixed"),
                                     robot_->getFrameIdxByName("FL_foot_fixed"),
                                     robot_->getFrameIdxByName("RR_foot_fixed"),
                                     robot_->getFrameIdxByName("RL_foot_fixed")};
          for (int i = 0; i < 4; i++) {
            robot_->getFrameVelocity(footFrameIdxs[i], endVel);
            sum += spdRwdWeight[i] * endVel.e().norm();
          }
          return sum;
        }

        Vec4 getContactFlagVec() {
          Vec4 cnctFlagVec;
          cnctFlagVec.setZero();
          size_t shankBodyIdxs[4] = {robot_->getBodyIdx("FR_calf"),
                                     robot_->getBodyIdx("FL_calf"),
                                     robot_->getBodyIdx("RR_calf"),
                                     robot_->getBodyIdx("RL_calf")};
          for (auto &contact: robot_->getContacts()) {
            if (contact.skip()) continue;
            for (int i = 0; i < 4; i++)
              if (shankBodyIdxs[i] == contact.getlocalBodyIndex())
                cnctFlagVec[i] = 1;
          }
          return cnctFlagVec;
        }

        void setBaseVelTarget(Vec3 velTarg) {
          bVel_fin = velTarg;
          randomVelFlag = false;
        }

        void curriculumUpdate() {};

        double roundmod(double x){
          double ret = x;
          while (ret>M_1_PI) ret-=2* M_2_PI;
          while (ret<-M_1_PI) ret+=2* M_2_PI;
          return ret;
        }


    private:
        int gcDim_, gvDim_, nJoints_;
        bool visualizable_ = false;
        raisim::ArticulatedSystem *robot_;
        Eigen::VectorXd gc_init_, gv_init_, gc_, gv_, pTarget_, pTarget12_, vTarget_;
        double terminalRewardCoeff_ = -200.;
        double heightTarget_ = .3;
        double accMax = 1e-3;

        double phase{}, gaitFreq{}, dutyCycle{}, transRadius = 0.1;
        double maxVxCmd = 0.6, maxVyCmd=0.;
        Vec4 gaitOffset, frcRwdWeight, spdRwdWeight;
        std::string gaitType;
        const std::string gaitSet[3] = {"trot", "bound", "gallop"};


        Vec3 bVel_des, bVel_fin;
        Eigen::VectorXd actionMean_, actionStd_, obDouble_;
        Eigen::Vector3d bodyLinearVel_, bodyAngularVel_;
        std::set<size_t> footIndices_;

        /// these variables are not in use. They are placed to show you how to create a random number sampler.
        std::normal_distribution<double> normDist_;
        thread_local static std::mt19937 gen_;


        double inertiaRandRate = 0, massRandRate = 0., comRandRate = 0.;
        double extFrcPeriod{}, extFrcValidTime{}, extFrcCnt{};
        bool randExtFrcFlag = false, randDynFlag = false, randStatFlag = false, extFrcLastFlag = false, randVyFlag{};

        Eigen::VectorXd obRaw, obNoise, obAmp;
        Vec3 rpyNoise;
        Vec3 extFrcRange;
        Vec3 pushPos, extFrc;
    };

    thread_local std::mt19937 raisim::ENVIRONMENT::gen_;

}

