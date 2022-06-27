//
// Created by jemin on 3/1/22.
//

#ifndef RAISIM_RAISIMGYMTORCH_RAISIMGYMTORCH_ENV_ENVS_COMMON_H_
#define RAISIM_RAISIMGYMTORCH_RAISIMGYMTORCH_ENV_ENVS_COMMON_H_

#include <Eigen/Core>

using Dtype=float;
using EigenRowMajorMat=Eigen::Matrix<Dtype, -1, -1, Eigen::RowMajor>;
using EigenVec=Eigen::Matrix<Dtype, -1, 1>;
using EigenBoolVec=Eigen::Matrix<bool, -1, 1>;
using EigenSeqVec=Eigen::Matrix<Eigen::Matrix<Dtype, -1, 1>, -1, 1>;
using SeqRMajorMat=Eigen::Matrix<Eigen::Matrix<Dtype, -1, 1>, -1, -1, Eigen::RowMajor>;

typedef Eigen::Matrix<double, 2, 1> Vec2;
typedef Eigen::Matrix<double, 3, 1> Vec3;
typedef Eigen::Matrix<double, 4, 1> Vec4;
typedef Eigen::Matrix<double, 6, 1> Vec6;
typedef Eigen::Matrix<double, 6, 6> Mat6;
typedef Eigen::Matrix<double, 3, 3> Mat3;

extern int threadCount;

#endif //RAISIM_RAISIMGYMTORCH_RAISIMGYMTORCH_ENV_ENVS_COMMON_H_
