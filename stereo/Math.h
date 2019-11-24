#pragma once
#include <Eigen/Dense>
#include <vector>

/*
	Mathematical functions
*/

Eigen::Matrix3f SkewSymmetric(_In_ const Eigen::Vector3f& v);
Eigen::Vector3f SO3_log(_In_ const Eigen::Matrix3f& R);
Eigen::Matrix3f SO3_exp(_In_ const Eigen::Vector3f& r);