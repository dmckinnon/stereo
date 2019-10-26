#pragma once
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <vector>
#include <utility>
#include <Eigen/Dense>
#include "Features.h"

#define BAD_DEPTH -1

struct StereoPair
{
	ImageDescriptor img1;
	ImageDescriptor img2;
	Eigen::Matrix3f F;
	Eigen::Matrix3f E;
};

/*
	Stereography functions
*/

bool FindFundamentalMatrix(const std::vector<std::pair<Feature, Feature>>& matches, Eigen::Matrix3f& F);

bool Triangulate(float& depth0, float& depth1, cv::Point2f& x, cv::Point2f& xprime, const Eigen::Matrix3f E);

void DecomposeProjectiveMatrixIntoKAndE(const Eigen::MatrixXf& P, Eigen::Matrix3f& K, Eigen::Matrix3f& E);

bool DecomposeEssentialMatrix(const Eigen::Matrix3f& E, Eigen::Matrix3f& R, Eigen::Vector3f& t);