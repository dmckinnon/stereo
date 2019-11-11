#pragma once
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <vector>
#include <utility>
#include <Eigen/Dense>
#include "Features.h"

#define BAD_DEPTH -1

#define FUNDAMENTAL_REPROJECTION_ERROR_THRESHOLD 70
#define MIN_NUM_INLIERS 20
#define FUNDAMENTAL_RANSAC_ITERATIONS 200

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

bool FindFundamentalMatrixWithRANSAC(const std::vector<std::pair<Feature, Feature>>& matches, Eigen::Matrix3f& F, StereoPair& stereo);

bool Triangulate(float& depth0, float& depth1, Eigen::Vector3f& x, Eigen::Vector3f& xprime, Eigen::Matrix3f& E);

void DecomposeProjectiveMatrixIntoKAndE(const Eigen::MatrixXf& P, Eigen::Matrix3f& K, Eigen::Matrix3f& E);

bool DecomposeEssentialMatrix(Eigen::Matrix3f& E, Eigen::Matrix3f& R, Eigen::Vector3f& t);

void ReadCalibrationMatricesFromFile(_In_ const std::string& calibFile, _Inout_ std::vector<ImageDescriptor>& images);