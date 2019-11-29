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

bool DecomposeEssentialMatrix(
	_In_ Eigen::Matrix3f& E,
	_Out_ Eigen::Matrix3f& R1,
	_Out_ Eigen::Matrix3f& R2,
	_Out_ Eigen::Vector3f& t);

void ComputeRectificationRotations(
	_In_ Eigen::Matrix3f& E,
	_In_ const cv::Mat& img0,
	_In_ const cv::Mat& img1,
	_Out_ Eigen::Matrix3f& R_0,
	_Out_ Eigen::Matrix3f& R_1);

void RectifyImage(
	_In_ const cv::Mat& original,
	_Out_ cv::Mat& rectified,
	_In_ const Eigen::Matrix3f& H);

cv::Mat ComputeDepthImage(
	_In_ const cv::Mat& img0,
	_In_ const cv::Mat& img1);

void ReadCalibrationMatricesFromFile(_In_ const std::string& calibFile, _Inout_ std::vector<ImageDescriptor>& images);