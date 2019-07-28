#pragma once
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <vector>
#include <utility>
#include <Eigen/Dense>
#include "Features.h"

#define BAD_DEPTH -1

/*
	Stereography functions
*/

bool FindFundamentalMatrix(const std::vector<std::pair<Feature, Feature>>& matches, Eigen::Matrix3f& F);

float Triangulate(cv::Point2f& x, cv::Point2f& xprime, const Eigen::Matrix3f E);