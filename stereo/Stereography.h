#pragma once
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <vector>
#include <utility>
#include <Eigen/Dense>
#include "Features.h"

/*
	Stereography functions
*/

bool FindFundamentalMatrix(const std::vector<std::pair<Feature, Feature>>& matches, Eigen::Matrix3f& F);