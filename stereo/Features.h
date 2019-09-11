#pragma once

#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <Eigen/Dense>
#include <Eigen/Core>
#include <vector>
#include <utility>
#include <fstream>
#include <iostream>

// Parameters to tune
#define FAST_THRESHOLD 30
#define ST_THRESH 10000.f
#define HARRIS_THRESH 10000.f
#define NMS_WINDOW 2
#define MAX_NUM_FEATURES 100
#define MATCH_THRESHOLD 0.1f

// Other parameters
#define HARRIS_WINDOW 5
#define HARRIS_CONSTANT 0.05f //https://courses.cs.washington.edu/courses/cse576/06sp/notes/HarrisDetector.pdf
#define ST_WINDOW 3
#define FAST_SPACING 3
#define ANGLE_WINDOW 9
#define ORIENTATION_HIST_BINS 36
#define DESC_BINS 8
#define DESC_BIN_SIZE 45
#define DESC_WINDOW 16
#define DESC_SUB_WINDOW 4
#define ILLUMINANCE_BOUND 0.2f
#define NN_RATIO 0.7

// DOH constants
#define DOH_WINDOW 11
#define SCALE_SPACE_ITERATIONS 11
#define DOH_THRESHOLD 10000000.0


#define PI 3.14159f
#define RAD2DEG(A) (A*180.f/PI)
#define DEG2RAD(A) (A*PI/180.f)

#define DESC_LENGTH 128
struct FeatureDescriptor
{
	float vec[DESC_LENGTH];
};

struct Feature
{
	int scale;
	cv::Point2f p;
	float score;
	float angle;
	FeatureDescriptor desc;
	float distFromBestMatch;
	float depth;

	friend std::ostream& operator << (std::ostream& os, const Feature& f)
	{
		os << f.scale << " " << f.p.x << " " << f.p.y << " " << f.score << " " << f.angle << " " << f.distFromBestMatch << " " << f.depth;
		for (int i = 0; i < DESC_LENGTH; ++i)
		{
			os << " " << f.desc.vec[i];
		}
		return os;
	}
	friend std::istream& operator >> (std::istream& is, Feature& f)
	{
		float x = 0;
		float y = 0;
		is >> f.scale >> x >> y >> f.score >> f.angle >> f.distFromBestMatch >> f.depth;
		f.p.x = x;
		f.p.y = y;
		for (int i = 0; i < DESC_LENGTH; ++i)
		{
			is >> f.desc.vec[i];
		}
		return is;
	}
};

struct ImageDescriptor
{
	std::vector<Feature> features;
	std::string filename;
	Eigen::Matrix3f K;
	Eigen::Matrix3f E;

	friend std::ostream& operator << (std::ostream& os, const ImageDescriptor& i)
	{
		os << i.filename << std::endl;
		for (int k = 0; k < 3; ++k)
		{
			for (int j = 0; j < 3; ++j)
				os << i.K(k, j) << " ";
			os << std::endl;
		}
		for (int k = 0; k < 3; ++k)
		{
			for (int j = 0; j < 3; ++j)
				os << i.E(k, j) << " ";
			os << std::endl;
		}
		os << i.features.size() << std::endl;
		for (auto& f : i.features)
		{
			os << f << " ";
		}
		return os;
	}
	friend std::istream& operator >> (std::istream& is, ImageDescriptor& i)
	{
		size_t numFeatures = 0;
		is >> i.filename;
		
		for (int k = 0; k < 3; ++k)
		{
			for (int j = 0; j < 3; ++j)
				is >> i.K(k, j);
		}
		for (int k = 0; k < 3; ++k)
		{
			for (int j = 0; j < 3; ++j)
				is >> i.E(k, j);
		}
		is >> numFeatures;
		for (size_t idx = 0; idx < numFeatures; ++idx)
		{
			Feature f;
			is >> f;
			i.features.push_back(f);
		}
		return is;
	}
};

/*
	Equality of features
	Two features are equal if their descriptors are equal
*/
inline bool operator==(const Feature& a, const Feature& b)
{
	for (int i = 0; i < DESC_LENGTH; ++i)
	{
		if (a.desc.vec[i] != b.desc.vec[i])
		{
			return false;
		}
	}
	return true;
}

// Feature comparator
bool FeatureCompare(Feature a, Feature b);

/*
	Feature Detection functions
*/
bool FindFASTFeatures(cv::Mat img, std::vector<Feature>& features);

bool FindDoHFeatures(cv::Mat input, cv::Mat mask, std::vector<Feature>& features);

std::vector<Feature> ClusterFeatures(std::vector<Feature>& features, const int windowSize);

std::vector<Feature> FindHarrisCorners(const cv::Mat& img, int nmsWindowSize);

std::vector<Feature> ScoreAndClusterFeatures(cv::Mat img, std::vector<Feature>& features);

bool CreateSIFTDescriptors(cv::Mat img, std::vector<Feature>& features, std::vector<FeatureDescriptor>& descriptors);

std::vector<std::pair<Feature, Feature> > MatchDescriptors(std::vector<Feature> list1, std::vector<Feature> list2);

void GetImageDescriptorsForFile(const std::vector<std::string>& filenames, const std::string& folder, std::vector<ImageDescriptor>& images, const std::vector<Eigen::MatrixXf>& calibrationMatrices, const cv::Mat& mask);

bool SaveImageDescriptorsToFile(const std::string& filename, std::vector<ImageDescriptor>& images);

bool ReadDescriptorsFromFile(const std::string& filename, std::vector<ImageDescriptor>& images);

/*
	Feature Detection Unit Test functions
*/
void TestSequential12(void);